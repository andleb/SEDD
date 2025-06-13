import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, cond, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    """This is the Euler predictor."""

    def update_fn(self, score_fn, x, t, cond, step_size):
        sigma, dsigma = self.noise(t)

        score = score_fn(x, sigma, cond)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, cond, step_size):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    """This is the Tweedie predictor."""

    def update_fn(self, score_fn, x, t, cond, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, cond)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, cond=None):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, cond)
        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]

        # return probs.argmax(dim=-1)
        return sample_categorical(probs)


def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device)

    return sampling_fn


def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x):
    """
    Assuming batch dims to be the image dims if using images/
    """

    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    B, C, H, W = batch_dims

    @torch.no_grad()
    def pc_sampler(model, cond=None):
        # FIXME: how to handle the batch wrt the conditional?
        # cond = cond.view(C*H*W) if cond is not None else None

        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True,
                                                B=B, C=C, H=H, W=W)
        # Now, it's flattened:
        x = graph.sample_limit(B, C*H*W).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, cond, dt)

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t, cond)

        return x.view(B, C, H, W)

    return pc_sampler



class PCSampler:
    """
    Let's make this a class os the attributes can be adjusted.
    """
    def __init__(self, graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5,
                 device=torch.device('cpu'),
                 proj_fun=lambda x: x):

        self.graph = graph
        self.noise = noise

        self.predictor = get_predictor(predictor)(graph, noise)
        self.denoiser = Denoiser(self.graph, self.noise)
        self.projector = proj_fun

        self.steps = steps
        self.denoise = denoise
        self.eps = eps
        self.device = device
        self.proj_fun = proj_fun

        self.B, self.C, self.H, self.W = batch_dims

    @torch.no_grad()
    def __call__(self, model, cond=None):

        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True,
                                                B=self.B, C=self.C, H=self.H,
                                                W=self.W)
        # Now, it's flattened:
        x = self.graph.sample_limit(self.B, self.C*self.H*self.W).to(self.device)
        timesteps = torch.linspace(1, self.eps, self.steps + 1, device=self.device)
        dt = (1 - self.eps) / self.steps

        for i in range(self.steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            x = self.projector(x)
            x = self.predictor.update_fn(sampling_score_fn, x, t, cond, dt)

        if self.denoise:
            # denoising step
            x = self.projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            x = self.denoiser.update_fn(sampling_score_fn, x, t, cond)

        return x.view(self.B, self.C, self.H, self.W)