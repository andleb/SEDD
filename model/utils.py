import torch
import torch.nn.functional as F


def get_model_fn(model, train=False):
    """Create a function to give the output of the score-based model.

    Args:
        model: The score model.
        train: `True` for training and `False` for evaluation.
        mlm: If the input model is a mlm and models the base probability 

    Returns:
        A model function.
    """

    def model_fn(x, sigma, cond):
        """Compute the output of the score-based model.

        Args:
            x: A mini-batch of input data.
            cond: A mini-batch of conditioning variables for time steps. Should be interpreted differently
              for different models.

        Returns:
            A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
            # otherwise output the raw values (we handle mlm training in losses.py)
        return model(x, sigma, cond)

    return model_fn


# TODO: implement unflattening wrapper
def get_score_fn(model, train=False, sampling=False):
    if sampling:
        assert not train, "Must sample in eval mode"
    model_fn = get_model_fn(model, train=train)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        def score_fn(x, sigma, cond):
            sigma = sigma.reshape(-1)
            score = model_fn(x, sigma, cond)
            
            if sampling:
                # when sampling return true score (not log used for training)
                return score.exp()
                
            return score

    return score_fn



def get_score_fn(model, train=False, sampling=False, B, C, H, W):
    """
    Returns a function that:
      1) Unflattens x from [B, L] -> [B, C, H, W]
      2) Unflattens cond if needed
      3) Calls `model(...)` to get [B, vocab, H, W]
      4) Flattens the output -> [B, L, vocab]
    """
    L = C * H * W

    def score_fn(x, sigma, cond=None):
        if sampling:
            assert not train, "Must sample in eval mode"
        model_fn = get_model_fn(model, train=train)

        # x is [B, L] in discrete form
        B_ = x.shape[0]
        assert B_ == B, f"Expected batch size {B} got {B_}"
        assert x.shape[1] == L, f"Expected length {L} got {x.shape[1]}"

        # 1) Unflatten x -> [B, C, H, W]
        x_img = x.view(B_, C, H, W)

        # 2) If cond is also discrete 2D, unflatten it, or if it's an image
        #    shape [B, cond_channels, H, W], do similarly.
        #    If cond is already [B, channels, H, W], just pass it through.
        #    e.g.:
        if cond is not None and cond.dim() == 2:
            # example: cond is [B, H*W], single channel
            cond = cond.view(B_, 1, H, W)

        # 3) Call the CNN model => [B, vocab_size, H, W]
        logits_4d = model(x_img, sigma, cond)

        # 4) Flatten => [B, L, vocab_size]
        vocab_size = logits_4d.shape[1]
        logits_2d = logits_4d.permute(0, 2, 3, 1).reshape(B_, L, vocab_size)

        return logits_2d  # the log-scores

    return score_fn
