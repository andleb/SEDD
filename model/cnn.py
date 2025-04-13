import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

#####################################################################
#                         Timestep Embedder                          #
#####################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # same as in many diffusion repos
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

#####################################################################
#                 Simple Discrete Embedding for Z                   #
#####################################################################

class EmbeddingLayer(nn.Module):
    """
    Learns an embedding table for discrete tokens (vocab_dim).
    Will output an embedding dimension 'hidden_dim'.
    Later we will reshape from (B, H*W, hidden_dim) to (B, hidden_dim, H, W).
    """
    def __init__(self, hidden_dim, vocab_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_dim = vocab_dim

        # An embedding for discrete tokens
        self.embedding = nn.Embedding(vocab_dim, hidden_dim)
        # Initialize
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, indices):
        """
        indices: (B, H*W) or (B, seq_len) of discrete tokens
        returns: (B, seq_len, hidden_dim)
        """
        return self.embedding(indices)

#####################################################################
#                     CNN Residual Building Block                    #
#####################################################################

class CNNResidualBlock(nn.Module):
    """
    A simple CNN residual block that also supports FiLM-like conditioning
    from a global embedding (e.g., the time embedding).
    """

    def __init__(self, num_channels, time_emb_dim, dropout=0.0):
        """
        :param num_channels: number of channels for the conv layers
        :param time_emb_dim: dimension of the global time embedding
        :param dropout: dropout probability
        """
        super().__init__()
        self.num_channels = num_channels

        # Two conv layers for the residual block
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        # Norms
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=num_channels)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=num_channels)

        # FiLM (or AdaIN) parameters derived from time embedding
        # We'll produce (scale, shift) for conv1 and conv2
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * num_channels)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        """
        x: (B, C, H, W)
        time_emb: (B, time_emb_dim) -- global embedding from TimestepEmbedder
        """
        # Derive FiLM parameters
        film_params = self.time_mlp(time_emb)  # (B, 2*C)
        gamma1, beta1 = film_params[:, :self.num_channels], film_params[:, self.num_channels:2*self.num_channels]
        # We'll re-use the same FiLM parameters for both convs in this simple design
        gamma2, beta2 = gamma1, beta1

        # First conv
        h = self.norm1(x)
        # Apply FiLM for the first conv
        # shape of gamma1, beta1 is (B, C) -> we need to unsqueeze to broadcast
        h = h * (1 + gamma1.unsqueeze(-1).unsqueeze(-1)) + beta1.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv1(h)

        # Second conv
        h = self.norm2(h)
        h = h * (1 + gamma2.unsqueeze(-1).unsqueeze(-1)) + beta2.unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.dropout(h)  # optional dropout
        h = self.conv2(h)

        # Residual connection
        return x + h

#####################################################################
#                     The CNN-based SEDD Model                       #
#####################################################################

class SEDD_CNN(nn.Module, PyTorchModelHubMixin):
    """
    A CNN-based variant of the SEDD model, replacing the Transformer blocks
    with CNN residual blocks.  The main idea is to concatenate:
      - The discrete diffusion variable Z (embedded as channels)
      - The noisy conditioning X (raw or possibly with 1 channel)
    Then run multiple CNN residual blocks.
    Finally, output a per-pixel classification over the vocabulary.
    """

    def __init__(self, config):
        super().__init__()

        # Handle config as OmegaConf
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        # Are we in "absorb" mode? (Original code used this to add 1 to vocab size)
        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)

        # For discrete tokens Z
        self.vocab_embed_dim = config.model.hidden_size  # e.g. 256
        self.vocab_embed = EmbeddingLayer(self.vocab_embed_dim, vocab_size)

        # Timestep embedding
        self.time_embed_dim = config.model.cond_dim   # e.g. 256
        self.sigma_map = TimestepEmbedder(self.time_embed_dim)

        # Whether or not to scale the outputs by sigma at the end
        self.scale_by_sigma = config.model.scale_by_sigma

        # Number of CNN channels in the main body
        self.num_cnn_channels = config.model.cnn_channels  # e.g. 64 or 128

        # We will build a small sequence of CNN blocks
        n_cnn_blocks = config.model.n_blocks  # re-using config.model.n_blocks for CNN depth

        # 1) A "prep" conv that merges embedded Z and the conditioning X
        #    so the shape is [B, self.num_cnn_channels, H, W].
        #
        #    - Suppose Z -> shape [B, vocab_embed_dim, H, W]
        #    - Suppose X -> shape [B, 1, H, W], or possibly more channels
        #      if your conditioning has more channels.
        #    - We just concat along channel dimension, then reduce to num_cnn_channels.
        #
        #    We do a 1x1 conv from (vocab_embed_dim + cond_channels) -> num_cnn_channels.
        cond_channels = config.model.cond_channels  # e.g. 1 if X is just a single channel
        in_channels = self.vocab_embed_dim + cond_channels

        self.prep_conv = nn.Conv2d(in_channels, self.num_cnn_channels, kernel_size=1)

        # 2) CNN residual blocks
        blocks = []
        for _ in range(n_cnn_blocks):
            blocks.append(CNNResidualBlock(self.num_cnn_channels,
                                           time_emb_dim=self.time_embed_dim,
                                           dropout=config.model.dropout))
        self.cnn_blocks = nn.ModuleList(blocks)

        # 3) Final projection to get logits over the vocab
        #    We'll do a 1x1 conv from (num_cnn_channels) -> vocab_size
        self.final_conv = nn.Conv2d(self.num_cnn_channels, vocab_size, kernel_size=1)

    def forward(self, indices, sigma, cond):
        """
        :param indices: (B, H*W) discrete tokens for star/no-star
        :param sigma:   (B,) diffusion times
        :param cond:    (B, cond_channels, H, W) or possibly (B, H*W) to reshape
        :return: logits over vocab, shape (B, vocab_size, H, W)
        """
        B = indices.shape[0]

        # ------------------------------------------------------
        # 1) Embed the discrete variable Z
        #    Suppose indices has shape (B, H*W). We want to reshape to (B, H, W).
        #    Let the config say the image is config.model.img_size x img_size
        # ------------------------------------------------------
        H = self.config.model.img_size
        W = self.config.model.img_size
        x_embed = self.vocab_embed(indices)   # (B, H*W, embed_dim)
        x_embed = rearrange(x_embed, "b (h w) c -> b c h w", h=H, w=W)  # (B, embed_dim, H, W)

        # cond might already be shape (B, cond_channels, H, W).
        # if it's flattened, reshape here
        if cond.dim() == 2 and cond.shape[1] == H*W:
            cond = cond.view(B, 1, H, W)  # assume 1 channel for the conditioning

        # ------------------------------------------------------
        # 2) Concatenate along channel dimension [Z_embed, cond]
        #    => shape (B, embed_dim + cond_channels, H, W)
        # ------------------------------------------------------
        x = torch.cat([x_embed, cond], dim=1)

        # ------------------------------------------------------
        # 3) Map to the base CNN channels
        # ------------------------------------------------------
        x = self.prep_conv(x)  # (B, num_cnn_channels, H, W)

        # ------------------------------------------------------
        # 4) Timestep embedding
        #    We'll pass this to each CNN residual block for FiLM
        # ------------------------------------------------------
        t_emb = self.sigma_map(sigma)  # shape (B, time_embed_dim)

        # ------------------------------------------------------
        # 5) CNN blocks
        # ------------------------------------------------------
        for block in self.cnn_blocks:
            x = block(x, t_emb)

        # ------------------------------------------------------
        # 6) Final projection to vocab logits
        # ------------------------------------------------------
        logits = self.final_conv(x)  # (B, vocab_size, H, W)

        # ------------------------------------------------------
        # 7) scale_by_sigma logic (optional)
        #    This was done for "absorb" case in the original code
        # ------------------------------------------------------
        if self.scale_by_sigma:
            # Typically used for discrete offset in some SED approaches
            # This exactly follows the original:
            #   esigm1_log = log(expm1(sigma)) or log(exp(sigma) - 1)
            assert self.absorb, "Haven’t configured scale_by_sigma unless absorb is True."
            esigm1_log = torch.where(sigma < 0.5,
                                     torch.expm1(sigma),
                                     sigma.exp() - 1).log().to(logits.dtype)
            esigm1_log = esigm1_log[:, None, None, None]  # broadcast
            # Subtract off log( number_of_classes - 1 ) as in the original code
            logits = logits - esigm1_log - np.log(logits.shape[1] - 1)

        # Optionally zero out the logit for the "same token" as the input (original code did scatter)
        # That was something like:
        #   x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        # You can replicate that if you still want to force the network to "exclude" the input token.
        # For a 2D grid, you'd flatten the last dimension again or do something else.

        return logits

