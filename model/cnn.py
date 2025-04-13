import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

##############################################################################
#                          Timestep Embedder                                 #
##############################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into a vector representation of size `hidden_size`.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        Create sinusoidal timestep embeddings of shape (B, dim).
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 == 1:
            # zero pad
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        """
        :param t: shape (B,) with diffusion timesteps
        :return: shape (B, hidden_size)
        """
        t_sin = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_sin)

##############################################################################
#                     A Simple CNN Residual Block (FiLM)                     #
##############################################################################

class CNNResidualBlock(nn.Module):
    """
    A residual CNN block that uses FiLM-like conditioning from a global embedding
    (e.g. the diffusion timestep embedding).
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

        # FiLM parameters derived from time embedding
        # We'll produce (gamma, beta) for each conv
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, 2 * num_channels),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb):
        """
        :param x: (B, C, H, W)
        :param time_emb: (B, time_emb_dim)
        """
        # FiLM parameters
        film = self.time_mlp(time_emb)  # shape (B, 2*C)
        gamma1, beta1 = film[:, :self.num_channels], film[:, self.num_channels:]
        # For simplicity, use the same FiLM parameters for both convs
        gamma2, beta2 = gamma1, beta1

        # First conv
        # Apply FiLM for the first conv
        # shape of gamma1, beta1 is (B, C) -> we need to unsqueeze to broadcast
        out = self.norm1(x)
        out = out * (1 + gamma1.view(-1, self.num_channels, 1, 1)) + beta1.view(-1, self.num_channels, 1, 1)
        out = F.silu(out)
        out = self.conv1(out)

        # Second conv
        out = self.norm2(out)
        out = out * (1 + gamma2.view(-1, self.num_channels, 1, 1)) + beta2.view(-1, self.num_channels, 1, 1)
        out = F.silu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        # Residual
        return x + out

##############################################################################
#                  CNN-Based SEDD Model (Discrete Diffusion)                 #
##############################################################################

class SEDD_CNN(nn.Module, PyTorchModelHubMixin):
    """
    CNN-based variant of your Score Entropy Diffusion model.
    - Z: [B, 1, H, W] with 0/1 star positions
    - X: [B, 1, H, W] with noisy conditioning image
    - Output: logits over the discrete states for each pixel, shape [B, vocab_size, H, W].
      For binary diffusion, set vocab_size=2.
    """

    def __init__(self, config):
        super().__init__()

        # If config is dict, convert to OmegaConf for convenience
        if isinstance(config, dict):
            config = OmegaConf.create(config)
        self.config = config

        # 'absorb' or not
        self.absorb = (config.graph.type == "absorb")
        # In a binary case, you might do config.tokens=2,
        # plus 1 if 'absorb' is used.
        vocab_size = config.tokens + (1 if self.absorb else 0)
        self.vocab_size = vocab_size

        # Timestep embedding dimension
        self.time_emb_dim = config.model.cond_dim
        self.sigma_map = TimestepEmbedder(self.time_emb_dim)

        # If the code requires scaling by sigma at the end
        self.scale_by_sigma = config.model.scale_by_sigma

        # For CNN
        self.num_cnn_channels = config.model.cnn_channels  # e.g. 64
        n_cnn_blocks = config.model.n_blocks  # how many residual blocks
        dropout = config.model.dropout

        # We'll combine Z and X by concatenating channels => 2 input channels
        in_channels = 2

        # "Prep" 1×1 conv: (2 -> num_cnn_channels)
        self.prep_conv = nn.Conv2d(in_channels, self.num_cnn_channels, kernel_size=1)

        # Build CNN residual blocks
        blocks = []
        for _ in range(n_cnn_blocks):
            blocks.append(CNNResidualBlock(
                num_channels=self.num_cnn_channels,
                time_emb_dim=self.time_emb_dim,
                dropout=dropout
            ))
        self.cnn_blocks = nn.ModuleList(blocks)

        # Final 1×1 conv to produce per-pixel logits => shape [B, vocab_size, H, W]
        self.final_conv = nn.Conv2d(self.num_cnn_channels, vocab_size, kernel_size=1)

    def forward(self, z_img, sigma, x_img):
        """
        :param z_img: [B, 1, H, W], the discrete star map (0 or 1)
        :param sigma: [B,], diffusion timesteps
        :param x_img: [B, 1, H, W], the noisy conditioning
        :return: logits over the discrete states, shape [B, vocab_size, H, W]
        """

        B, _, H, W = z_img.shape
        # Concatenate Z and X along channel dimension => shape [B, 2, H, W]
        combined = torch.cat([z_img, x_img], dim=1)

        # Map to CNN channels
        h = self.prep_conv(combined)  # (B, num_cnn_channels, H, W)

        # Get the time embedding
        t_emb = self.sigma_map(sigma)  # (B, time_emb_dim)

        # Pass through CNN residual blocks
        for block in self.cnn_blocks:
            h = block(h, t_emb)

        # Final projection
        logits = self.final_conv(h)  # (B, vocab_size, H, W)

        # Optionally scale by sigma (from original code)
        if self.scale_by_sigma:
            # Typically used if we do 'absorb' logic
            assert self.absorb, "scale_by_sigma set but 'absorb' not configured in config."
            # Mirror original approach
            esigm1_log = torch.where(
                sigma < 0.5,
                torch.expm1(sigma),
                sigma.exp() - 1
            ).log().to(logits.dtype)
            esigm1_log = esigm1_log.view(B, 1, 1, 1)  # broadcast
            logits = logits - esigm1_log - np.log(self.vocab_size - 1)

        # TODO:
        # If you want to forcibly zero out the logit for the "same token" (like original code's scatter),
        # you need to do that carefully in 2D. Typically that line was:
        #   x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))
        # but you'd have to adapt it for 2D.
        # For a binary case (star/no-star), you might not strictly need that.

        return logits
