# SEDD for conditional discrete label maps

Research modifications of [Score Entropy Discrete Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)
(Lou, Meng, Ermon, 2024; [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)). The upstream MIT license is retained in `LICENSE`; the original README is kept as `README_upstream.md`.

## Overview

The upstream code targets language modeling: one-dimensional token sequences, an unconditional transformer, and
flash-attention kernels. This repository adapts it to **discrete label maps on a pixel grid, generated conditionally on
an observed image**.

Research work from 2025. The training and evaluation scripts are not provided.

## What was changed

- **Conditioning.** `SEDD.forward(indices, sigma, cond)` accepts a conditioning image. `CNNXEmbedder`
  (`model/transformer.py`) encodes it with a small CNN and adds the result to the timestep embedding that drives the
  AdaLN modulation. `LabelEmbedderDiT` (DiT-style label dropout for classifier-free guidance) is included but not used.
- **Image-shaped state spaces.** `losses.py` and `model/utils.get_score_fn` flatten `[B, C, H, W]` label maps to
  sequences for the score-entropy loss and the graph operations, and unflatten them for the network. Index dtype fixes
  in `graph_lib.py`.
- **CNN score network.** `model/cnn.py` adds `SEDD_CNN`: a FiLM-conditioned residual CNN that works directly on the
  grid, without a token embedding, with the conditioning image concatenated as an extra channel (self-conditioning
  when absent), optional sigma scaling for the absorbing graph, and masking of the current-state logit as in the
  original transformer head.
- **No flash-attention dependency.** `DDiTBlock` uses separate q/k/v projections, a pure-PyTorch rotary embedding
  (`model/rotary.py`, ported from Meta's flow_matching text example) and
  `torch.nn.functional.scaled_dot_product_attention`.
- **Conditional sampling.** All predictors and the denoiser in `sampling.py` take `cond`; `get_pc_sampler` handles image
  batch dimensions; `PCSampler` is a configurable class with sub-batching.
- **Configs** for small two-token vocabularies on 80×80 grids (`configs/config_mod.yaml`,
  `configs/config_very_small.yaml`, `configs/model/{smallest,very_small,small_mod}.yaml`).

## Status

Research snapshot; not actively maintained.

## License

MIT, as upstream. Original code copyright Aaron Lou. Modifications by Andrej Leban, 2025.
