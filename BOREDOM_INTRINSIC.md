Boredom as an Intrinsic Motivation: Design, Math, and Implementation
=================================================================

Overview
--------
This document describes how we implemented a "boredom" intrinsic signal into the NaSA_TD3 codebase. The goal of the boredom signal is to penalize the agent for repeatedly visiting highly familiar states (i.e. low novelty for extended periods). Boredom is computed as an exponential moving average (EMA) of familiarity; a higher boredom value corresponds to the agent spending long stretches of time in predictable/familiar states. We subtract a scaled boredom penalty from the extrinsic + other intrinsic rewards.

Mathematical formulation
------------------------
Let:
- x_t be the stacked image observation at time t.
- z_t = Encoder(x_t) be the latent embedding produced by the autoencoder's encoder.
- z_hat_{t+1} be the ensemble prediction of the next latent (average across ensemble), and z_{t+1} be the encoded next latent.

Surprise (S_t): measured as MSE prediction error in latent space (same as original paper):

  S_t = || z_hat_{t+1} - z_{t+1} ||^2

Novelty (N_t): measured as 1 - SSIM between reconstructed and original stacked frames (same as original code):

  N_t = 1 - SSIM(x_t, x_hat_t)    where N_t in [0,1]

Familiarity (F_t): defined as the complement of novelty:

  F_t = 1 - N_t

Boredom (B_t): an exponential moving average (EMA) of familiarity, tracking how familiar the agent's recent experiences have been. Using decay parameter lambda in (0,1):

  B_t = lambda * B_{t-1} + (1 - lambda) * F_t

We use boredom as a penalty subtracted from the total reward. The combined intrinsic formulation used in code is:

  r_intrinsic_t = alpha_s * S_t + alpha_n * N_t - beta * B_t

where alpha_s and alpha_n are weights for surprise and novelty (in our current code both set to 1.0), and beta is the scale for the boredom penalty (set via `--boredom_beta`).

Implementation details (where to look)
-------------------------------------
- Intrinsic computation entry point: `AE_TD3.get_intrinsic_values` in `nasa_td3.py`.
  - This method now returns three values: `(surprise_rate, novelty_rate, boredom_rate)`.
  - It computes surprise via `get_surprise_rate(...)` and novelty via `get_novelty_rate(...)` (unchanged from original code).
  - Boredom is updated inside `get_intrinsic_values` as an EMA:

      familiarity = 1 - novelty_rate
      boredom = lambda * boredom + (1 - lambda) * familiarity

  - The code stores `boredom` on the agent instance (variable `self.boredom`).

- Boredom configuration and reset:
  - `AE_TD3` exposes these attributes: `boredom_lambda` (EMA decay), `boredom_beta` (penalty scale), and `use_boredom` (boolean flag to enable).
  - `AE_TD3.reset_boredom()` resets the boredom EMA to zero; the training loop calls this whenever the environment is reset.

- Applying the boredom penalty to RL reward:
  - In `train_loop.py`, after computing intrinsic surprise and novelty, we now subtract the boredom penalty (scaled by `agent.boredom_beta`) from the total reward before adding it to the replay buffer.

- CLI and args:
  - `train_loop.py` adds new command-line arguments:
    - `--boredom` (bool) — enable boredom intrinsic.
    - `--boredom_beta` (float) — scale of boredom penalty (default 0.0).
    - `--boredom_lambda` (float) — EMA decay for boredom (default 0.99).
  - additionally:
    - `--alpha_s` (float) — weight for surprise intrinsic term (default 1.0).
    - `--alpha_n` (float) — weight for novelty intrinsic term (default 1.0).
  - These `alpha` weights allow running pure boredom (set both to 0.0) or hybrid combinations.

Code pointers (exact files / functions)
-------------------------------------
- `nasa_td3.py`:
  - AE_TD3.__init__: boredom attributes initialization (`self.boredom`, `self.boredom_lambda`, `self.boredom_beta`, `self.use_boredom`) and `reset_boredom()`.
  - AE_TD3.get_intrinsic_values: computes surprise, novelty, updates boredom EMA and returns boredom rate.

- `train_loop.py`:
  - `define_parse_args`: new args for boredom.
  - `main`: sets `agent.use_boredom`, `agent.boredom_beta`, and `agent.boredom_lambda` from parsed args.
  - `train`: when `intrinsic` is enabled, unpacks `(surprise, novelty, boredom)` from the agent and applies the boredom penalty: total_reward = extrinsic + surprise + novelty - boredom_beta * boredom.

Design choices and rationale
---------------------------
- Why EMA of familiarity? Boredom is thought of as the accumulation of low-novelty/familiar experiences — one-off familiar states shouldn't cause high boredom, but staying in familiar states over time should. EMA properly accumulates recent familiarity while decaying older events.
- Why subtract boredom? The idea is to penalize the agent when it repeatedly visits familiar states; that encourages exploration away from repetitive/predictable behaviors. In practice, the scale (`beta`) requires tuning: too large and the agent ignores extrinsic goals; too small and the effect is negligible.

Practical tips for experiments
--------------------------------
- Start with `--boredom False` and verify baseline behavior.
- Then enable `--boredom True` and try small `--boredom_beta` values (e.g., 0.01, 0.1, 0.5) to observe effects.
- `--boredom_lambda` controls how quickly boredom accumulates; use high values (0.98–0.995) for slow accumulation and lower values (0.9) for faster response.
- If you keep intrinsic reward (surprise/novelty) enabled, the combined interaction between novelty and boredom can be subtle; analyze trajectories and intrinsic value logs to understand dynamics.

How to reproduce and run
------------------------
Example fast-run (no boredom):
```
python3 train_loop.py --max_steps_training 200000 --G 1 --batch_size 64 --latent_size 128 --render_height 64 --render_width 64 --intrinsic True --boredom False
```

Example run with boredom enabled (small penalty):
```
python3 train_loop.py --max_steps_training 200000 --G 1 --batch_size 64 --latent_size 128 --render_height 64 --render_width 64 --intrinsic True --boredom True --boredom_beta 0.1 --boredom_lambda 0.99
```

Example run (boredom-only):
```
python3 train_loop.py --max_steps_training 200000 --G 1 --batch_size 64 --latent_size 128 --render_height 64 --render_width 64 --intrinsic True --boredom True --alpha_s 0.0 --alpha_n 0.0 --boredom_beta 0.1
```

Notes on logging/analysis
-------------------------
- The code currently does not log per-step intrinsic values to disk by default; you can add logging calls where `get_intrinsic_values` is invoked in `train_loop.py` to record `surprise_rate`, `novelty_rate`, and `boredom_rate` for later plotting and analysis.

Next possible extensions
------------------------
- Compute boredom per latent cluster / per semantic region (rather than a single global scalar) to penalize local repetition while allowing safe repetition in other regions.
- Combine boredom with count-based estimates or density models in latent space for richer measures of familiarity.
- Use learned curiosity bonus schedules to balance extrinsic goals and boredom adaptively.

If you want, I can:
- add per-step logging of intrinsic values (CSV) so you can plot timelines for surprise/novelty/boredom, or
- expose `alpha_s` and `alpha_n` weights as CLI args, or
- implement a per-state (or per-latent-cluster) boredom tracker.

End of document
