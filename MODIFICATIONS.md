MODIFICATIONS to speed up training
=================================

This file documents the changes made to the original NaSA_TD3 codebase in this fork to allow much faster experiments suitable for a class project. The intent was to reduce wall-clock training time (target: < 1 day) while keeping a reasonable result for demonstration.

Summary of edits
- `train_loop.py`
  - Lowered default `--max_steps_training` from `1_000_000` to `200_000`.
  - Added new CLI args: `--render_height` and `--render_width` (defaults set to `64`).
  - Training now constructs `FrameStack` with the requested render size.
  - Training now passes the render size into `AE_TD3` so the networks match the input size.

- `utils/Frame_Stack.py`
  - `FrameStack` now accepts `height` and `width` parameters and uses them when calling `env.physics.render(...)`. This makes it easy to reduce image resolution (and therefore conv compute) via CLI args.

- `networks/Encoder.py`
  - Encoder now accepts `img_h` and `img_w` and computes the flattened conv output size dynamically. The hard-coded linear input size (`39200`) was removed so the encoder supports different render resolutions robustly.

- `networks/Decoder.py`
  - Decoder now accepts the encoder's conv output shape and builds its `fc`/`reshape` dynamically rather than using a hard-coded `(32,35,35)` reshape. This keeps encoder/decoder shapes consistent across image sizes.

- `nasa_td3.py`
  - `AE_TD3` constructor now accepts `img_h` and `img_w`, creates the encoder with those sizes, reads the encoder conv shape, and constructs the decoder with the matching shape.
  - `AE_TD3.get_intrinsic_values` now computes and returns a boredom value (EMA of familiarity).
  - Added CLI-controllable intrinsic weights `--alpha_s` and `--alpha_n` (see below) so you can run pure-boredom, pure-surprise/novelty, or hybrid experiments.

Notes on default behavior vs. recommended experiment settings
- Code defaults (after edits):
  - `--max_steps_training` default = `200000`
  - `--render_height` default = `64`, `--render_width` default = `64`
  - Frame stack `k` default still = `3` (so input channels remain `3*k` for RGB)

- Recommended CLI for a fast, still demonstrative run (used in experiments):
```
python3 train_loop.py \
  --max_steps_training 200000 \
  --G 1 \
  --batch_size 64 \
  --latent_size 128 \
  --render_height 64 \
  --render_width 64 \
  --intrinsic False
```

  - `--G`: number of gradient updates per environment step. Reducing `G` from the original `5` to `1` reduces the number of network updates and therefore compute.
  - `--latent_size`: lowering the latent dimension reduces network FLOPs and memory.
  - `--intrinsic False`: disabling intrinsic objectives avoids additional ensemble training overhead; if you require intrinsic rewards, consider lowering the ensemble size.

Other knobs you can change for more speed (more invasive)
- Convert frames to grayscale (1 channel instead of 3) — large conv savings. This requires changing `FrameStack` to convert to gray and adjusting `k` accounting for channels in `AE_TD3`.
- Reduce convolutional filters (e.g., `num_filters` from 32 → 16) or model depth in `Encoder`/`Decoder`.
- Reduce `ensemble_size` (default in code is 5) to 1–3 or disable `train_predictive_model` when intrinsic rewards are off.

Dependencies and environment notes
- DeepMind Control Suite (`dm_control`) requires MuJoCo and the appropriate system setup. On many systems you can install via pip and follow MuJoCo setup instructions.

Suggested pip installs (minimum to run training):
```
pip install torch torchvision numpy pandas matplotlib opencv-python scikit-image dm_control mujoco
```

If you use a GPU, install the matching `torch` + `torchvision` for your CUDA version from https://pytorch.org/.

Why these changes help
- Lowering `max_steps_training` reduces the number of environment steps (linear reduction in run time).
- Reducing `G` reduces the number of gradient updates per step (multiplicative effect).
- Lowering image resolution reduces convolution FLOPs roughly proportionally to the number of pixels (quadratic reduction as both height and width shrink).
- Smaller latent sizes and fewer filters reduce per-step forward/backward computation.

If you want, I can (choose one):
- implement grayscale pipeline + adjust networks (largest safe speedup),
- expose `ensemble_size` and `num_filters` as CLI args and set smaller defaults, or
- run a short smoke test (2000 steps) locally in this environment to confirm everything runs end-to-end.

New intrinsic / boredom options and sample commands
- `--alpha_s` and `--alpha_n` (floats, default 1.0) control the weight applied to surprise and novelty when `get_intrinsic_values` is computed. Use these to run boredom-only or hybrid experiments.

Examples:

- Boredom-only (disable surprise/novelty contribution):
```
python3 train_loop.py --intrinsic True --boredom True --alpha_s 0.0 --alpha_n 0.0 --boredom_beta 0.1
```

- Hybrid (surprise + novelty + boredom penalty):
```
python3 train_loop.py --intrinsic True --boredom True --alpha_s 1.0 --alpha_n 1.0 --boredom_beta 0.1
```

-- end
