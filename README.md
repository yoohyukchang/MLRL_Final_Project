# NaSA_TD3
Official PyTorch Implementation of Image-Based Deep Reinforcement Learning with Intrinsically Motivated Stimuli: On the Execution of Complex Robotic Tasks


## General Overview
See our  [Paper-Blog](https://sites.google.com/aucklanduni.ac.nz/nasa-td3-pytorch/home) for details  for pseudocode with more details of the training process as well as details of hyperparameters, full source code and videos of each task.


## Prerequisites

|Library         | Version |
|----------------------|----|
| Our RL Support Libray |[link](https://github.com/UoA-CARES/cares_reinforcement_learning)|
| DeepMind Control Suite |[link](https://github.com/deepmind/dm_control) |


## Network Architecture
<p align="center">
  <img src="https://github.com/UoA-CARES/NaSA_TD3/blob/main/repo_images/AE_TD3_network_diagram.png">
</p>


## Instructions Training
To train the NaSA-TD3 algorithm on the deep mind control suite from image-based observations, please run:
```
python3 train_loop.py --env=ball_in_cup --task=catch --seed=1 --intrinsic=True
```

## Quickstart — Fast training (for class projects)
The repository has a set of modifications and recommended run settings to significantly reduce training time for demonstration purposes. Those changes allow you to run a demonstrative experiment in under a day on a modern GPU.

Install the main Python dependencies (example):
```
pip install torch torchvision numpy pandas matplotlib opencv-python scikit-image dm_control mujoco
```

Recommended fast-run command used for experiments:
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

- `--max_steps_training`: total environment steps (default reduced from 1,000,000 to 200,000 in this fork)
- `--G`: number of gradient updates per env step (use `1` to reduce compute)
- `--render_height/--render_width`: lower the image resolution (default set to 64×64 here)
- `--intrinsic False`: turn off intrinsic ensemble training for faster runs

See `MODIFICATIONS.md` in the project root for a full list of edits, rationale, and additional suggestions for trading performance for speed (grayscale frames, smaller convs, smaller ensemble size, etc.).
 
Additional examples (boredom / hybrid runs):

- Hybrid (surprise + novelty + boredom penalty):
```
python3 train_loop.py --intrinsic True --boredom True --alpha_s 1.0 --alpha_n 1.0 --boredom_beta 0.1
```

- Boredom-only (disable surprise & novelty):
```
python3 train_loop.py --intrinsic True --boredom True --alpha_s 0.0 --alpha_n 0.0 --boredom_beta 0.1
```
## Our Results


<p align="center">
  <img src="https://github.com/UoA-CARES/NaSA_TD3/blob/main/repo_images/results_simulations.png">
</p>



## Citation
If you use either the paper or code in your paper or project, please kindly star this repo and cite our work.
