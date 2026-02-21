# Reinforcement Learning with Differentiable Simulation for Quadrotors

## Overview

An implementation of a reinforcement learning framework for quadrotor control using differentiable simulation, built on top of [Crazyflow](https://github.com/utiasDSL/crazyflow) and [JAX](https://github.com/jax-ml/jax). The core idea is to leverage gradients through the simulator dynamics to train policies via backpropagation through time (BPTT), enabling highly sample-efficient policy learning.

This project implements and compares gradient-based reinforcement learning algorithms such as:

- Short Horizon Actor-Critic (SHAC)
	- From “Accelerated Policy Learning with Parallel Differentiable Simulation”
- (Short-Horizon) Backpropagation Through Time (BPTT)
- PPO as a control baseline

## Highlights

- Refactored the entire RL training pipeline in pure JAX, achieving **~17×** speedup over the original PyTorch implementation.

- The environment is fully **JAX-native** and supports Gym-like wrapper usage—no PyTorch dependencies.

- Demonstrated **~70×** higher sample efficiency of BPTT over PPO while reducing wall-clock training time by **~50%**.


## Installation

### Prerequisites

Install [Pixi](https://pixi.sh) if you don't have it already.

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### One-line installation & activation

Clone this repository and run:

```bash
pixi shell
# or the GPU version
pixi shell -e gpu
```

## Training

Training scripts are organized under:

```
rl_diffsim/
├── envs/          # JAX Environments
├── control/       # Controllers for deployment
├── bptt/          # Back-Propagation Through Time
├── shac/          # Short Horizon Actor-Critic
├── ppo/           # Proximal Policy Optimization
saves/             # Saved models and plots
scripts/           # Deploy scripts
```

Each algorithm provides standalone training scripts for different tasks.

### BPTT

#### Figure-8 Tracking

```bash
python rl_diffsim/bptt/train_bptt_figure8.py
python rl_diffsim/bptt/train_bptt_figure8ft.py      # Force torque interface
python rl_diffsim/bptt/train_bptt_figure8rv.py      # Rotor velocity interface
```

#### Reach Position

```bash
python rl_diffsim/bptt/train_bptt_reachpos.py
python rl_diffsim/bptt/train_bptt_reachposft.py     # Force torque interface
python rl_diffsim/bptt/train_bptt_reachposrv.py     # Rotor velocity interface
```

#### Random Trajectory

```bash
python rl_diffsim/bptt/train_bptt_randtraj.py
python rl_diffsim/bptt/train_bptt_randtrajrv.py     # Rotor velocity interface
```

#### Drone Racing

```bash
python rl_diffsim/bptt/train_bptt_race.py
python rl_diffsim/bptt/train_bptt_race_lv2.py       # Level 2
```

### SHAC

#### Figure-8 Tracking

```bash
python rl_diffsim/shac/train_shac_figure8.py
```

#### Drone Racing

```bash
python rl_diffsim/shac/train_shac_race.py
python rl_diffsim/shac/train_shac_race_lv2.py       # Level 2
```

### PPO (Baseline)

#### Figure-8 Tracking

```bash
python rl_diffsim/ppo/train_ppo_figure8.py
python rl_diffsim/ppo/train_ppo_figure8ft.py        # Force torque interface
```

#### Reach Position

```bash
python rl_diffsim/ppo/train_ppo_reachpos.py
```

#### Drone Racing

```bash
python rl_diffsim/ppo/train_ppo_race.py
python rl_diffsim/ppo/train_ppo_race_lv2.py         # Level 2
```

### CL Arguments

All training scripts support the following optional flags:

| Flag | Description |
| --- | --- |
| -w False | Disable Weights & Biases logging |
| -t False | Skip training and directly run evaluation |
| -n N | Number of evaluation episodes |
| -r False | Disable rendering during evaluation |
| -p False | Disable plotting during evaluation |

#### Example Usage

Train without wandb and evaluate without rendering:

```bash
python rl_diffsim/bptt/train_bptt_figure8.py -w False -r False
```

Evaluate a trained model 10 times without retraining:

```bash
python rl_diffsim/bptt/train_bptt_race.py -t False -n 10
```



