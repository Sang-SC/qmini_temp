<div align="center">
  <h1 align="center">Qmini RL GYM</h1>
  <p align="center">
    <span> 🌎English </span> | <a href="README_zh.md"> 🇨🇳中文 </a>
  </p>
</div>

---

## Installation

### Create a Python Virtual Environment with Anaconda

```bash
conda create -y -n qmini_rl python=3.8
conda activate qmini_rl
```

### Install PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Install Isaac Gym

```bash
# Download Isaac Gym Preview 3 (or 4) from https://developer.nvidia.com/isaac-gym.
# Extract the Isaac Gym archive and run the following command in the `isaacgym/python` folder:
pip install -e .

# Verify successful installation by running an example in the `isaacgym/python/examples` folder:
python 1080_balls_of_solitude.py
```

### Install rsl_rl

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e .
```

### Install qmini_rl

```bash
git clone https://github.com/Sang-SC/qmini.git
cd qmini_rl
pip install -e .
```

### Install ONNX and ONNX Runtime

```bash
pip install onnx
pip install onnxruntime
```

## Usage

### Training

```bash
python legged_gym/scripts/train.py --task=qmini --headless
```

### Running

```bash
python legged_gym/scripts/play.py --task=qmini
```

### Visualize Training Progress with TensorBoard

```bash
# Open http://localhost:6006/ in your browser
tensorboard --logdir=logs
```

## Algorithm Overview

### Network Structure

There are two networks: an Explicit Estimator network trained using supervised learning and an asymmetric Actor-Critic network trained using the PPO algorithm.

The input to the Explicit Estimator network is as follows, with a total of 215 dimensions:
- Body angular velocity (3 dimensions)
- Gravity projection (3 dimensions)
- Velocity command (3 dimensions)
- Joint angles (10 dimensions)
- Joint velocities (10 dimensions)
- Previous action (10 dimensions)
- Gait clock (4 dimensions)
- Historical information of the above states for the past 4 steps

The output of the Explicit Estimator network is as follows, with a total of 5 dimensions:
- Body linear velocity (3 dimensions)
- Foot contact state (2 dimensions)

The Actor network’s input includes the input and output of the Explicit Estimator network, totaling 220 dimensions.

The Actor network’s output consists of joint actions, with 10 dimensions.

The Critic network’s input includes the Actor network’s input plus additional privileged observations, such as terrain information and domain randomization parameters.

### Domain Randomization

Domain randomization is applied to joint friction, damping, armature, body mass, body center of mass position, ground friction coefficient, joint PD gains, joint torque, and joint initial zero points to reduce the sim-to-real gap.

## Comparison with legged_gym

### Modified Files and Their Purposes
- `legged_gym/runners` folder: Added `on_policy_runner_ee.py` file, which incorporates an additional Explicit Estimator network in the runner.
- `legged_gym/envs/__init__.py` file: Registers the Qmini training task.
- `legged_gym/envs/qmini` folder: Contains `qmini.py` and `qmini_config.py` files, which are the core training files for Qmini.
- `legged_gym/envs/scripts` folder: Modified `play.py` file to include code related to the Explicit Estimator network and ONNX model export.
- `legged_gym/envs/utils` folder: Modified `task_registry.py` file to use `OnPolicyRunnerEE` for task registration.
- `legged_gym/envs/utils` folder: Modified `terrain.py` file to specify a less challenging terrain curriculum for Qmini training.

### Unmodified Files
- `legged_gym/envs/base` folder: `base_config.py`, `base_task.py`, `legged_robot_config.py`, `legged_robot.py` files.
- `legged_gym/envs/scripts` folder: `train.py` file.
- `legged_gym/envs/utils` folder: `helpers.py`, `logger.py`, `math.py` files.

### Additional Notes
- In `qmini_config.py`, the default `num_envs` is set to 256 for testing environment setup. For training, it is recommended to increase this to 4096.

## Acknowledgments

This repository is built upon the support and contributions of the following open-source projects. Special thanks to:

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): The foundation for training and running codes.
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithm implementation.