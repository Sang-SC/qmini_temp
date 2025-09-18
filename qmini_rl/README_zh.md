<div align="center">
  <h1 align="center">Qmini RL GYM</h1>
  <p align="center">
    <a href="README.md">🌎 English</a> | <span>🇨🇳 中文</span>
  </p>
</div>

---

## 安装

### 使用 Anaconda 创建 Python 虚拟环境

```bash
conda create -y -n qmini_rl python=3.8
conda activate qmini_rl
```

### 安装 PyTorch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 安装 Isaac Gym

```bash
# 从 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 3 （或 4 ）压缩包
# 解压 Isaac Gym 压缩包，在 isaacgym——python 文件夹下执行命令
pip install -e .

# 在 isaacgym——python——examples 文件夹下，验证是否能成功运行例程
python 1080_balls_of_solitude.py
```

### 安装 rsl_rl

```bash
git clone https://github.com/leggedrobotics/rsl_rl.git
cd rsl_rl
git checkout v1.0.2
pip install -e .
```

### 安装 qmini_rl

```bash
git clone https://github.com/Sang-SC/qmini.git
cd qmini_rl
pip install -e .
```

### 安装 ONNX 和 ONNX Runtime

```bash
pip install onnx
pip install onnxruntime
```

## 使用

### 训练

```bash
cd qmini_rl
python legged_gym/scripts/train.py --task=qmini --headless
```

### 运行

```bash
python legged_gym/scripts/play.py --task=qmini
```

### 使用 TensorBoard 查看训练过程。

```bash
# 浏览器打开 http://localhost:6006/
tensorboard --logdir=logs
```

## 算法简介

### 网络结构

共有两个网络：基于监督学习训练的显示估计器 Explicit Estimator 网络，以及基于 PPO 算法训练的非对称 Actor-Critic 网络

Explicit Estimator 网络的输入如下，共 215 维：
- 机身角速度（3 维）
- 重力投影（3 维）
- 速度指令（3 维）
- 关节角度（10 维）
- 关节速度（10 维）
- 上一步动作（10 维）
- 步态时钟（4 维）
- 以及上述状态的过去 4 步历史信息

Explicit Estimator 网络的输出如下，共 5 维：
- 机身线速度（3 维）
- 足底接触状态（2 维）

Actor 网络的输入包含 Explicit Estimator 网络的输入和输出，共 220 维。

Actor 网络的输出为关节动作，共 10 维。

Critic 网络的输入除了包含 Actor 网络的输入外，还有额外的特权观测，例如地形信息、域随机化的一些信息。

### 域随机化

对关节摩擦、阻尼、电枢，机身质量，机身质心位置，地面摩擦系数，关节 PD ，关节扭矩以及关节初始零点进行域随机化，以减小 sim2real gap。

## 和 legged_gym 对比：

修改的文件及作用简介：
- `legged_gym/runners` 文件夹，新增 `on_policy_runner_ee.py` 文件，在运行器中添加了额外的 Explicit Estimator 网络
- `legged_gym/envs/__init__.py` 文件，用于注册 Qmini 训练任务。
- `legged_gym/envs/qmini` 文件夹 `qmini.py`，`qmini_config.py` 文件。Qmini 核心训练文件。
- `legged_gym/envs/scripts` 文件夹中的 `play.py` 文件。添加了 Explicit Estimator 网络相关代码，并添加了导出 ONNX 模型的代码
- `legged_gym/envs/utils` 文件夹中的 `task_registry.py` 文件。注册任务时，使用 `OnPolicyRunnerEE`。
- `legged_gym/envs/utils` 文件夹中的 `terrain.py` 文件。训练 Qmini 时需要指定难度较小的地形课程。

未修改的文件
- `legged_gym/envs/base` 文件夹中的 `base_config.py`，`base_task.py`，`legged_robot_config.py`，`legged_robot.py` 文件
- `legged_gym/envs/scripts` 文件夹中的 `train.py` 文件
- `legged_gym/envs/utils` 文件夹中的 `helpers.py`，`logger.py`，`math.py` 文件

### 其它

`qmini_config.py` 中 num_envs 默认为 256 用于测试环境配置是否成功。训练时建议修改为 4096 。

## 致谢

本仓库开发离不开以下开源项目的支持与贡献，特此感谢：

- [legged\_gym](https://github.com/leggedrobotics/legged_gym): 构建训练与运行代码的基础。
- [rsl\_rl](https://github.com/leggedrobotics/rsl_rl.git): 强化学习算法实现。