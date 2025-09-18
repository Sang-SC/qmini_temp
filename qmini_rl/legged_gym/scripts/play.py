# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

import torch.onnx
import onnx
import onnxruntime as ort
import copy

import matplotlib.pyplot as plt

# Define the function to export a PyTorch model to ONNX
def export_model_to_onnx(model, input_shape, onnx_file_path, device='cpu'):
    """
    Export a PyTorch model to ONNX format.

    Args:
        model (torch.nn.Module): The PyTorch model to export.
        input_shape (tuple): The shape of the input tensor (e.g., (batch_size, num_obs)).
        onnx_file_path (str): The path to save the ONNX model.
        device (str): The device to use for the model and input ('cpu' or 'cuda').
    """
    # Move model to the specified device and set to evaluation mode
    model = model.to(device)
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(input_shape, device=device)

    # Export the model to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model exported to {onnx_file_path}")

    # Load and check the ONNX model
    onnx_model = onnx.load(onnx_file_path)
    try:
        onnx.checker.check_model(onnx_model)
    except onnx.checker.ValidationError as e:
        print(f"The ONNX model is invalid: {e}")
        return

    # Create an ONNX Runtime session for validation
    ort_session = ort.InferenceSession(onnx_file_path)

    # Prepare a test input
    test_input = torch.zeros(input_shape, device=device, dtype=torch.float32)
    test_input[0, :] = torch.linspace(0, 1, input_shape[1], device=device)

    # PyTorch model inference
    torch_output = model(test_input).detach().cpu().numpy()

    # ONNX Runtime inference
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy().astype(np.float32)}
    ort_outputs = ort_session.run(None, ort_inputs)[0]

    # Print inference results for comparison
    print("\nExample inference results:")
    # print("Input: ", test_input)
    print("PyTorch model inference result: \n", torch_output)
    print("ONNX Runtime inference result: \n", ort_outputs)
    print("Error: \n", ort_outputs - torch_output, "\n")

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    estimator = ppo_runner.get_inference_estimator(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)
    
    if EXPORT_ONNX:
        onnx_file_folder = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported')
        os.makedirs(onnx_file_folder, exist_ok=True)
        export_model_to_onnx(
            model=copy.deepcopy(ppo_runner.alg.actor_critic.actor),
            input_shape=(1, env.cfg.env.num_actor_observations),
            onnx_file_path= os.path.join(onnx_file_folder, 'qmini_actor.onnx'),
            device='cpu'
        )
        export_model_to_onnx(
            model=copy.deepcopy(estimator),
            input_shape=(1, env.num_obs),
            onnx_file_path=os.path.join(onnx_file_folder, 'qmini_estimator.onnx'),
            device='cpu'
        )

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 500 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    for i in range(10*int(env.max_episode_length)):
        TEST_ZERO_COMMANDS = False # True or False
        if TEST_ZERO_COMMANDS:
            env.command_ranges["lin_vel_x"] = [-0.0, 0.0]
            env.command_ranges["lin_vel_y"] = [-0.0, 0.0]
            env.command_ranges["ang_vel_yaw"] = [-0.0, 0.0]
            env.command_ranges["heading"] = [-0.0, 0.0]
            env.commands[:, 0:3] = 0.0

        estimator_output = estimator(obs.detach())
        actor_obs = torch.cat((obs.detach(), estimator_output), dim=1)
        actions = policy(actor_obs)
        obs, _, rews, dones, infos, _ = env.step(actions.detach())
        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    EXPORT_POLICY = False
    EXPORT_ONNX = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    play(args)
