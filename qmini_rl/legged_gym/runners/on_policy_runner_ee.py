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

import time
import os
from collections import deque
import statistics

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

class Estimator:
    def __init__(self, input_dim, output_dim, buffer_capacity, batch_size, lr, num_epochs, device):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.learning_rate = lr
        self.num_epochs = num_epochs

        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.output_dim)
        ).to(self.device)
        print(f"Estimator MLP: {self.net}")

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                # nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.weight, 0, 0.1)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        self.replay_inputs = torch.zeros(self.buffer_capacity, self.input_dim, device=self.device)
        self.replay_outputs = torch.zeros(self.buffer_capacity, self.output_dim, device=self.device)
        self.current_size = 0
        self.write_idx = 0

        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.2, total_iters=2000)

    def collect_data(self, estimator_input, estimator_output_ref):
        if not isinstance(estimator_input, torch.Tensor):
            print("Warning: estimator_input is not a torch.Tensor.")
            estimator_input = torch.tensor(estimator_input, dtype=torch.float, device=self.device)
        if not isinstance(estimator_output_ref, torch.Tensor):
            print("Warning: estimator_output_ref is not a torch.Tensor.")
            estimator_output_ref = torch.tensor(estimator_output_ref, dtype=torch.float, device=self.device)
        input_env_nums = estimator_input.shape[0]
        start_idx = self.write_idx
        end_idx = min(start_idx + input_env_nums, self.buffer_capacity)
        self.replay_inputs[start_idx:end_idx] = estimator_input[:end_idx - start_idx]
        self.replay_outputs[start_idx:end_idx] = estimator_output_ref[:end_idx - start_idx]

        if end_idx < start_idx + input_env_nums:
            remaining = input_env_nums - (end_idx - start_idx)
            self.replay_inputs[0:remaining] = estimator_input[end_idx - start_idx:]
            self.replay_outputs[0:remaining] = estimator_output_ref[end_idx - start_idx:]

        self.write_idx = (self.write_idx + input_env_nums) % self.buffer_capacity
        self.current_size = min(self.current_size + input_env_nums, self.buffer_capacity)

    def update(self):
        if self.current_size < self.batch_size:
            return None

        inputs = self.replay_inputs[:self.current_size]
        outputs_ref = self.replay_outputs[:self.current_size]
        dataset = TensorDataset(inputs, outputs_ref)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_loss = 0.0
        for epoch in range(self.num_epochs):
            for batch_inputs, batch_outputs_ref in dataloader:
                self.optimizer.zero_grad()
                batch_outputs = self.net(batch_inputs)
                loss = nn.MSELoss()(batch_outputs, batch_outputs_ref)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

        # Valid for logging
        self.net.eval()
        num_valid = min(4 * self.batch_size, self.current_size)
        inputs_valid = self.replay_inputs[:num_valid]
        outputs_ref_valid = self.replay_outputs[:num_valid]
        if num_valid > 0:
            with torch.no_grad():
                valid_outputs = self.net(inputs_valid)
                self.avg_absolute_error_valid = torch.abs(valid_outputs - outputs_ref_valid).mean().item()
            self.net.train()

        self.scheduler.step()
        self.learning_rate = self.optimizer.param_groups[0]['lr']
        self.estimator_loss = total_loss / (self.num_epochs * len(dataloader))
        return self.estimator_loss

class OnPolicyRunnerEE:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.cfg.env.num_actor_observations,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.cfg.env.num_actor_observations], [self.env.num_privileged_obs], [self.env.num_actions])

        # Estimator
        self.estimator_cfg = train_cfg["estimator"]
        self.estimator = Estimator(input_dim=self.estimator_cfg["input_dim"], 
                                   output_dim=self.estimator_cfg["output_dim"], 
                                   buffer_capacity=self.estimator_cfg["buffer_capacity"], 
                                   batch_size=self.estimator_cfg["batch_size"], 
                                   lr=self.estimator_cfg["lr"],
                                   num_epochs=self.estimator_cfg["num_epochs"],
                                   device=self.device)

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        # Estimator
        self.estimator.net.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    estimator_output = self.estimator.net(obs)
                    actor_obs = torch.cat((obs, estimator_output), dim=1)
                    actions = self.alg.act(actor_obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos, estimator_output_ref = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones, estimator_output_ref = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device), estimator_output_ref.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)

                    # Estimator
                    self.estimator.collect_data(obs, estimator_output_ref)  
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            mean_value_loss, mean_surrogate_loss = self.alg.update()

            # Estimator
            estimator_loss = self.estimator.update()
            if estimator_loss is not None:
                self.writer.add_scalar('Estimator/loss', estimator_loss, it)
                self.writer.add_scalar('Estimator/learning_rate', self.estimator.learning_rate, it)
                self.writer.add_scalar('Estimator/avg_absolute_error_valid', self.estimator.avg_absolute_error_valid, it)

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)),
                            os.path.join(self.log_dir, 'estimator_{}.pt'.format(it)))
            ep_infos.clear()
        
        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)),
                 os.path.join(self.log_dir, 'estimator_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string

        # Estimator
        log_string += (f"""\n{'Estimator loss:':>{pad}} {self.estimator.estimator_loss:.4f}\n"""
                       f"""{'Estimator learning rate:':>{pad}} {self.estimator.learning_rate:.6f}\n"""
                       f"""{'Estimator average absolute error valid:':>{pad}} {self.estimator.avg_absolute_error_valid:.4f}\n"""
                       )

        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, estimator_path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)
        # Estimator
        torch.save(self.estimator.net.state_dict(), estimator_path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        
        # Estimator
        estimator_path = path.replace('model_', 'estimator_')
        self.estimator.net.load_state_dict(torch.load(estimator_path))
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_inference_estimator(self, device=None):
        # Estimator
        self.estimator.net.eval() 
        if device is not None:
            self.estimator.net.to(device)
        return self.estimator.net