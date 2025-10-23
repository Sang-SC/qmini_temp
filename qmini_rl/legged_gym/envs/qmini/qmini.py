
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil, torch_utils
import torch

class Qmini(LeggedRobot):
    def _process_rigid_shape_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                num_buckets = 64
                bucket_ids = torch.randint(0, num_buckets, (self.num_envs, 1))
                friction_buckets = torch_rand_float(friction_range[0], friction_range[1], (num_buckets,1), device='cpu')
                self.friction_coeffs = friction_buckets[bucket_ids]

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

            self.env_frictions[env_id] = self.friction_coeffs[env_id]
        return props

    def _process_dof_props(self, props, env_id):
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
        
        if self.cfg.domain_rand.randomize_joint_friction:
            min_friction, max_friction = self.cfg.domain_rand.joint_friction_range
            self.joint_frictions[env_id] = torch_rand_float(min_friction, max_friction, (1, 1), device=self.device)
            for i in range(len(props)):
                props['friction'][i] = self.joint_frictions[env_id]

        if self.cfg.domain_rand.randomize_joint_damping:
            min_damping, max_damping = self.cfg.domain_rand.joint_damping_range
            self.joint_dampings[env_id] = torch_rand_float(min_damping, max_damping, (1, 1), device=self.device)
            for i in range(len(props)):
                props['damping'][i] = self.joint_dampings[env_id]

        if self.cfg.domain_rand.randomize_joint_armature:
            min_armature, max_armature = self.cfg.domain_rand.joint_armature_range
            self.joint_armatures[env_id] = torch_rand_float(min_armature, max_armature, (1, 1), device=self.device)
            for i in range(len(props)):
                props['armature'][i] = self.joint_armatures[env_id]

        if self.cfg.domain_rand.randomize_pd_gain:
            min_p_gain_scale, max_p_gain_scale = self.cfg.domain_rand.p_gain_scale_range
            min_d_gain_scale, max_d_gain_scale = self.cfg.domain_rand.d_gain_scale_range
            self.p_gain_scales[env_id, :] = torch_rand_float(min_p_gain_scale, max_p_gain_scale, (1, self.num_actions), device=self.device)
            self.d_gain_scales[env_id, :] = torch_rand_float(min_d_gain_scale, max_d_gain_scale, (1, self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_torque:
            min_torque_scale, max_torque_scale = self.cfg.domain_rand.torque_scale_range
            self.torque_scales[env_id, :] = torch_rand_float(min_torque_scale, max_torque_scale, (1, self.num_actions), device=self.device)

        if self.cfg.domain_rand.randomize_motor_zero_deviation:
            min_motor_zero_deviation, max_motor_zero_deviation = self.cfg.domain_rand.motor_zero_deviation_range
            self.motor_zero_deviations[env_id, :] = torch_rand_float(min_motor_zero_deviation, max_motor_zero_deviation, (1, self.num_actions), device=self.device)

        return props

    def _process_rigid_body_props(self, props, env_id):
        if self.cfg.domain_rand.randomize_base_mass:
            min_added_base_mass, max_added_base_mass = self.cfg.domain_rand.added_base_mass_range
            props[0].mass += torch_rand_float(min_added_base_mass, max_added_base_mass, (1, 1), device=self.device).item()
            self.base_mass[env_id] = props[0].mass
        
        if self.cfg.domain_rand.randomize_base_com:
            min_added_base_com_x, max_added_base_com_x = self.cfg.domain_rand.added_base_com_x_range
            min_added_base_com_y, max_added_base_com_y = self.cfg.domain_rand.added_base_com_y_range
            min_added_base_com_z, max_added_base_com_z = self.cfg.domain_rand.added_base_com_z_range
            props[0].com += gymapi.Vec3(
                torch_rand_float(min_added_base_com_x, max_added_base_com_x, (1, 1), device=self.device).item(),
                torch_rand_float(min_added_base_com_y, max_added_base_com_y, (1, 1), device=self.device).item(),
                torch_rand_float(min_added_base_com_z, max_added_base_com_z, (1, 1), device=self.device).item()
            )
            self.base_com[env_id] = torch.tensor([props[0].com.x, props[0].com.y, props[0].com.z], device=self.device)
        return props

    def _create_envs(self):
        self.env_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_mass = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_com = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_frictions = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False) # assuming all
        self.joint_dampings = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.joint_armatures = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gain_scales = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gain_scales = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torque_scales = torch.ones(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_zero_deviations = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        super()._create_envs()

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        push_robots_vel_x_range = self.cfg.domain_rand.push_robots_vel_x
        push_robots_vel_y_range = self.cfg.domain_rand.push_robots_vel_y
        self.root_states[:, 7] += torch_rand_float(push_robots_vel_x_range[0], push_robots_vel_x_range[1], (self.num_envs, 1), device=self.device).squeeze(1)
        self.root_states[:, 8] += torch_rand_float(push_robots_vel_y_range[0], push_robots_vel_y_range[1], (self.num_envs, 1), device=self.device).squeeze(1)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _init_history_buf(self):
        if self.cfg.env.history_encoding:
            self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, self.cfg.env.num_proprio, dtype=torch.float, device=self.device, requires_grad=False)
            self.privileged_obs_history_buf = torch.zeros(self.num_envs, self.cfg.env.history_len, (self.cfg.env.num_proprio + self.cfg.env.num_priv), dtype=torch.float, device=self.device, requires_grad=False)

    def _init_base_state(self):
        self.base_rpy = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_feet_state(self):
        self.feet_num = len(self.feet_indices)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]
        self.feet_rpy = torch.zeros(self.num_envs, self.feet_num, 3, dtype=torch.float, device=self.device, requires_grad=False)

    def _init_contact(self):
        self.contact_filt = torch.zeros(self.num_envs, self.feet_num, dtype=torch.bool, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, self.feet_num, dtype=torch.bool, device=self.device, requires_grad=False)

    def _init_gait(self):
        self.phase = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.leg_phase = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)
        self.stance_mask = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device, requires_grad=False)
        self.clock_input = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        self.period = self.cfg.custom.gait.period
        offset_rand = torch.randint(0, 2, (self.num_envs,), device=self.device).float() * 0.5
        self.offsets = torch.stack([offset_rand, offset_rand + 0.5], dim=1) % 1
        self.stand_leg_phase = torch.tensor([[0.0, 0.5]], device=self.device).repeat(self.num_envs, 1)

    def _init_buffers(self):
        super()._init_buffers()
        self._init_history_buf()

        self._init_base_state()
        self._init_feet_state()
        self._init_contact()
        self._init_gait()

        self.low_vel_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type=="P":
            p_gains = self.p_gains * self.p_gain_scales
            d_gains = self.d_gains * self.d_gain_scales
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_zero_deviations) - d_gains * self.dof_vel
            torques = torques * self.torque_scales
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def update_base_state(self):
        base_roll, base_pitch, base_yaw = get_euler_xyz(self.base_quat)
        self.base_rpy = torch.stack((base_roll, base_pitch, base_yaw), dim=1)
        # Change the range of rpy to [-pi, pi]
        self.base_rpy = ((self.base_rpy + np.pi) % (2 * np.pi)) - np.pi
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

    def update_feet_state(self):
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_quat = self.feet_state[:, :, 3:7]
        self.feet_vel = self.feet_state[:, :, 7:10]

        for i in range(self.feet_num):
            foot_quat = self.feet_quat[:, i, :]
            foot_roll, foot_pitch, foot_yaw = get_euler_xyz(foot_quat)
            self.feet_rpy[:, i, :] = torch.stack((foot_roll, foot_pitch, foot_yaw), dim=1)
            # Change the range of feet_rpy to [-pi, pi]
            self.feet_rpy[:, i, :] = ((self.feet_rpy[:, i, :] + np.pi) % (2 * np.pi)) - np.pi

    def update_contact(self):
        contact = torch.norm(self.contact_forces[:, self.feet_indices], dim=-1) > 2.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

    def update_gait(self):
        self.t = self.episode_length_buf * self.dt
        self.phase = self.t / self.period

        # If the commanded velocity is too low, set phase to 0
        self.phase[self.low_vel_mask] = 0.0

        self.leg_phase = (self.phase.unsqueeze(1) + self.offsets) % 1
        self.stance_mask = self.leg_phase < 0.55
        self.clock_input = torch.cat((torch.sin(2 * torch.pi * self.leg_phase), 
                            torch.cos(2 * torch.pi * self.leg_phase)), dim=1)

    def _post_physics_step_callback(self):
        self.update_base_state()
        self.update_feet_state()
        self.update_contact()
        self.update_gait()

        return super()._post_physics_step_callback()

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        low_vel_thresholds = self.cfg.custom.low_vel_thresholds
        low_vel_mask_x = torch.abs(self.commands[:, 0]) <= low_vel_thresholds[0]
        low_vel_mask_y = torch.abs(self.commands[:, 1]) <= low_vel_thresholds[1]
        low_vel_mask_yaw = torch.abs(self.commands[:, 2]) <= low_vel_thresholds[2]
        # low_vel_mask_yaw = low_vel_mask_yaw | (low_vel_mask_x & low_vel_mask_y)  # if both x and y are small, yaw is also considered small
        self.low_vel_mask = low_vel_mask_x & low_vel_mask_y & low_vel_mask_yaw
        self.commands[env_ids, 0] *= (~self.low_vel_mask[env_ids]).float()
        self.commands[env_ids, 1] *= (~self.low_vel_mask[env_ids]).float()
        self.commands[env_ids, 2] *= (~self.low_vel_mask[env_ids]).float()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        if self.cfg.env.history_encoding:
            self.obs_history_buf[env_ids, :, :] = 0.
            self.privileged_obs_history_buf[env_ids, :, :] = 0.

        self.last_contacts[env_ids, :] = False

        # Reset gait
        offset_rand = torch.randint(0, 2, (len(env_ids),), device=self.device).float() * 0.5
        self.offsets[env_ids] = torch.stack([offset_rand, offset_rand + 0.5], dim=1) % 1

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = self.default_dof_pos.repeat(len(env_ids), 1)
        self.dof_vel[env_ids] = 0.
        if self.cfg.custom.reset_dof_pos_flag:
            for i in range(self.num_dof):
                self.dof_pos[env_ids, i] += torch_rand_float(self.cfg.custom.reset_dof_pos_ranges[i][0], self.cfg.custom.reset_dof_pos_ranges[i][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.custom.reset_dof_vel_flag:
            for i in range(self.num_dof):
                self.dof_vel[env_ids, i] += torch_rand_float(self.cfg.custom.reset_dof_vel_ranges[i][0], self.cfg.custom.reset_dof_vel_ranges[i][1], (len(env_ids), 1), device=self.device).squeeze(1)

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        # base position
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base orientation
        self.root_states[env_ids, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        if self.cfg.custom.reset_base_rpy_flag:
            roll = torch_rand_float(self.cfg.custom.reset_base_rpy_ranges[0][0], self.cfg.custom.reset_base_rpy_ranges[0][1], (len(env_ids), 1), device=self.device).squeeze(1)
            pitch = torch_rand_float(self.cfg.custom.reset_base_rpy_ranges[1][0], self.cfg.custom.reset_base_rpy_ranges[1][1], (len(env_ids), 1), device=self.device).squeeze(1)
            yaw = torch_rand_float(self.cfg.custom.reset_base_rpy_ranges[2][0], self.cfg.custom.reset_base_rpy_ranges[2][1], (len(env_ids), 1), device=self.device).squeeze(1)
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = 0.
        if self.cfg.custom.reset_base_vel_flag:
            for i in range(3):
                self.root_states[env_ids, 7 + i] = torch_rand_float(self.cfg.custom.reset_base_vel_ranges[i][0], self.cfg.custom.reset_base_vel_ranges[i][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.custom.reset_base_ang_flag:
            for i in range(3):
                self.root_states[env_ids, 10 + i] = torch_rand_float(self.cfg.custom.reset_base_ang_ranges[i][0], self.cfg.custom.reset_base_ang_ranges[i][1], (len(env_ids), 1), device=self.device).squeeze(1)
        
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def step(self, actions):
        obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras = super().step(actions)
        estimator_output_ref = torch.cat((self.base_lin_vel * self.obs_scales.lin_vel, 
                                          self.contact_filt), dim=-1)

        # Check for NaN values
        if torch.isnan(obs_buf).any() or torch.isnan(privileged_obs_buf).any():
            raise ValueError("NaN detected in observations or privileged observations")
        if torch.isnan(rew_buf).any():
            raise ValueError("NaN detected in rewards")

        return obs_buf, privileged_obs_buf, rew_buf, reset_buf, extras, estimator_output_ref

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5)

        # Slow down the curriculum change
        prob_up = 0.1
        random_mask_up = torch.rand(len(env_ids), device=self.device) < prob_up
        self.terrain_levels[env_ids] += 1 * (move_up & random_mask_up)
        self.terrain_levels[env_ids] -= 1 * (move_down & random_mask_up)

        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros(43, device=self.device, dtype=torch.float)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:19] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[19:29] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[29:39] = 0. # previous actions
        noise_vec[39:43] = 0. # clock input
        return noise_vec

    def compute_observations(self):
        obs_buf = torch.cat(( self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    self.clock_input
                                    ),dim=-1)
        privileged_obs_buf = torch.cat(( obs_buf,
                                    self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.env_frictions,
                                    self.base_mass / 20.0,
                                    self.base_com,
                                    self.joint_frictions,
                                    self.joint_dampings,
                                    self.joint_armatures,
                                    self.p_gain_scales,
                                    self.d_gain_scales,
                                    self.torque_scales,
                                    self.motor_zero_deviations
            ),dim=-1)
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        if self.cfg.env.history_encoding:
            self.obs_buf = torch.cat([obs_buf, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
            self.privileged_obs_buf = torch.cat([privileged_obs_buf, self.privileged_obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = obs_buf
            self.privileged_obs_buf = privileged_obs_buf

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                obs_buf.unsqueeze(1),
                self.obs_history_buf[:, :-1]
            ], dim=1)
        )
        self.privileged_obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([privileged_obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                privileged_obs_buf.unsqueeze(1),
                self.privileged_obs_history_buf[:, :-1]
            ], dim=1)
        )
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat((self.privileged_obs_buf, heights), dim=-1)

    def _reward_gait_match(self):
        stance_mask_new = self.stance_mask | self.low_vel_mask.unsqueeze(1)
        gait_match = ~(self.contact_filt ^ stance_mask_new)
        rew_gait_match = torch.sum(gait_match, dim=1) / 2.0
        return rew_gait_match

    def _reward_gait_feet_air_time(self):
        first_contact = (self.feet_air_time > 0.) * self.contact_filt
        self.feet_air_time += self.dt
        target_air_time = self.period * (1.0 - 0.55)
        feet_air_time = self.feet_air_time.clamp(0.0, 1.2 * target_air_time)
        self.feet_air_time *= ~self.contact_filt
        rew_air_time = torch.sum(feet_air_time * first_contact, dim=1) # only on first contact with the ground
        return rew_air_time
    
    def _reward_gait_feet_swing_height(self):
        error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        for i in range(self.feet_num):
            height_error = torch.square(
                self.feet_pos[:, i, 2] - 
                self.cfg.rewards.gait_feet_swing_height_target -
                self.cfg.rewards.foot_height_offset -
                torch.mean(self.measured_heights, dim=1)
            )
            error += height_error * (~self.stance_mask[:, i]).float()
        error[self.low_vel_mask] = 0.
        return error

    def _reward_tracking_lin_vel_x(self):
        error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-error/self.cfg.rewards.lin_vel_x_tracking_sigma)

    def _reward_tracking_lin_vel_y(self):
        lin_vel_y_error_threshold = 0.2
        error = self.commands[:, 1] - self.base_lin_vel[:, 1]
        error = torch.where(torch.abs(error) <= lin_vel_y_error_threshold, 
                                    torch.zeros_like(error), 
                                    torch.abs(error) - lin_vel_y_error_threshold)
        error = torch.square(error)
        return torch.exp(-error/self.cfg.rewards.lin_vel_y_tracking_sigma)

    def _reward_tracking_ang_vel(self):
        error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-error/self.cfg.rewards.ang_vel_tracking_sigma)

    def _reward_base_height(self):
        return torch.square(self.base_height - self.cfg.rewards.base_height_target)

    def _reward_base_orientation(self):
        roll_error = torch.square(self.base_rpy[:, 0] - self.cfg.rewards.base_roll_target)
        pitch_error = torch.square(self.base_rpy[:, 1] - self.cfg.rewards.base_pitch_target)
        error = roll_error + pitch_error
        return error

    def _reward_feet_orientation(self):
        error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.feet_num):
            roll_error = torch.square(self.feet_rpy[:, i, 0])
            pitch_error = torch.square(self.feet_rpy[:, i, 1])
            error += roll_error + pitch_error
        return error

    def _reward_alive(self):
        return 1.0

    def _reward_action(self):
        return torch.sum(torch.abs(self.actions), dim=1)

    def _reward_stand_still(self):
        error = torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        error[~self.low_vel_mask] = 0.
        return error

    def _reward_contact_no_vel(self):
        contact_feet_vel = self.feet_vel * self.contact_filt.unsqueeze(-1)
        return torch.sum(torch.square(contact_feet_vel[:, :, :3]), dim=(1,2))

    def _reward_hip_close_to_default(self):
        hip_dof_indice = [0, 1, 5, 6]
        return torch.sum(torch.square(self.dof_pos[:, hip_dof_indice] - self.default_dof_pos[:, hip_dof_indice]), dim=-1)

    def _reward_close_to_default(self):
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)