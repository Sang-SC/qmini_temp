from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class QminiCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 256
        num_actions = 10

        # observation history related
        history_encoding = True
        history_len = 4
        num_proprio = 9 + 3*10 + 4  # proprioceptive.
        num_priv = 3 + 1 + 1 + 3 + 1 + 1 + 1 + 10 + 10 + 10 + 10     # privileged.
        num_est = 3 + 2       # estimator
        num_measure_heights = 187  # terrain height samples
        
        num_observations = num_proprio * (1 + history_len)
        num_privileged_obs = num_proprio * (1 + history_len) + num_priv * (1 + history_len) + num_measure_heights
        num_actor_observations = num_proprio * (1 + history_len) + num_est

    class terrain(LeggedRobotCfg.terrain):
        # curriculum = False
        # mesh_type = 'plane'
        curriculum = True
        mesh_type = 'trimesh'

        measure_heights = True
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.2, 0.2, 0.1, 0.1, 0.4]
        # terrain_proportions = [0.0, 0.05, 0.0, 0.4, 0.05]
        terrain_length = 5.
        terrain_width = 5.
        horizontal_scale = 0.1 # [m]
        num_rows = 10 # number of terrain rows (levels)
        num_cols = 20 # number of terrain cols (types)
        max_init_terrain_level = 0 # starting curriculum state

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.48] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'hip_yaw_l': 0.35,
            'hip_roll_l': -0.05,
            'hip_pitch_l': -1.5,
            'knee_pitch_l': 1.2,
            'ankle_pitch_l': -1.0,
            'hip_yaw_r': -0.35,
            'hip_roll_r': 0.05,
            'hip_pitch_r': 1.5,
            'knee_pitch_r': -1.2,
            'ankle_pitch_r': 1.0,
        }

    class commands(LeggedRobotCfg.commands):
        curriculum = False
        max_curriculum = 0.6
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0 # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-0.4, 0.5] # min max [m/s]
            lin_vel_y = [-0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-0.6, 0.6]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_yaw': 20., 'hip_roll': 60., 'hip_pitch': 30., 'knee': 30., 'ankle': 20.}   # [N*m/rad]
        damping = {'hip_yaw': 0.75, 'hip_roll': 2.0, 'hip_pitch': 1.0, 'knee': 1.0, 'ankle': 0.75}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
  
    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 140.0
        base_height_target = 0.42
        base_roll_target = 0.0
        base_pitch_target = 0.0
        foot_height_offset = 0.095   # height of the foot coordinate origin above ground [m]
        gait_feet_swing_height_target = 0.05

        tracking_sigma = 0.0 # tracking reward = exp(-error^2/sigma)
        lin_vel_x_tracking_sigma = 0.04
        lin_vel_y_tracking_sigma = 0.04
        ang_vel_tracking_sigma = 0.16

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -200.0
            collision = -1.5
            alive = 0.3

            # Tracking
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            tracking_lin_vel_x = 1.2
            tracking_lin_vel_y = 0.1
            tracking_ang_vel = 1.0
            base_height = -40.0
            base_orientation = -1.5

            # Feet
            feet_air_time = 0.0
            contact_no_vel = -2.0
            feet_orientation = -2.0
            feet_contact_forces = -0.005

            # Gait
            gait_match = 0.3
            gait_feet_swing_height = -50.0
            gait_feet_air_time = 4.0
            gait_symmetry = 0.0

            # Other
            stand_still = -1.0
            lin_vel_z = -0.5
            ang_vel_xy = -0.025
            hip_close_to_default = -2.0
            close_to_default = 0.0
            torques = -2.e-5
            dof_vel = -1.e-4
            dof_acc = -1.e-7
            action = -1.e-4
            action_rate = -2.e-3
            base_acc = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_base_mass_range = [-1.0, 3.0]
        push_robots = True
        push_interval_s = 5.0
        push_robots_vel_x = [-0.15, 0.15]
        push_robots_vel_y = [-0.1, 0.1]
        randomize_base_com = True
        added_base_com_x_range = [-0.02, 0.02]
        added_base_com_y_range = [-0.01, 0.01]
        added_base_com_z_range = [-0.01, 0.01]
        randomize_joint_friction = True
        joint_friction_range = [0.0001, 0.01]
        randomize_joint_damping = True
        joint_damping_range = [0.001, 0.01]
        randomize_joint_armature = True
        joint_armature_range = [0.001, 0.02]
        randomize_pd_gain = True
        p_gain_scale_range = [0.8, 1.2]
        d_gain_scale_range = [0.8, 1.2]
        randomize_torque = True
        torque_scale_range = [0.8, 1.2]
        randomize_motor_zero_deviation = False
        motor_zero_deviation_range = [-0.026, 0.026]

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/qmini/urdf/qmini.urdf'
        name = "qmini"
        foot_name = "ankle_pitch"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = ["base", "hip", "knee"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False # Some .obj meshes must be flipped from y-up to z-up

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0 # scales other values
        class noise_scales:
            dof_pos = 0.02
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt =  0.005
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            max_gpu_contact_pairs = 2**24 #2**24 -> needed for 8000 envs and more

    class custom:
        low_vel_thresholds = [0.05, 0.05, 0.1]

        reset_dof_pos_flag = True
        reset_dof_pos_ranges = [[-0.1, 0.1],
                                [-0.1, 0.1],
                                [-0.2, 0.2],
                                [-0.2, 0.2],
                                [-0.2, 0.2],
                                [-0.1, 0.1],
                                [-0.1, 0.1],
                                [-0.2, 0.2],
                                [-0.2, 0.2],
                                [-0.2, 0.2]]
        reset_dof_vel_flag = True
        reset_dof_vel_ranges = [[-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0],
                                 [-0.0, 0.0]]
        reset_base_vel_flag = True
        reset_base_vel_ranges = [[-0.1, 0.1],
                                 [-0.1, 0.1],
                                 [-0.1, 0.1]]
        reset_base_ang_flag = True
        reset_base_ang_ranges = [[-0.1, 0.1],
                                 [-0.1, 0.1],
                                 [-0.1, 0.1]]
        reset_base_rpy_flag = True
        reset_base_rpy_ranges = [[-0.1, 0.1],
                                 [-0.1, 0.1],
                                 [-0.1, 0.1]]
        class gait:
            period = 0.5

class QminiCfgPPO( LeggedRobotCfgPPO ):
    seed = 52
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'qmini'
        save_interval = 100
        num_steps_per_env = 24 # per iteration
        max_iterations = 10000
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model

    class estimator:
        input_dim = QminiCfg.env.num_observations
        output_dim = QminiCfg.env.num_est
        buffer_capacity = 4096 * 6
        batch_size = 1024
        lr = 2.e-3
        num_epochs = 1