#include "user/FSMState_Walk.h"
#include <filesystem> 

Vec3<float> GetGravityOrientation(const Vec4<float>& quaternion) {
    float qx = quaternion[0];
    float qy = quaternion[1];
    float qz = quaternion[2];
    float qw = quaternion[3];

    Vec3<float> gravity_orientation;
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy);
    gravity_orientation[1] = -2 * (qz * qy + qw * qx);
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz);
    return gravity_orientation;
}

FSMState_Walk::FSMState_Walk(ControlFSMData *data):
                  FSMState(data, FSMStateName::WALK, "walk"){
    command_scale << obs_scales_lin_vel, obs_scales_lin_vel, obs_scales_ang_vel;

    qj = DVec<float>::Zero(num_joints);
    dqj = DVec<float>::Zero(num_joints);

    action = DVec<float>::Zero(num_actions);
    obs_current = DVec<float>::Zero(num_obs_per_step);
    obs_history = DMat<float>::Zero(history_len, num_obs_per_step);
    obs_input = DVec<float>::Zero(num_obs);

    // Initialize ONNX Runtime
    string onnx_model_dir_path = std::filesystem::path(__FILE__).parent_path().string() + "/../../onnx_models";
    string estimator_model_path = onnx_model_dir_path + "/qmini_estimator.onnx";
    string actor_model_path = onnx_model_dir_path + "/qmini_actor.onnx";

    env_estimator = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_estimator");
    session_options_estimator = std::make_unique<Ort::SessionOptions>();
    session_estimator = std::make_unique<Ort::Session>(*env_estimator, estimator_model_path.c_str(), *session_options_estimator);
    onnx_inference_estimator = std::make_unique<OnnxInference>();
    onnx_inference_estimator->init(num_obs, num_estimator_output);

    env_actor = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "onnxruntime_actor");
    session_options_actor = std::make_unique<Ort::SessionOptions>();
    session_actor = std::make_unique<Ort::Session>(*env_actor, actor_model_path.c_str(), *session_options_actor);
    onnx_inference_actor = std::make_unique<OnnxInference>();
    onnx_inference_actor->init(num_actor_observations, num_actions);

    // Others
    try {
        YAML::Node config = *_data->config;
        low_vel_flags = {0, 0, 0};
        default_angles = DVec<float>::Zero(num_joints);
        for (int i = 0; i < num_joints; i++) {
            default_angles[i] = config["FSMState_Walk"]["default_joint_angles"][i].as<float>();
        }
        joint_kp.resize(num_joints);
        joint_kd.resize(num_joints);
        for (int i = 0; i < num_joints; i++) {
            joint_kp[i] = config["FSMState_Walk"]["joint_kp"][i].as<float>();
            joint_kd[i] = config["FSMState_Walk"]["joint_kd"][i].as<float>();
        }
        low_vel_thresholds = {config["FSMState_Walk"]["low_vel_thresholds"][0].as<float>(),
                            config["FSMState_Walk"]["low_vel_thresholds"][1].as<float>(),
                            config["FSMState_Walk"]["low_vel_thresholds"][2].as<float>()};
        lin_vel_x_range = {config["FSMState_Walk"]["lin_vel_x_range"][0].as<float>(),
                        config["FSMState_Walk"]["lin_vel_x_range"][1].as<float>()};
        lin_vel_y_range = {config["FSMState_Walk"]["lin_vel_y_range"][0].as<float>(),
                        config["FSMState_Walk"]["lin_vel_y_range"][1].as<float>()};
        ang_vel_yaw_range = {config["FSMState_Walk"]["ang_vel_yaw_range"][0].as<float>(),
                            config["FSMState_Walk"]["ang_vel_yaw_range"][1].as<float>()};
        gait_period = config["FSMState_Walk"]["gait_period"].as<float>();
        gait_offsets.resize(2);
        for (int i = 0; i < gait_offsets.size(); i++) {
            gait_offsets[i] = config["FSMState_Walk"]["gait_offsets"][i].as<float>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading FSMState_Walk parameters from config: " << e.what() << std::endl;
        std::exit(-1);
    }

}

void FSMState_Walk::enter()
{
    std::cout << "Enter FSMState_Walk" << std::endl;

    phase_counter = 0;
    obs_init_flag = 0;
    obs_history.setZero();
}

void FSMState_Walk::update_leg_phase() {
    // Check for dynamic stand toggle
    GamepadData data = _data->gamepad_interface->GetGamepadData();
    if (data.X.pressed) {
        dynamic_stand_flag = 0;
    }
    if (data.Y.pressed) {
        dynamic_stand_flag = 1;
    }

    // Update phase counter
    if (dynamic_stand_flag == 1) {
        phase_counter++;
    } else {
        // If velocity commands are low, wait for phase to be near 0 to reset
        if (low_vel_flags[0] && low_vel_flags[1] && low_vel_flags[2]) {
            if (std::abs(phase - 0.0) < 0.05) {
                static_stand_phase_ready_flag = 1;
            }
        } else {
            static_stand_phase_ready_flag = 0;
        }
        if (static_stand_phase_ready_flag == 1) {
            phase_counter = 0;
        } else {
            phase_counter++;
        }
    }

    // Update leg phases and clock inputs
    phase = std::fmod(phase_counter * dt, gait_period) / gait_period;
    leg_phases[0] = std::fmod(phase + gait_offsets[0], 1.0);
    leg_phases[1] = std::fmod(phase + gait_offsets[1], 1.0);
    for (int i = 0; i < 2; ++i) {
        clock_input[i] = std::sin(2 * M_PI * leg_phases[i]);
        clock_input[i + 2] = std::cos(2 * M_PI * leg_phases[i]);
    }
}

void FSMState_Walk::run()
{
    // std::cout << "Running FSMState_Walk" << std::endl;

    auto current_time = std::chrono::steady_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - run_time_prev).count();
    run_time_prev = current_time;
    if (elapsed_time >= 25) {
        cout << "Warning: FSMState_Walk run time exceeded 25 ms, elapsed time: " << elapsed_time << " ms" << endl;
    }
    // cout << "FSMState_Walk running, elapsed time: " << elapsed_time << " ms" << endl;

    // Update 
    IMUData imu_data = _data->imu_interface->GetIMUDataTransformed();
    ang_vel = imu_data.gyroscope;
    gravity_orientation = GetGravityOrientation(imu_data.quaternion);

    // Update command
    GamepadData gamepad_data_temp = _data->gamepad_interface->GetGamepadData();
    cmd[0] = -1.0 * gamepad_data_temp.ly;
    cmd[1] = -1.0 * gamepad_data_temp.lx;
    cmd[2] = -1.0 * gamepad_data_temp.rx;
    cmd[0] *= std::max(std::abs(lin_vel_x_range[0]), std::abs(lin_vel_x_range[1]));
    cmd[1] *= std::max(std::abs(lin_vel_y_range[0]), std::abs(lin_vel_y_range[1]));
    cmd[2] *= std::max(std::abs(ang_vel_yaw_range[0]), std::abs(ang_vel_yaw_range[1]));
    cmd[0] = std::max(std::min(cmd[0], lin_vel_x_range[1]), lin_vel_x_range[0]);
    cmd[1] = std::max(std::min(cmd[1], lin_vel_y_range[1]), lin_vel_y_range[0]);
    cmd[2] = std::max(std::min(cmd[2], ang_vel_yaw_range[1]), ang_vel_yaw_range[0]);

    // Set small commands to zero
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(cmd[i]) <= low_vel_thresholds[i]) {
            cmd[i] = 0.0;
            low_vel_flags[i] = 1;
        } else {
            low_vel_flags[i] = 0;
        }
    }

    // Update joint positions and velocities
    vector<JointData> joint_datas = _data->motor_interface->GetJointDatas();
    for (int i = 0; i < num_joints; ++i) {
        qj[i] = joint_datas[i].q;
        dqj[i] = joint_datas[i].dq;
    }

    // Update leg phases
    update_leg_phase();

    // Update the current observation
    obs_current.head<3>() = ang_vel * obs_scales_ang_vel;
    obs_current.segment<3>(3) = gravity_orientation;
    obs_current.segment<3>(6) = cmd.cwiseProduct(command_scale);
    obs_current.segment<10>(9) = (qj - default_angles) * obs_scales_dof_pos;
    obs_current.segment<10>(19) = dqj * obs_scales_dof_vel;
    obs_current.segment<10>(29) = action;
    obs_current.segment<4>(39) = clock_input;
    // cout << "obs_current: " << obs_current << endl;
    // cout << "qj: " << qj.transpose() << endl;

    // Initialize the history observation
    if (obs_init_flag == 0) {
        for (int i = 0; i < history_len; ++i) {
            obs_history.row(i) = obs_current;
        }
        obs_init_flag = 1;
    }

    // Update the input observation
    obs_input.head(num_obs_per_step) = obs_current;
    for (int i = 0; i < history_len; ++i) {
        obs_input.segment(num_obs_per_step * (i + 1), num_obs_per_step) = obs_history.row(i);
    }

    // Update the observation history
    for (int i = history_len - 1; i > 0; --i) {
        obs_history.row(i) = obs_history.row(i - 1);
    }
    obs_history.row(0) = obs_current;

    // Onnxruntime inference
    // 1. Estimator inference
    DVec<float> estimator_output = onnx_inference_estimator->inference(session_estimator.get(), obs_input);
    // cout << "Estimator output: " << estimator_output.transpose() << endl;
    // 2. Actor inference
    DVec<float> actor_input = DVec<float>::Zero(num_actor_observations);
    actor_input.head(num_obs) = obs_input;
    actor_input.segment(num_obs, num_estimator_output) = estimator_output;
    action = onnx_inference_actor->inference(session_actor.get(), actor_input);
    // cout << "Action: " << action.transpose() << endl;

    // Convert action to target joint positions
    action_scale = 0.25;
    DVec<float> target_dof_pos = default_angles + action * action_scale;

    vector<JointCmd> joint_cmds(10);
    for (int i = 0; i < joint_cmds.size(); ++i) {
        joint_cmds[i].kp = joint_kp[i];
        joint_cmds[i].kd = joint_kd[i];
        joint_cmds[i].q = target_dof_pos[i];
        joint_cmds[i].dq = 0.0;
        joint_cmds[i].tau = 0.0;
    }
    _data->motor_interface->SetJointCmds(joint_cmds);

    // cout << "Ang Vel: " << ang_vel.transpose() << endl;
    // cout << "Gravity Orientation: " << gravity_orientation.transpose() << endl;
    cout << "Command: " << cmd.transpose() << endl;
    // cout << "default_angles: " << default_angles.transpose() << endl;
    // cout << "Target DOF Positions: " << target_dof_pos.transpose() << endl;
}

void FSMState_Walk::exit()
{
    std::cout << "Exiting FSMState_Walk" << std::endl;  
}

FSMStateName FSMState_Walk::checkTransition()
{
    GamepadData data = _data->gamepad_interface->GetGamepadData();
    if (data.L2.pressed && data.B.pressed) {
        return FSMStateName::PASSIVE;
    }
    if (data.A.pressed) {
        return FSMStateName::STANDUP;
    }

    // Safety check
    IMUData imu_data = _data->imu_interface->GetIMUDataTransformed();
    if (std::fabs(imu_data.rpy[0]) > M_PI / 9 || std::fabs(imu_data.rpy[1]) > M_PI / 9) {
        std::cout << "IMU Roll or Pitch exceeded 20 degrees! Exiting to PASSIVE state." << std::endl;
        return FSMStateName::PASSIVE;
    }

    return FSMStateName::WALK;
}