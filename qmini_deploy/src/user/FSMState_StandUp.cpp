#include "user/FSMState_StandUp.h"
#include <iomanip>

float JointLinearInterpolation(float init_pos, float target_pos, float rate)
{
    float p;
    rate = std::min(std::max(rate, 0.0f), 1.0f);
    p = init_pos * (1 - rate) + target_pos * rate;
    return p;
}

FSMState_StandUp::FSMState_StandUp(ControlFSMData *data):
                  FSMState(data, FSMStateName::STANDUP, "standup"){
    joint_enter_q.resize(num_joints);
    joint_standup_q.resize(num_joints);
    joint_kp.resize(num_joints);
    joint_kd.resize(num_joints);
    try {
        YAML::Node config = *_data->config;
        for (int i = 0; i < num_joints; i++) {
            joint_standup_q[i] = config["FSMState_StandUp"]["joint_standup_q"][i].as<float>();
            joint_kp[i] = config["FSMState_StandUp"]["joint_kp"][i].as<float>();
            joint_kd[i] = config["FSMState_StandUp"]["joint_kd"][i].as<float>();
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading FSMState_StandUp parameters from config: " << e.what() << std::endl;
        std::exit(-1);
    }
}

void FSMState_StandUp::enter()
{
    vector<JointData> joint_datas = _data->motor_interface->GetJointDatas();
    for (int i = 0; i < 10; i++)
    {
        joint_enter_q[i] = joint_datas[i].q;
    }
    rate_count = 0;
}

void FSMState_StandUp::run()
{
    int rate_count_max = 200;
    if (rate_count < rate_count_max) {
        rate_count++;
    }
    float rate = rate_count / (float)rate_count_max;

    vector<JointCmd> joint_cmds(10);
    for (int i = 0; i < joint_cmds.size(); ++i) {
        joint_cmds[i].kp = joint_kp[i];
        joint_cmds[i].kd = joint_kd[i];
        joint_cmds[i].q = JointLinearInterpolation(joint_enter_q[i], joint_standup_q[i], rate);
        joint_cmds[i].dq = 0.0;
        joint_cmds[i].tau = 0.0;
    }
    _data->motor_interface->SetJointCmds(joint_cmds);
}

void FSMState_StandUp::exit()
{
}

FSMStateName FSMState_StandUp::checkTransition()
{
    GamepadData data = _data->gamepad_interface->GetGamepadData();
    if (data.L2.pressed && data.B.pressed) {
        return FSMStateName::PASSIVE;
    }
    if (data.X.pressed) {
        return FSMStateName::WALK;
    }
    return FSMStateName::STANDUP;
}