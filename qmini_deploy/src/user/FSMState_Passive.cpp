#include "user/FSMState_Passive.h"
#include <iomanip>

FSMState_Passive::FSMState_Passive(ControlFSMData *data):
                  FSMState(data, FSMStateName::PASSIVE, "passive"){}

void FSMState_Passive::enter()
{
}

void FSMState_Passive::run()
{
    vector<JointCmd> joint_cmds(10);
    for (int i = 0; i < joint_cmds.size(); ++i) {
        joint_cmds[i].kp = 0.0;
        joint_cmds[i].kd = 2.0;
        joint_cmds[i].q = 0.0;
        joint_cmds[i].dq = 0.0;
        joint_cmds[i].tau = 0.0;
    }
    _data->motor_interface->SetJointCmds(joint_cmds);

    // Print joint angles for debugging (If needed)
    bool print_joint_angles = true;
    if (print_joint_angles) {
        vector<JointData> motor_data = _data->motor_interface->GetJointDatas();
        std::cout << "Joint Angles: ";
        for (int i = 0; i < motor_data.size(); ++i) {
            std::cout << std::fixed << std::setprecision(3) << motor_data[i].q << " ";
        }
        std::cout << std::endl;
    }
}

void FSMState_Passive::exit()
{
}

FSMStateName FSMState_Passive::checkTransition()
{
    GamepadData data = _data->gamepad_interface->GetGamepadData();
    if (data.A.pressed) {
        return FSMStateName::STANDUP;
    }
    return FSMStateName::PASSIVE;
}