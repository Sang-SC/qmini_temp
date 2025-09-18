#ifndef CONTROLFSMDATA_H
#define CONTROLFSMDATA_H

#include "user/gamepad_interface.h"
#include "user/imu_interface.h"
#include "user/motor_interface.h"

struct ControlFSMData {
    GamepadInterface* gamepad_interface;
    IMUInterface* imu_interface;
    MotorInterface* motor_interface;
    YAML::Node* config;
};

#endif  // CONTROLFSMDATA_H
