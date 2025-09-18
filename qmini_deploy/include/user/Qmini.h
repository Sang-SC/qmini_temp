#ifndef QMINI_H
#define QMINI_H

#include <unitree/common/thread/recurrent_thread.hpp>
#include "user/FSM.h"
#include "user/motor_interface.h"
#include "user/gamepad_interface.h"
#include "user/imu_interface.h"
#include <yaml-cpp/yaml.h>
#include <filesystem> 
namespace fs = std::filesystem;

class Qmini {
public:
    Qmini() {
        string config_path = fs::path(__FILE__).parent_path().string() + "/../../config/qmini_config.yaml";
        config = YAML::LoadFile(config_path);
        cout << "Loaded configuration from: " << config_path << endl;

        control_fsm_data_ = new ControlFSMData();
        control_fsm_data_->gamepad_interface = &gamepad_interface;
        control_fsm_data_->imu_interface = &imu_interface;
        control_fsm_data_->motor_interface = &motor_interface;
        control_fsm_data_->config = &config;
        fsm = new FSM(control_fsm_data_);
    }

    ~Qmini() {
        delete control_fsm_data_;
        delete fsm;
    }

    void Run() {
        // Motor threads
        vector<string> motor_port_names = config["motor"]["port_names"].as<vector<string>>();
        motor_interface.Init(motor_port_names);
        motor_interface.print_flag = false;  // true or false
        vector<int> motor_cpu_ids = {0, 0, 1, 1};
        for (int i = 0; i < motor_port_names.size(); ++i) {
            unitree::common::ThreadPtr thread_ptr = unitree::common::CreateRecurrentThreadEx(
                "motor_serial_port_" + std::to_string(i), motor_cpu_ids[i], 0.002 * 1e6, 
                &MotorInterface::UpdatePort, &motor_interface, i);
            motor_interface_thread_ptrs.push_back(thread_ptr);
        }

        // Gamepad thread
        int gamepad_cpu_id = 3;
        gamepad_interface.Init();
        gamepad_interface.SetSmooth(0.02);
        gamepad_interface.print_flag = false;  // true or false
        gamepad_interface_thread_ptr = unitree::common::CreateRecurrentThreadEx("gamepad", gamepad_cpu_id, 0.02 * 1e6, &GamepadInterface::Update, &gamepad_interface);

        // IMU thread (No interval loop)
        int imu_cpu_id = 2;
        imu_interface.Init(config["imu"]["port"].as<string>(), config["imu"]["baudrate"].as<int>());
        imu_interface.print_flag = false;  // true or false
        imu_interface_thread_ptr = unitree::common::CreateRecurrentThreadEx("imu", imu_cpu_id, 0, &IMUInterface::Update, &imu_interface);

        // FSM thread
        int fsm_cpu_id = 3;
        fsm_thread_ptr = unitree::common::CreateRecurrentThreadEx("fsm", fsm_cpu_id, 0.02 * 1e6, &FSM::run, fsm);

        while (1)
        {
            std::this_thread::sleep_for(chrono::milliseconds(100));
        }
    }

private:
    YAML::Node config;

    MotorInterface motor_interface;
    vector<unitree::common::ThreadPtr> motor_interface_thread_ptrs;

    GamepadInterface gamepad_interface;
    unitree::common::ThreadPtr gamepad_interface_thread_ptr;

    IMUInterface imu_interface;
    unitree::common::ThreadPtr imu_interface_thread_ptr;

    ControlFSMData *control_fsm_data_;
    FSM *fsm;
    unitree::common::ThreadPtr fsm_thread_ptr;
};

#endif // QMINI_H