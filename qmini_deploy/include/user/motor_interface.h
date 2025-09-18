#ifndef MOTOR_INTERFACE_H
#define MOTOR_INTERFACE_H

#include "user/type.h"
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include <yaml-cpp/yaml.h>
#include <mutex>

struct JointCmd {
    float tau;
    float q;
    float dq;
    float kp;
    float kd;
};

struct JointData {
    float q;
    float dq;              
};

class MotorInterface {
public:
    MotorInterface(){}

    ~MotorInterface() {}

    int Init(vector<string> port_names) {
        // Open serial ports
        if (port_names.size() != 4) {
            std::cerr << "Error: Expected 4 serial ports, but got " << port_names.size() << std::endl;
            exit(-1);
        }
        for (int i = 0; i < port_names.size(); ++i) {
            try {
                serial_ports.emplace_back(std::make_unique<SerialPort>(port_names[i]));
            } catch (const std::exception& e) {
                std::cerr << "SerialPort initialization failed for " << port_names[i] << ": " << e.what() << std::endl;
                std::cerr << "Please check:" << std::endl;
                std::cerr << "1. Is the serial device connected?" << std::endl;
                std::cerr << "2. Is the port name correct?" << std::endl;
                std::cerr << "3. Do you have permission to access the port?" << std::endl;
                exit(-1);
            }
        }

        // Initialize motor IDs and joint IDs
        motor_ids = {
            {0, 1},       // Port 0
            {0, 1},       // Port 1
            {0, 1, 2},    // Port 2
            {0, 1, 2},    // Port 3
        };
        joint_ids = {
            {0, 5},       // Port 0
            {1, 6},       // Port 1
            {2, 3, 4},    // Port 2
            {7, 8, 9},    // Port 3
        };

        // Calculate total number of motors
        motor_num = 0;
        for (int i = 0; i < motor_ids.size(); ++i) {
            motor_num += motor_ids[i].size();
        }

        // Set up motor commands and data structures
        MotorType motor_type = MotorType::GO_M8010_6;
        unsigned short motor_mode  = queryMotorMode(motor_type, MotorMode::FOC);
        motor_cmds.resize(motor_num);
        motor_datas.resize(motor_num);
        motor_datas_init_flag.resize(motor_num);
        motor_datas_init_q.resize(motor_num);
        for (int i = 0; i < joint_ids.size(); ++i) {
            for (int j = 0; j < joint_ids[i].size(); ++j) {
                int joint_id = joint_ids[i][j];
                motor_cmds[joint_id].motorType = motor_type;
                motor_cmds[joint_id].mode = motor_mode;
                motor_cmds[joint_id].id = motor_ids[i][j];
                motor_cmds[joint_id].kp = 0.0;
                motor_cmds[joint_id].kd = 0.0;
                motor_cmds[joint_id].q = 0.0;
                motor_cmds[joint_id].dq = 0.0;
                motor_cmds[joint_id].tau = 0.0;

                motor_datas[joint_id].motorType = motor_type;
                motor_datas[joint_id].q = 0.0;
                motor_datas[joint_id].dq = 0.0;

                motor_datas_init_flag[joint_id] = 0;

                motor_datas_init_q[joint_id] = 0.0;
            }
        }
        return 0;
    }

    vector<JointData> GetJointDatas() {
        // Ensure motor_datas is initialized
        for (int i = 0; i < motor_num; ++i) {
            if (motor_datas_init_flag[i] == 0) {
                std::cerr << "Error: Motor data for joint " << i << " is not initialized." << std::endl;
                exit(-1);
            }
        }

        // Lock the mutex to safely access motor data
        std::lock_guard<std::mutex> lock(motor_mutex);
        vector<JointData> joint_datas(motor_num);
        for (int i = 0; i < motor_num; ++i) {
            joint_datas[i].q = (motor_datas[i].q - motor_datas_init_q[i]) / joint_gear_ratios[i] + joint_datas_init_q[i];
            joint_datas[i].dq = motor_datas[i].dq / joint_gear_ratios[i];
        }
        return joint_datas;
    }

    void SetJointCmds(const vector<JointCmd>& joint_cmds) {
        // Ensure motor_datas is initialized
        for (int i = 0; i < motor_num; ++i) {
            if (motor_datas_init_flag[i] == 0) {
                std::cerr << "Error: Motor data for joint " << i << " is not initialized." << std::endl;
                exit(-1);
            }
        }

        // Lock the mutex to safely update motor commands
        std::lock_guard<std::mutex> lock(motor_mutex);
        if (joint_cmds.size() != motor_num) {
            std::cerr << "Invalid cmd size!" << std::endl;
            return;
        }
        for (int i = 0; i < motor_num; ++i) {
            motor_cmds[i].kp = joint_cmds[i].kp / (joint_gear_ratios[i] * joint_gear_ratios[i]);
            motor_cmds[i].kd = joint_cmds[i].kd / (joint_gear_ratios[i] * joint_gear_ratios[i]);
            motor_cmds[i].q = (joint_cmds[i].q - joint_datas_init_q[i]) * joint_gear_ratios[i] + motor_datas_init_q[i];
            motor_cmds[i].dq = joint_cmds[i].dq * joint_gear_ratios[i];
        }
    }

    void UpdatePort(int port_index) {
        SerialPort& serial = *serial_ports[port_index];

        for (int i = 0; i < joint_ids[port_index].size(); ++i) {
            int joint_id = joint_ids[port_index][i];

            MotorCmd motor_cmd_temp;
            {
                std::lock_guard<std::mutex> lock(motor_mutex);
                motor_cmd_temp = motor_cmds[joint_id];
            }

            MotorData motor_data_temp;
            motor_data_temp.motorType = motor_cmd_temp.motorType;
            bool res = serial.sendRecv(&motor_cmd_temp, &motor_data_temp);
            if (!res) {
                std::cerr << "Send/Recv failed for joint_id " << joint_id 
                            << " on port index " << port_index << std::endl;
                return;
            }

            {
                std::lock_guard<std::mutex> lock(motor_mutex);
                motor_datas[joint_id] = motor_data_temp;

                // Check if motor data is initialized
                //! The GO_M8010_6 motor has only a single encoder. We can only use relative position. So we need to initialize the zero position.
                if (motor_datas_init_flag[joint_id] == 0) {
                    motor_datas_init_q[joint_id] = motor_datas[joint_id].q;
                    motor_datas_init_flag[joint_id] = 1;
                }
            }

            // Print the current state of the motor (if needed)
            if (print_flag) {
                std::cout << "------------------------" << std::endl;
                std::cout << "Serial Port Index: " << port_index << std::endl;
                std::cout << "Joint ID: " << joint_id << std::endl;
                std::cout << "Motor ID: " << motor_cmds[joint_id].id << std::endl;
                std::cout << "Motor cmd q: " << motor_cmds[joint_id].q << std::endl;
                std::cout << "Motor init_q: " << motor_datas_init_q[joint_id] << std::endl;
                std::cout << "Motor data q: " << motor_datas[joint_id].q << std::endl;
                std::cout << "Motor data q - init_q: " << motor_datas[joint_id].q - motor_datas_init_q[joint_id] << std::endl;
                std::cout << "Motor data dq: " << motor_datas[joint_id].dq << std::endl;
                std::cout << "Motor temp: " << motor_datas[joint_id].temp << std::endl;
                std::cout << "Motor merror: " << motor_datas[joint_id].merror << std::endl;
            }
        }
    }

    bool print_flag = false;

private:
    vector<std::unique_ptr<SerialPort>> serial_ports;
    vector<vector<int>> motor_ids;
    vector<vector<int>> joint_ids;

    int motor_num;
    vector<MotorCmd> motor_cmds;
    vector<MotorData> motor_datas;
    std::mutex motor_mutex;
    vector<int> motor_datas_init_flag;
    vector<float> motor_datas_init_q;
    // The Qmini robot needs to start from a prone posture. 
    // The corresponding urdf joint angles for this position are joint_datas_init_q
    vector<float> joint_datas_init_q = {-0.05, -0.3, -0.45, 0.0, 0.0,
                                        0.05, 0.3, 0.45, 0.0, 0.0};
    vector<float> joint_gear_ratios = {6.33, 6.33 * 3.0, 6.33, 6.33, 6.33,
                                       6.33, 6.33 * 3.0, 6.33, 6.33, 6.33};
};

#endif // MOTOR_INTERFACE_H