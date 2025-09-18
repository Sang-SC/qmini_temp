#ifndef IMU_INTERFACE_H
#define IMU_INTERFACE_H

#include <libserial/SerialPort.h>
#include "user/type.h"
#include "user/math_utils.h"
#include <mutex>
#include <cmath>
#include <cstdlib>  // for exit()
#include <iostream>

// Define constants for IMU data types and lengths
const uint8_t FRAME_HEAD = 0xFC;
const uint8_t TYPE_IMU = 0x40;
const uint8_t TYPE_AHRS = 0x41;
const uint8_t IMU_LEN = 0x38; // 56 bytes
const uint8_t AHRS_LEN = 0x30; // 48 bytes

class IMUData {
public:
    Vec3<float> gyroscope;     // Unit: rad/s
    Vec3<float> accelerometer; // Unit: m/s^2
    Vec4<float> quaternion;    // Order: x, y, z, w
    Vec3<float> rpy;           // Unit: rad
    uint8_t temperature;       // Unit: degree Celsius
};

class IMUInterface {
public:
    IMUInterface() {}

    ~IMUInterface() {}

    int Init(string port, int baudrate) {
        // Open the serial port for IMU communication
        try {
            serial.Open(port);
            serial.SetBaudRate(LibSerial::BaudRate::BAUD_921600);
            serial.SetCharacterSize(LibSerial::CharacterSize::CHAR_SIZE_8);
            serial.SetParity(LibSerial::Parity::PARITY_NONE);
            serial.SetStopBits(LibSerial::StopBits::STOP_BITS_1);
            serial.SetFlowControl(LibSerial::FlowControl::FLOW_CONTROL_NONE);
            std::cout << "Serial port " << port << " opened with baudrate: " << baudrate << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "IMU serial port open error: " << e.what() << std::endl;
            std::cerr << "Please check:" << std::endl;
            std::cerr << "1. Is the IMU connected?" << std::endl;
            std::cerr << "2. Is the port name and baudrate correct?" << std::endl;
            std::cerr << "3. Do you have permission to access the port?" << std::endl;
            exit(-1);
        }

        return 0;
    }

    void Update() {
        // Check if the serial port is open
        if (!serial.IsOpen()) {
            std::cerr << "IMU disconnected!" << std::endl;
            exit(-1);
            return;
        }

        try {
            // Read the frame head, type, and length
            uint8_t check_head;
            serial.ReadByte(check_head);
            if (check_head != FRAME_HEAD) {
                return;
            }
            uint8_t head_type;
            serial.ReadByte(head_type);
            if (head_type != TYPE_IMU && head_type != TYPE_AHRS) {
                return;
            }
            uint8_t check_len;
            serial.ReadByte(check_len);
            if ((head_type == TYPE_IMU && check_len != IMU_LEN) ||
                (head_type == TYPE_AHRS && check_len != AHRS_LEN)) {
                std::cerr << "Data length check failed, type: 0x" << std::hex << (int)head_type 
                          << " length: 0x" << (int)check_len << std::endl;
                return;
            }

            // Skip sequence number and checksum bytes
            uint8_t dummy;
            serial.ReadByte(dummy); // check_sn
            serial.ReadByte(dummy); // head_crc8
            serial.ReadByte(dummy); // crc16_H
            serial.ReadByte(dummy); // crc16_L

            // Read and parse data based on type
            if (head_type == TYPE_IMU) {
                // Read data
                std::vector<uint8_t> data_raw(IMU_LEN);
                serial.Read(data_raw, IMU_LEN);
                float data_part1[12];
                int32_t data_part2[2];
                std::memcpy(data_part1, data_raw.data(), sizeof(float) * 12);
                std::memcpy(data_part2, data_raw.data() + sizeof(float) * 12, sizeof(int32_t) * 2);

                // Lock the mutex to safely update IMU data
                {
                    std::lock_guard<std::mutex> lock(imu_mutex);
                    imu_data.gyroscope = Vec3<float>(data_part1[0], data_part1[1], data_part1[2]);
                    imu_data.accelerometer = Vec3<float>(data_part1[3], data_part1[4], data_part1[5]);
                    imu_data.temperature = static_cast<uint8_t>(data_part1[9]);
                }
            } else if (head_type == TYPE_AHRS) {
                // Read data
                std::vector<uint8_t> data_raw(AHRS_LEN);
                serial.Read(data_raw, AHRS_LEN);
                float data_part1[10];
                int32_t data_part2[2];
                std::memcpy(data_part1, data_raw.data(), sizeof(float) * 10);
                std::memcpy(data_part2, data_raw.data() + sizeof(float) * 10, sizeof(int32_t) * 2);

                // Lock the mutex to safely update IMU data
                {
                    std::lock_guard<std::mutex> lock(imu_mutex);
                    imu_data.rpy = Vec3<float>(data_part1[3], data_part1[4], data_part1[5]);

                    // The quaternion is stored as (w, x, y, z) in the data, so we need to rearrange it
                    imu_data.quaternion = Vec4<float>(data_part1[7], data_part1[8], data_part1[9], data_part1[6]);

                    // Print the current state of the IMU (if needed)
                    if (print_flag) {
                        std::cout << "IMU Data: " << std::endl;
                        std::cout << "Gyroscope (rad/s): " << imu_data.gyroscope.transpose() << std::endl;
                        std::cout << "Accelerometer (m/s^2): " << imu_data.accelerometer.transpose() << std::endl;
                        std::cout << "Quaternion (x, y, z, w): " << imu_data.quaternion.transpose() << std::endl;
                        std::cout << "RPY (rad): " << imu_data.rpy.transpose() << std::endl;
                        std::cout << "Temperature (°C): " << static_cast<int>(imu_data.temperature) << std::endl;
                        std::cout << std::endl;
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return;
        }
    }

    IMUData GetIMUDataRaw() {
        IMUData imu_data_raw;
        {
            // Lock the mutex to safely access IMU data
            std::lock_guard<std::mutex> lock(imu_mutex);
            imu_data_raw = imu_data;
        }
        return imu_data_raw;
    }

    IMUData GetIMUDataTransformed() {
        IMUData imu_data_raw;
        {
            // Lock the mutex to safely access IMU data
            std::lock_guard<std::mutex> lock(imu_mutex);
            imu_data_raw = imu_data;
        }

        // The robot coordinate system is inconsistent with the IMU coordinate system,
        // we need to convert the IMU data to the robot coordinate system.
        IMUData imu_data_transformed;
        imu_data_transformed.accelerometer[0] = -imu_data_raw.accelerometer[1];
        imu_data_transformed.accelerometer[1] = -imu_data_raw.accelerometer[0];
        imu_data_transformed.accelerometer[2] = -imu_data_raw.accelerometer[2];
        imu_data_transformed.gyroscope[0] = -imu_data_raw.gyroscope[1];
        imu_data_transformed.gyroscope[1] = -imu_data_raw.gyroscope[0];
        imu_data_transformed.gyroscope[2] = -imu_data_raw.gyroscope[2];
        imu_data_transformed.rpy[0] = -imu_data_raw.rpy[1];
        imu_data_transformed.rpy[1] = -imu_data_raw.rpy[0];
        imu_data_transformed.rpy[2] = -imu_data_raw.rpy[2];
        Vec3<float> euler_angle_tem = Vec3<float>(imu_data_transformed.rpy[2], imu_data_transformed.rpy[1], imu_data_transformed.rpy[0]); // Convert to roll, pitch, yaw
        imu_data_transformed.quaternion = Euler2Quat(euler_angle_tem, Order::ZYX);

        imu_data_transformed.temperature = imu_data_raw.temperature;
        return imu_data_transformed;
    }

    bool print_flag = false;

private:
    LibSerial::SerialPort serial;
    IMUData imu_data;
    std::mutex imu_mutex;
};

#endif // IMU_INTERFACE_H