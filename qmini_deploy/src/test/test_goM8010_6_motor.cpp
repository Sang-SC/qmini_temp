#include <unistd.h>
#include "serialPort/SerialPort.h"
#include "unitreeMotor/unitreeMotor.h"
#include <cmath>
#include <iostream>

using std::cout, std::endl;
using std::sin, std::cos;

int main() {
    SerialPort  serial("/dev/ttyUSB0");

    MotorCmd cmd;
    cmd.motorType = MotorType::GO_M8010_6;
    cmd.mode = queryMotorMode(MotorType::GO_M8010_6,MotorMode::FOC);

    MotorData data;
    data.motorType = MotorType::GO_M8010_6;

    float gear_ratio = queryGearRatio(MotorType::GO_M8010_6);
    float kp_output = 2.0;
    float kd_output = 0.2;
    float q_output = 0.0;
    float dq_output = 0.0;
    
    float t = 0.0;
    float dt = 0.001;
    while(true) 
    {
        q_output = M_PI * sin(2 * M_PI * 0.25 * t);
        dq_output = M_PI * 0.25 * cos(2 * M_PI * 0.25 * t);

        cmd.id   = 0;
        cmd.kp   = kp_output / (gear_ratio * gear_ratio);
        cmd.kd   = kd_output / (gear_ratio * gear_ratio);
        cmd.q    = q_output * gear_ratio;
        cmd.dq   = dq_output * gear_ratio;
        cmd.tau  = 0.0;

        serial.sendRecv(&cmd,&data);

        usleep(dt * 1e6);
        t += dt;

        cout << "\nt: " << t << endl;
        cout << "q_output: " << q_output << endl;
        cout << "motor.q: " << data.q << endl;
        cout << "motor.temp: " << data.temp << endl;
        cout << "motor.merror: " << data.merror << endl;
    }
}