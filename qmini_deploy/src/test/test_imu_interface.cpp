#include <chrono>
#include <thread>
#include "user/imu_interface.h"

int main() {
    IMUInterface imu_interface;
    imu_interface.Init("/dev/ttyUSB0", 921600);
    imu_interface.print_flag = true;  // true or false

    while (true) {
        imu_interface.Update();
    }

    return 0;
}