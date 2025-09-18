#include <chrono>
#include <thread>
#include "user/gamepad_interface.h"
#include <csignal>

void signal_handler(int signal) {
    if (signal == SIGINT) {
        exit(0);
    }
}

int main() {
    std::signal(SIGINT, signal_handler);

    GamepadInterface gamepad_interface;
    gamepad_interface.Init();
    gamepad_interface.SetSmooth(0.02);
    gamepad_interface.print_flag = true;  // true or false

    while (true) {
        gamepad_interface.Update();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return 0;
}