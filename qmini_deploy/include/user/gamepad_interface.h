#ifndef GAMEPAD_INTERFACE_H
#define GAMEPAD_INTERFACE_H

#include <SDL2/SDL.h>
#include <mutex>
#include <cmath>
#include <cstdlib>  // for exit()
#include <iostream>

class Button {
public:
    Button() {}

    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class GamepadData {
public:
    float lx = 0.;
    float rx = 0.;
    float ry = 0.;
    float ly = 0.;

    Button R1;
    Button L1;
    Button start;
    Button select;
    Button R2;
    Button L2;
    Button F1;
    Button F2;
    Button A;
    Button B;
    Button X;
    Button Y;
    Button up;
    Button right;
    Button down;
    Button left;
};

class GamepadInterface {
public:
    GamepadInterface() {
        joystick = nullptr;
    }

    ~GamepadInterface() {
        if (joystick) {
            SDL_JoystickClose(joystick);
        }
        SDL_Quit();
    }

    int Init() {
        // Initialize SDL2 and enable the joystick subsystem
        if (SDL_Init(SDL_INIT_JOYSTICK) < 0) {
            std::cerr << "SDL initialization failed: " << SDL_GetError() << std::endl;
            exit(-1);
        }

        // Check if there are any joysticks connected
        if (SDL_NumJoysticks() <= 0) {
            std::cerr << "No joystick detected!" << std::endl;
            exit(-1);
        }

        // Open the first joystick
        joystick = SDL_JoystickOpen(0);
        if (!joystick) {
            std::cerr << "Failed to open joystick: " << SDL_GetError() << std::endl;
            exit(-1);
        }

        return 0;
    }

    void Update() {
        // Update the joystick state
        SDL_JoystickUpdate();

        {
            // Lock the mutex to safely update gamepad data
            std::lock_guard<std::mutex> lock(gamepad_mutex);

            // Update stick values with smooth and deadzone
            float lx_raw = SDL_JoystickGetAxis(joystick, 0) / 32768.0f;
            float ly_raw = SDL_JoystickGetAxis(joystick, 1) / 32768.0f;
            float rx_raw = SDL_JoystickGetAxis(joystick, 2) / 32768.0f;
            float ry_raw = SDL_JoystickGetAxis(joystick, 3) / 32768.0f;
            gamepad_data.lx = gamepad_data.lx * (1 - smooth) + (std::fabs(lx_raw) < dead_zone ? 0.0 : lx_raw) * smooth;
            gamepad_data.ly = gamepad_data.ly * (1 - smooth) + (std::fabs(ly_raw) < dead_zone ? 0.0 : ly_raw) * smooth;
            gamepad_data.rx = gamepad_data.rx * (1 - smooth) + (std::fabs(rx_raw) < dead_zone ? 0.0 : rx_raw) * smooth;
            gamepad_data.ry = gamepad_data.ry * (1 - smooth) + (std::fabs(ry_raw) < dead_zone ? 0.0 : ry_raw) * smooth;

            // Update button status (except up, right, down, and left)
            gamepad_data.A.update(SDL_JoystickGetButton(joystick, 0));
            gamepad_data.B.update(SDL_JoystickGetButton(joystick, 1));
            gamepad_data.X.update(SDL_JoystickGetButton(joystick, 3));
            gamepad_data.Y.update(SDL_JoystickGetButton(joystick, 4));
            gamepad_data.L1.update(SDL_JoystickGetButton(joystick, 6));
            gamepad_data.R1.update(SDL_JoystickGetButton(joystick, 7));
            gamepad_data.L2.update(SDL_JoystickGetButton(joystick, 8));
            gamepad_data.R2.update(SDL_JoystickGetButton(joystick, 9));
            gamepad_data.select.update(SDL_JoystickGetButton(joystick, 10));
            gamepad_data.start.update(SDL_JoystickGetButton(joystick, 11));

            // Print the current state of the gamepad (if needed)
            if (print_flag) {
                std::cout << "lx: " << gamepad_data.lx << ", ly: " << gamepad_data.ly
                        << ", rx: " << gamepad_data.rx << ", ry: " << gamepad_data.ry << std::endl;
                std::cout << "A: " << (gamepad_data.A.pressed ? "Pressed" : "Released")
                            << ", B: " << (gamepad_data.B.pressed ? "Pressed" : "Released")
                            << ", X: " << (gamepad_data.X.pressed ? "Pressed" : "Released")
                            << ", Y: " << (gamepad_data.Y.pressed ? "Pressed" : "Released") << std::endl;
                std::cout << "L1: " << (gamepad_data.L1.pressed ? "Pressed" : "Released")
                        << ", R1: " << (gamepad_data.R1.pressed ? "Pressed" : "Released")
                        << ", L2: " << (gamepad_data.L2.pressed ? "Pressed" : "Released")
                        << ", R2: " << (gamepad_data.R2.pressed ? "Pressed" : "Released") << std::endl;
                std::cout << "Select: " << (gamepad_data.select.pressed ? "Pressed" : "Released")
                        << ", Start: " << (gamepad_data.start.pressed ? "Pressed" : "Released") << std::endl;
                // std::cout << "Up: " << (gamepad_data.up.pressed ? "Pressed" : "Released")
                //         << ", Right: " << (gamepad_data.right.pressed ? "Pressed" : "Released")
                //         << ", Down: " << (gamepad_data.down.pressed ? "Pressed" : "Released")
                //         << ", Left: " << (gamepad_data.left.pressed ? "Pressed" : "Released") << std::endl;
                std::cout << std::endl;
            }
        }
    }

    GamepadData GetGamepadData() {
        // Lock the mutex to safely access gamepad data
        std::lock_guard<std::mutex> lock(gamepad_mutex);
        return gamepad_data;
    }

    void SetSmooth(float smooth_new) {
        if (smooth_new >= 0.0 && smooth_new <= 1.0) {
            smooth = smooth_new;
        } else {
            std::cerr << "Invalid smooth value: " << smooth_new << ". Must be between 0 and 1." << std::endl;
        }
    }

    void SetDeadZone(float dead_zone_new) {
        if (dead_zone_new >= 0.0 && dead_zone_new <= 1.0) {
            dead_zone = dead_zone_new;
        } else {
            std::cerr << "Invalid dead zone value: " << dead_zone_new << ". Must be between 0 and 1." << std::endl;
        }
    }

    bool print_flag = false;

private:
    GamepadData gamepad_data;
    std::mutex gamepad_mutex;
    float smooth = 0.03;
    float dead_zone = 0.01;
    SDL_Joystick* joystick;
};

#endif // GAMEPAD_INTERFACE_H