cmake_minimum_required(VERSION 3.5)
project(qmini_deploy)

## Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## Check system architecture
message(STATUS "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}")

## Import thirdparty libraries
add_subdirectory(thirdparty)

## Threads
find_package(Threads REQUIRED)

## Unitree Motor SDK, Unitree SDK2 and ONNX Runtime
add_library(unitree_motor_sdk STATIC IMPORTED GLOBAL)
set(unitree_motor_sdk_lib_dirs "${CMAKE_CURRENT_LIST_DIR}/lib/unitreeMotor")
add_library(unitree_sdk2 STATIC IMPORTED GLOBAL)
set(unitree_sdk2_lib_dirs "${CMAKE_CURRENT_LIST_DIR}/lib/unitree_sdk2")
add_library(onnxruntime STATIC IMPORTED GLOBAL)
set(onnxruntime_lib_dirs "${CMAKE_CURRENT_LIST_DIR}/lib/onnxruntime")
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
    set_target_properties(unitree_motor_sdk PROPERTIES IMPORTED_LOCATION "${unitree_motor_sdk_lib_dirs}/libUnitreeMotorSDK_Arm64.so")
    set_target_properties(unitree_sdk2 PROPERTIES IMPORTED_LOCATION "${unitree_sdk2_lib_dirs}/aarch64/libunitree_sdk2.a")
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${onnxruntime_lib_dirs}/aarch64/libonnxruntime.so")
else()
    set_target_properties(unitree_motor_sdk PROPERTIES IMPORTED_LOCATION "${unitree_motor_sdk_lib_dirs}/libUnitreeMotorSDK_Linux64.so")
    set_target_properties(unitree_sdk2 PROPERTIES IMPORTED_LOCATION "${unitree_sdk2_lib_dirs}/x86_64/libunitree_sdk2.a")
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${onnxruntime_lib_dirs}/x86_64/libonnxruntime.so")
endif()
target_link_libraries(unitree_sdk2 INTERFACE ddsc ddscxx Threads::Threads)

## Serial port library
find_library(LIBSERIAL_LIBRARIES NAMES serial)
if(LIBSERIAL_LIBRARIES)
    message(STATUS "LibSerial library found: ${LIBSERIAL_LIBRARIES}")
else()
    message(FATAL_ERROR "LibSerial library not found")
endif()

## Ensure that the dynamic library can be found at runtime
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${unitree_motor_sdk_lib_dirs};${onnxruntime_lib_dirs}")

## Include directories
include_directories(include)

## Test
add_executable(test_yaml_cpp src/test/test_yaml_cpp.cpp)
target_link_libraries(test_yaml_cpp yaml-cpp)

add_executable(test_gamepad_interface src/test/test_gamepad_interface.cpp)
target_link_libraries(test_gamepad_interface SDL2)

add_executable(test_imu_interface src/test/test_imu_interface.cpp
                                  src/user/math_utils.cpp)
target_link_libraries(test_imu_interface ${LIBSERIAL_LIBRARIES})

add_executable(test_goM8010_6_motor src/test/test_goM8010_6_motor.cpp)
target_link_libraries(test_goM8010_6_motor unitree_motor_sdk)

## User
add_library(user_FSM STATIC
    src/user/math_utils.cpp
    src/user/FSM.cpp
    src/user/FSMState.cpp
    src/user/FSMState_Walk.cpp
    src/user/FSMState_StandUp.cpp
    src/user/FSMState_Passive.cpp
)
target_link_libraries(user_FSM yaml-cpp SDL2 ${LIBSERIAL_LIBRARIES} unitree_motor_sdk unitree_sdk2 onnxruntime)

add_executable(main src/user/main.cpp)
target_link_libraries(main user_FSM)