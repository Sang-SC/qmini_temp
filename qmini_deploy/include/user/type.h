#ifndef TYPE_H
#define TYPE_H

#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <eigen3/Eigen/Dense>

using std::cin, std::cout, std::endl;
using std::string;
using std::vector;
namespace chrono = std::chrono;

template <typename T>
using Vec2 = typename Eigen::Matrix<T, 2, 1>;

template <typename T>
using Vec3 = typename Eigen::Matrix<T, 3, 1>;

template <typename T>
using Vec4 = typename Eigen::Matrix<T, 4, 1>;

template <typename T>
using Mat2 = typename Eigen::Matrix<T, 2, 2>;

template <typename T>
using Mat3 = typename Eigen::Matrix<T, 3, 3>;

template <typename T>
using DVec = typename Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using DMat = typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

#endif // TYPE_H