#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include "user/type.h"
#include <cmath>

extern double rad2deg;
extern double deg2rad;

enum class Order { ZYX };
Mat3<float> Euler2Rot(const Vec3<float>& euler, Order order);
Vec4<float> Euler2Quat(const Vec3<float>& euler, Order order);
Vec4<float> Rot2Quat(const Mat3<float>& R);
Vec3<float> Rot2Euler(const Mat3<float>& R, Order order);
Vec3<float> Quat2Euler(const Vec4<float>& quat, Order order);
Mat3<float> Quat2Rot(const Vec4<float>& quat);

#endif // MATH_UTILS_H