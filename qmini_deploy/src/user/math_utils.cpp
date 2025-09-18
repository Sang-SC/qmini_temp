#include "user/math_utils.h"

// Constants
double rad2deg = 180.0 / M_PI;
double deg2rad = M_PI / 180.0;

Mat3<float> Euler2Rot(const Vec3<float>& euler, Order order) {
    Mat3<float> R;
    float c1 = cos(euler(0));
    float s1 = sin(euler(0));
    float c2 = cos(euler(1));
    float s2 = sin(euler(1));
    float c3 = cos(euler(2));
    float s3 = sin(euler(2));

    switch (order) {
        case Order::ZYX:
            R << c1 * c2, c1 * s2 * s3 - c3 * s1, s1 * s3 + c1 * c3 * s2,
                 c2 * s1, c1 * c3 + s1 * s2 * s3, c3 * s1 * s2 - c1 * s3,
                -s2, c2 * s3, c2 * c3;
            break;
        default:
            throw std::runtime_error("Unsupported Euler angle order");
    }

    return R;
}

// The order of the returned quaternions is (x, y, z, w)
Vec4<float> Euler2Quat(const Vec3<float>& euler, Order order) {
    Eigen::Quaternion<float> quat;
    quat = Eigen::Quaternion<float>(Euler2Rot(euler, order));
    Vec4<float> quat_vec;
    quat_vec << quat.x(), quat.y(), quat.z(), quat.w();
    return quat_vec;
}

Vec3<float> Rot2Euler(const Mat3<float>& R, Order order) {
    Vec3<float> euler;
    switch (order) {
        case Order::ZYX:
            euler(0) = atan2(R(1, 0), R(0, 0));
            euler(1) = atan2(-R(2, 0), sqrt(R(2, 1) * R(2, 1) + R(2, 2) * R(2, 2)));
            euler(2) = atan2(R(2, 1), R(2, 2));
            break;
        default:
            throw std::runtime_error("Unsupported Euler angle order");
    }
    return euler;
}

Vec4<float> Rot2Quat(const Mat3<float>& R) {
    Eigen::Quaternion<float> quat(R);
    Vec4<float> quat_vec;
    quat_vec << quat.x(), quat.y(), quat.z(), quat.w();
    return quat_vec;
}

// The order of the quat is (x, y, z, w)
Vec3<float> Quat2Euler(const Vec4<float>& quat, Order order) {
    Eigen::Quaternion<float> quat_eigen(quat(3), quat(0), quat(1), quat(2));
    return Rot2Euler(quat_eigen.toRotationMatrix(), order);
}

Mat3<float> Quat2Rot(const Vec4<float>& quat) {
    Eigen::Quaternion<float> quat_eigen(quat(3), quat(0), quat(1), quat(2));
    return quat_eigen.toRotationMatrix();
}