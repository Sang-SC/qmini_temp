#ifndef FSMSTATE_WALK_H
#define FSMSTATE_WALK_H

#include "FSMState.h"
#include "user/onnx_inference.h"

class FSMState_Walk: public FSMState
{
    public:
        FSMState_Walk(ControlFSMData *data);
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();

        void update_leg_phase();
        float dt = 0.02;
        float phase = 0.0;
        Vec2<float> leg_phases = Vec2<float>::Zero();
        Vec4<float> clock_input = Vec4<float>::Zero();
        int dynamic_stand_flag = 0;
        int static_stand_phase_ready_flag = 0;

    private:
        uint64_t phase_counter;

        // Observation
        Vec3<float> ang_vel;
        Vec3<float> gravity_orientation;
        Vec3<float> cmd;
        DVec<float> qj;
        DVec<float> dqj;
        DVec<float> default_angles;
        DVec<float> action;
        DVec<float> obs_current;
        DMat<float> obs_history;
        DVec<float> obs_input;
        int obs_init_flag = 0;

        // Observation and action dimensions
        int num_joints = 10;
        int num_actions = num_joints;
        int history_len = 4;
        int num_obs_per_step = 43;
        int num_obs = num_obs_per_step * (history_len + 1);
        int num_estimator_output = 3 + 2;
        int num_actor_observations = num_obs + num_estimator_output;

        // Observation and action scales
        float obs_scales_lin_vel = 2.0;
        float obs_scales_ang_vel = 0.25;
        float obs_scales_dof_pos = 1.0;
        float obs_scales_dof_vel = 0.05;
        Vec3<float> command_scale;
        float action_scale = 0.25;

        // Onnxruntime
        std::unique_ptr<Ort::Env> env_actor;
        std::unique_ptr<Ort::SessionOptions> session_options_actor;
        std::unique_ptr<Ort::Session> session_actor;
        std::unique_ptr<OnnxInference> onnx_inference_actor;
        std::unique_ptr<Ort::Env> env_estimator;
        std::unique_ptr<Ort::SessionOptions> session_options_estimator;
        std::unique_ptr<Ort::Session> session_estimator;
        std::unique_ptr<OnnxInference> onnx_inference_estimator;

        // Time
        std::chrono::steady_clock::time_point run_time_prev;

        // Others
        vector<int> low_vel_flags;
        vector<float> low_vel_thresholds;
        vector<float> lin_vel_x_range;
        vector<float> lin_vel_y_range;
        vector<float> ang_vel_yaw_range;
        vector<float> joint_kp;
        vector<float> joint_kd;
        vector<float> gait_offsets;
        float gait_period;
};

#endif // FSMSTATE_WALK_H