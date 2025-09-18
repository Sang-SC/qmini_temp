#ifndef FSMSTATE_STANDUP_H
#define FSMSTATE_STANDUP_H

#include "FSMState.h"

class FSMState_StandUp: public FSMState
{
    public:
        FSMState_StandUp(ControlFSMData *data);
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();

    private:
        int num_joints = 10;
        vector<float> joint_enter_q;
        vector<float> joint_standup_q;
        vector<float> joint_kp;
        vector<float> joint_kd;
        uint64_t rate_count;
};

#endif // FSMSTATE_STANDUP_H