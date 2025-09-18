#ifndef FSMSTATE_PASSIVE_H
#define FSMSTATE_PASSIVE_H

#include "FSMState.h"

class FSMState_Passive: public FSMState
{
    public:
        FSMState_Passive(ControlFSMData *data);
        void enter();
        void run();
        void exit();
        FSMStateName checkTransition();
};

#endif // FSMSTATE_PASSIVE_H