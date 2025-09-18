#ifndef FSMSTATE_H
#define FSMSTATE_H

#include <string>
#include <iostream>
#include "ControlFSMData.h"

enum class FSMStateName{
    INVALID,
    PASSIVE,
    STANDUP,
    WALK,
};

class FSMState
{
    public:
        FSMState(ControlFSMData *data, FSMStateName stateName, std::string stateNameStr);

        virtual void enter() = 0;
        virtual void run() = 0;
        virtual void exit() = 0;
        virtual FSMStateName checkTransition() {return FSMStateName::INVALID;}

        FSMStateName _stateName;
        std::string _stateNameStr;

    protected:
        ControlFSMData *_data;
        FSMStateName _nextStateName;
};

#endif // FSMSTATE_H