#ifndef FSM_H
#define FSM_H

#include "FSMState.h"
#include "FSMState_Passive.h"
#include "FSMState_StandUp.h"
#include "FSMState_Walk.h"

struct FSMStateList{
    FSMState *invalid;
    FSMState_Passive *passive;
    FSMState_StandUp *standUp;
    FSMState_Walk *walk;
   
    void deletePtr(){
        delete invalid;
        delete passive;
        delete standUp;
        delete walk;
    }  
};

enum class FSMMode{
    NORMAL,
    CHANGE
};

class FSM{
    public:
        FSM(ControlFSMData *data);
        ~FSM();
        void initialize();
        void run();
        void set_passive();

    private:
        FSMState* getNextState(FSMStateName stateName);
        bool checkSafety();
        ControlFSMData *_data;
        FSMState *_currentState;
        FSMState *_nextState;
        FSMStateName _nextStateName;
        FSMStateList _stateList;
        FSMMode _mode;
};

#endif // FSM_H