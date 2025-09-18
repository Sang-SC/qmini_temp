#include "user/FSMState.h"

FSMState::FSMState(ControlFSMData *data, FSMStateName stateName, std::string stateNameStr):
            _data(data), _stateName(stateName), _stateNameStr(stateNameStr)
{
}