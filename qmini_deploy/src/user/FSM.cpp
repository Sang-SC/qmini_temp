#include "user/FSM.h"
#include <iostream>

FSM::FSM(ControlFSMData *data)
    :_data(data)
{
    _stateList.invalid = nullptr;
    _stateList.passive = new FSMState_Passive(_data);
    _stateList.standUp = new FSMState_StandUp(_data);
    _stateList.walk = new FSMState_Walk(_data);

    initialize();
}

FSM::~FSM(){
    _stateList.deletePtr();
}

void FSM::initialize()
{
    _currentState = _stateList.passive;
    _currentState -> enter();
    _nextState = _currentState;
    _mode = FSMMode::NORMAL;
}

void FSM::set_passive()
{
    _currentState = _stateList.passive;
    _currentState -> enter();
    _nextState = _currentState;
    _mode = FSMMode::NORMAL;
}

void FSM::run()
{
    if(!checkSafety())
    {
        std::cout << "Safety check failed!" << std::endl;
        set_passive();
        return;
    }

    if(_mode == FSMMode::NORMAL)
    {
        _currentState->run();
        _nextStateName = _currentState->checkTransition();
        if(_nextStateName != _currentState->_stateName)
        {
            _mode = FSMMode::CHANGE;
            _nextState = getNextState(_nextStateName);
        }
    }
    else if(_mode == FSMMode::CHANGE)
    {
        _currentState->exit();
        _currentState = _nextState;
        _currentState->enter();
        _mode = FSMMode::NORMAL;
        _currentState->run();       
    }
}

FSMState* FSM::getNextState(FSMStateName stateName)
{
    switch(stateName)
    {
        case FSMStateName::INVALID:
            return _stateList.invalid;
        break;
        case FSMStateName::PASSIVE:
            return _stateList.passive;
        break;
        case FSMStateName::STANDUP:
            return _stateList.standUp;
        break;
        case FSMStateName::WALK:
            return _stateList.walk;
        break;
        default:
            return _stateList.invalid;
        break;
    }
}

bool FSM::checkSafety()
{
    // Not implemented yet
    return true;
}