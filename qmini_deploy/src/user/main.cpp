#include "user/Qmini.h"
#include <csignal>

void signal_handler(int signal) {
    if (signal == SIGINT) {
        exit(0);
    }
}

int main() {
    std::signal(SIGINT, signal_handler);
    
    Qmini qmini;
    qmini.Run();
    return 0;
}