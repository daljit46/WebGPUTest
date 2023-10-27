#include "application.h"
#include <iostream>
#include <vector>
#include <exception>

int main()
{
    try {
        Application app;
        app.onCompute();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
