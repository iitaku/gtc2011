#include <iostream>
#include <stdio.h>
#include <string.h>

#include "display.hpp"

using namespace gtc;

int main(int argc, char* argv[])
{
    int width = 512;
    int height = 512;
             
    if (1<argc)
    {
        for (char** ptr = &argv[1]; NULL != *ptr; ++ptr)
        {
            if (0 == strcmp(*ptr, "-geometry"))
            {
                sscanf(*(ptr+1), "%dx%d", &width, &height);
                break;
            }
        }
    }
 
    if (0 != (width % 16) || 0 != (height % 16))
    {
        std::cerr << "window size must be multiples of 16" << std::endl;
        return -1;
    }
      
    Display<DrawImage> display(argc, argv, width, height);
    display.loop();

    display.finish();

    return 0;
}
