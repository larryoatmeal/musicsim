/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "FDTD3d.h"

#include <iostream>
#include <iomanip>

#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

#include <helper_functions.h>

#include <math.h>
#include <assert.h>
#include "Reference.h"
#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif



//// Name of the log file
//const char *printfFile = "FDTD3d.txt";

// Forward declarations
bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, char **argv)
{
    bool bTestResult = false;
    // Start the log
    printf("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
        bTestResult = true;
    }
    else
    {
        // Execute
        bTestResult = runTest(argc, (const char **)argv);
    }

    // Finish
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}

void showHelp(const int argc, const char **argv)
{
    std::cout << std::endl << "Syntax:" << std::endl;
    
}

bool runTest(int argc, const char **argv)
{
    

    //todo fix


    // device_output = (float *)calloc(volumeSize, sizeof(float));
    //
    // // Execute on the device
    printf("fdtdGPU...\n");
    fdtdGPUMine(128000, argc, argv);
    // printf("fdtdGPU complete\n");

    
    return 0;
}
