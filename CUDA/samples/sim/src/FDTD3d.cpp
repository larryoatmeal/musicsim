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

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif


typedef float number;

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


int index_of_padded(int w, int h){
  return (w + PAD_HALF) + (h + PAD_HALF) * (STRIDE_Y);
}

number* alloc_grid(){
  return (number *)calloc(N_TOTAL, sizeof(float));
}

void init(number *walls, number *excitor, number *beta, number *sigma, int *aux_cells){
    //walls

    for(int i = 40; i < 150; i++){
      walls[index_of_padded(i, 50)] = 1; //
      walls[index_of_padded(i, 55)] = 1;
    }

    walls[index_of_padded(55, 130)] = 0;
    walls[index_of_padded(55, 100)] = 0;

    //excitor
    for(int i = 51; i < 55; i++){
      excitor[index_of_padded(40, i)] = 1;
    }

    //beta
    for(int i = 0; i < N_TOTAL; i++){
      beta[i] = 1 - (walls[i] + excitor[i]);
    }

    //sigma
    for(int i = 0; i < PML_LAYERS + 1; i++){
      for(int x = i; x < W - i; x++){
        for(int y = i; y < H - i; y++){
          sigma[index_of_padded(x, y)] = 0.5 / DT * (PML_LAYERS - i)/PML_LAYERS;
        }
      }
    }

    for(int i = 0; i < N_TOTAL; i++){
      aux_cells[i] = ( (int) walls[i] << 2) + ( (int) excitor[i] << 1) + (int) beta[i];
    }


}


bool runTest(int argc, const char **argv)
{
    number *walls = alloc_grid();
    number *excitor = alloc_grid();
    number *beta = alloc_grid();
    number *sigma = alloc_grid();
    int *aux_cells = (int *)calloc(N_TOTAL, sizeof(int));

    init(walls, excitor, beta, sigma, aux_cells);


    number *p = alloc_grid();
    number *v_x = alloc_grid();
    number *v_y = alloc_grid();

    number *p_prev = alloc_grid();
    number *v_x_prev = alloc_grid();
    number *v_y_prev = alloc_grid();



    //todo fix

    free(beta);
    free(sigma);
    free(p);
    free(v_x);
    free(v_y);
    free(p_prev);
    free(v_x_prev);
    free(v_y_prev);
    // device_output = (float *)calloc(volumeSize, sizeof(float));
    //
    // // Execute on the device
    printf("fdtdGPU...\n");
    fdtdGPUMine(sigma, aux_cells, 100, argc, argv);
    // printf("fdtdGPU complete\n");

    return 0;
}
