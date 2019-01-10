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

#include "FDTD3dGPU.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Reference.h"
#include "FDTD3dGPUKernel.cuh"
#include <stdlib.h>

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int               deviceCount  = 0;
    int               targetDevice = 0;
    size_t            memsize      = 0;

    // Get the number of CUDA enabled GPU devices
    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    // Query target device for maximum memory allocation
    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    // Save the result
    *result = (memsize_t)memsize;
    return true;
}



bool fdtdGPUMine(const int timesteps, const int argc, const char **argv)
{
    number *walls = 0;
    number *excitor = 0;
    number *beta = 0;
    number *sigma = 0;
    int *aux_data = 0;
    number *p = 0;
    number *v_x = 0;
    number *v_y = 0;
    number *p_prev = 0;
    number *v_x_prev = 0;
    number *v_y_prev = 0; 

    ReferenceSim::genInitialState(
      &walls,
      &excitor,
      &beta,
      &sigma,
      &aux_data,
      &p,
      &v_x,
      &v_y,
      &p_prev,
      &v_x_prev,
      &v_y_prev
    );



  int               deviceCount  = 0;
  int               targetDevice = 0;

  float            *bufferP_in      = 0;
  float            *bufferVx_in     = 0;
  float            *bufferVy_in     = 0;

  float            *bufferP_out      = 0;
  float            *bufferVx_out     = 0;
  float            *bufferVy_out     = 0;

  float            *buffersSigma_in     = 0;

  int             *bufferAux_in     = 0;



  float * empty = (float *) calloc(N_TOTAL, sizeof(float));

  dim3              dimBlock;
  dim3              dimGrid;

  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, (const char **)argv);

  checkCudaErrors(cudaSetDevice(targetDevice));
  // Allocate memory buffers

  int size = N_TOTAL * sizeof(float);

  checkCudaErrors(cudaMalloc((void **)&bufferP_in, size));
  checkCudaErrors(cudaMalloc((void **)&bufferVx_in, size));
  checkCudaErrors(cudaMalloc((void **)&bufferVy_in, size));
  checkCudaErrors(cudaMalloc((void **)&bufferP_out,  size));
  checkCudaErrors(cudaMalloc((void **)&bufferVx_out, size));
  checkCudaErrors(cudaMalloc((void **)&bufferVy_out, size));
  checkCudaErrors(cudaMalloc((void **)&buffersSigma_in, size));
  checkCudaErrors(cudaMalloc((void **)&bufferAux_in, N_TOTAL * sizeof(int)));

  checkCudaErrors(cudaMemcpy(bufferP_in, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferVx_in, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferVy_in, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferP_out, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferVx_out, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferVy_out, empty, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(buffersSigma_in, sigma, size, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(bufferAux_in, aux_data, N_TOTAL * sizeof(int), cudaMemcpyHostToDevice));

  dimBlock.x = 16;
  dimBlock.y = 16;
  dimGrid.x  = W/dimBlock.x;
  dimGrid.y  = H/dimBlock.y; //for now assume this is perfect division
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);


  float * output_from_gpu = (float *) calloc(N_TOTAL, sizeof(float));


  for (int it = 0 ; it < timesteps ; it++)
  {
      if(it % 1000 == 0){
        printf("\tt = %d ", it);
      }

      // Launch the kernel
      // printf("launch kernel\n");
      // FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);
      AudioKernel<<<dimGrid, dimBlock>>>(
        bufferVx_in,
        bufferVy_in,
        bufferP_in,
        bufferVx_out,
        bufferVy_out,
        bufferP_out,
        bufferAux_in,
        buffersSigma_in
      );
      // check for error
        cudaError_t err = cudaGetLastError();
        if(err != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

      std::swap<float *>(bufferP_in, bufferP_out);
      std::swap<float *>(bufferVx_in, bufferVx_out);
      std::swap<float *>(bufferVy_in, bufferVy_out);

      //REFERENCE IMPLEMENTATION
      ReferenceSim::Reference(
        v_x_prev,
        v_y_prev,
        p_prev,
        v_x,
        v_y,
        p,
        aux_data,
        sigma
      );
      std::swap<float *>(p_prev, p);
      std::swap<float *>(v_x_prev, v_x);
      std::swap<float *>(v_y_prev, v_y);


      // Wait for the kernel to complete
      checkCudaErrors(cudaDeviceSynchronize());
      // Read the result back, result is in bufferP_in (after final toggle)
      checkCudaErrors(cudaMemcpy(output_from_gpu, bufferP_in, size, cudaMemcpyDeviceToHost));

      float error = 0;

      for(int i = 0; i < N_TOTAL; i++){
        // printf("\tGPU = %f ", output_from_gpu[i]);
        // printf("\tRef = %f ", p_prev[i]);

        error += abs(output_from_gpu[i] - p_prev[i]);
      }
      
      printf("\tError = %f ", error);



      //compare this with the reference which is in p_prev at this point

  }




  


  //free
  checkCudaErrors(cudaFree(bufferP_in));
  checkCudaErrors(cudaFree(bufferVx_in));
  checkCudaErrors(cudaFree(bufferVy_in));
  checkCudaErrors(cudaFree(bufferP_out));
  checkCudaErrors(cudaFree(bufferVx_out));
  checkCudaErrors(cudaFree(bufferVy_out));
  checkCudaErrors(cudaFree(bufferAux_in));


    free(beta);
    free(sigma);
    free(p);
    free(v_x);
    free(v_y);
    free(p_prev);
    free(v_x_prev);
    free(v_y_prev);


  return true;
}
