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
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            // print the CUDA error message and exit
            printf("CUDA error: %s\n", cudaGetErrorString(error));
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
        printf("\tGPU = %f ", output_from_gpu[i]);
        printf("\tRef = %f ", p_prev[i]);

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





bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps, const int argc, const char **argv)
{
    const int         outerDimx  = dimx + 2 * radius;
    const int         outerDimy  = dimy + 2 * radius;
    const int         outerDimz  = dimz + 2 * radius;
    const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
    int               deviceCount  = 0;
    int               targetDevice = 0;
    float            *bufferOut    = 0;
    float            *bufferIn     = 0;
    dim3              dimBlock;
    dim3              dimGrid;

    // Ensure that the inner data starts on a 128B boundary
    const int padding = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    cudaEvent_t profileStart = 0;
    cudaEvent_t profileEnd   = 0;
    const int profileTimesteps = timesteps - 1;

    if (profileTimesteps < 1)
    {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), profiling is disabled.\n", timesteps);
    }

#endif

    // Check the radius is valid
    if (radius != RADIUS)
    {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    // Get the number of CUDA enabled GPU devices
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Select target device (device 0 by default)
    targetDevice = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaSetDevice(targetDevice));

    // Allocate memory buffers
    checkCudaErrors(cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

    // Check for a command-line specified block size
    int userBlockSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "block-size"))
    {
        userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
        // Constrain to a multiple of k_blockDimX
        userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

        // Constrain within allowed bounds
        userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
    }
    else
    {
        userBlockSize = k_blockSizeMax;
    }

    // Check the device limit on the number of threads
    struct cudaFuncAttributes funcAttrib;
    checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

    userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    // Set the block size
    dimBlock.x = k_blockDimX;
    // Visual Studio 2005 does not like std::min
    //    dimBlock.y = std::min<size_t>(userBlockSize / k_blockDimX, (size_t)k_blockDimMaxY);
    dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX) : (size_t)k_blockDimMaxY;
    dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    // Check the block size is valid
    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS)
    {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }

    // Copy the input to the device input buffer
    checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the input to the device output buffer (actually only need the halo)
    checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    // Copy the coefficients to the device coefficient buffer
    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));


#ifdef GPU_PROFILING

    // Create the events
    checkCudaErrors(cudaEventCreate(&profileStart));
    checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

    // Execute the FDTD
    float *bufferSrc = bufferIn + padding;
    float *bufferDst = bufferOut + padding;
    printf(" GPU FDTD loop\n");


#ifdef GPU_PROFILING
    // Enqueue start event
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

    for (int it = 0 ; it < timesteps ; it++)
    {
        printf("\tt = %d ", it);

        // Launch the kernel
        printf("launch kernel\n");
        FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);

        // Toggle the buffers
        // Visual Studio 2005 does not like std::swap
        //    std::swap<float *>(bufferSrc, bufferDst);
        float *tmp = bufferDst;
        bufferDst = bufferSrc;
        bufferSrc = tmp;
    }

    printf("\n");

#ifdef GPU_PROFILING
    // Enqueue end event
    checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());

    // Read the result back, result is in bufferSrc (after final toggle)
    checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Report time
#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0)
    {
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0)
    {
        // Convert milliseconds to seconds
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        // Determine number of computations per timestep
        size_t pointsComputed = dimx * dimy * dimz;
        // Determine throughput
        double throughputM    = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, NumDevsUsed = %u, Blocksize = %u\n",
               throughputM, avgElapsedTime, pointsComputed, 1, dimBlock.x * dimBlock.y);
    }

#endif

    // Cleanup
    if (bufferIn)
    {
        checkCudaErrors(cudaFree(bufferIn));
    }

    if (bufferOut)
    {
        checkCudaErrors(cudaFree(bufferOut));
    }

#ifdef GPU_PROFILING

    if (profileStart)
    {
        checkCudaErrors(cudaEventDestroy(profileStart));
    }

    if (profileEnd)
    {
        checkCudaErrors(cudaEventDestroy(profileEnd));
    }

#endif
    return true;
}
