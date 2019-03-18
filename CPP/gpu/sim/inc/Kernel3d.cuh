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

#ifndef _KERNEL3D_H_
#define _KERNEL3D_H_

#include "vector_types.h"
#include "Kernel3dGPU.h"

__global__ void AudioKernel3D(
  float4 *input,
  float4 *output,
  int *inputAux,
  float *audioBuffer,
  const int dimx, 
  const int dimy, 
  const int dimz,
  int iter,
  int i_global_listener,
  int i_global_p_bore,
  float p_mouth
);
__constant__ float DT;
__constant__ float DS;
__constant__ float ZN;




#endif