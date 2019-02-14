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

 #ifndef _KERNEL_H_
 #define _KERNEL_H_


#include "FDTD3dGPU.h"

__global__ void AudioKernel(
  float *v_x_prev,
  float *v_y_prev,
  float *p_prev,
  float *v_x,
  float *v_y,
  float *p,
  int *aux,
  float *sigma,
  float *audioBuffer,
  int iter
);
// __constant__ int listen_index;
// __constant__ int num_excite;
// __constant__ int p_bore_index;

#endif