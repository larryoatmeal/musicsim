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

 #ifndef _FDTD3DGPUKernel_H_
 #define _FDTD3DGPUKernel_H_
#include "Kernel.cuh"
#include "FDTD3dGPU.h"
#include <helper_cuda.h>

// #include <cooperative_groups.h>

// namespace cg = cooperative_groups;


__device__ int getBeta(
int *aux,
int i
){
  return min(aux[i] & (1 << 0), 1);
}

__device__ int getExcitor(
int *aux,
int i
){
  return min(aux[i] & (1 << 1), 1);
}

__device__ int getWall(
int *aux,
int i
){
  return min(aux[i] & (1 << 2), 1);
}

__device__ float pressureStep(
  float *v_x_prev,
  float *v_y_prev,
  float *p_prev,
  int *aux,
  float *sigma,
  int i
)
{
  float divergence = v_x_prev[i] - v_x_prev[i - STRIDE_X] + v_y_prev[i] - v_y_prev[i - STRIDE_Y];
  float p_denom = 1 + (1 - getBeta(aux, i) + sigma[i]) * DT;
  return (p_prev[i] - COEFF_DIVERGENCE * divergence)/p_denom;
}


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
)
{
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  int idy=blockIdx.y*blockDim.y+threadIdx.y;

  int i = (idx + PAD_HALF) + STRIDE_Y * (idy + PAD_HALF);

  //PRESSURE------------------------------

  float p_current = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i);
  float p_right = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i + STRIDE_X);
  float p_down = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i + STRIDE_Y);
  p[i] = p_current;

  //VB------------------------------
  //TODO: not sure if this is supposed to be previous or next pressure
  float delta_p = max(P_MOUTH - p_prev[p_bore_index], 0.0f);
  float vb_x = 0;
  float vb_y = 0;
  int wall = getWall(aux, i);
  int excitor = getExcitor(aux, i);
  int wall_down = getWall(aux, i + STRIDE_Y);
  vb_x = excitor * (1 - delta_p / DELTA_P_MAX) * sqrt(2 * delta_p / RHO) * VB_COEFF / num_excite;
  vb_y = wall * ADMITTANCE * p_current + wall_down * -ADMITTANCE * p_down;

  //VELOCITY------------------------------
  int beta_current = getBeta(aux, i);

  float beta_x = min(beta_current, getBeta(aux, i + STRIDE_X));
  float grad_x = p_right - p_current;
  float sigma_prime_dt_x = (1 - beta_x + sigma[i]) * DT;
  v_x[i] = beta_x * v_x_prev[i] - beta_x * beta_x * COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x;

  float beta_y = min(beta_current, getBeta(aux, i + STRIDE_Y));
  float grad_y = p_down - p[i];
  float sigma_prime_dt_y = (1 - beta_y + sigma[i]) * DT;
  v_y[i] = beta_y * v_y_prev[i] - beta_y * beta_y * COEFF_GRADIENT * grad_y + sigma_prime_dt_y * vb_y;

  if(i == listen_index){
    audioBuffer[iter] = p_current;
  }

  
}

#endif