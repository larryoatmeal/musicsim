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
#include "sim_constants.h"
#include <stdio.h>

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
  float divergence = v_x_prev[i] - v_x_prev[i - STRIDE_X_SHARED] + v_y_prev[i] - v_y_prev[i - STRIDE_Y_SHARED];
  float p_denom = 1 + (1 - getBeta(aux, i) + sigma[i]) * DT;
  return (p_prev[i] - COEFF_DIVERGENCE * divergence)/p_denom;
}

__device__ int getIndexGlobal(int x, int y){
  return x + (y + 2) * STRIDE_Y;
}
__device__ int getIndexLocal(int thread_x, int thread_y){
  return (thread_x + 1) + (thread_y + 1) * 18;
}

template <class T>
__device__ T loadChunk(
  T *shared,
  T *global,
  int thread_x,
  int thread_y,
  int offset
)
{ 
  int local_index = thread_x + thread_y * 16 + offset;

  int local_x = local_index % 18;
  int local_y = local_index / 18;
  int global_x = local_x + blockIdx.x*blockDim.x - 1;
  int global_y = local_y + blockIdx.y*blockDim.y - 1;
  shared[local_index] = global[getIndexGlobal(global_x, global_y)];
  return local_index;
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

  int i_global = getIndexGlobal(idx, idy);

  // //load shared memory ----------------------------------------------------------------
  const int SHARED_N = (16 + 2) * (16 + 2);
  __shared__ float v_x_prev_shared[SHARED_N]; //padded memory
  __shared__ float v_y_prev_shared[SHARED_N];
  __shared__ float p_prev_shared  [SHARED_N];
  __shared__ int aux_shared  [SHARED_N];
  __shared__ float sigma_shared  [SHARED_N];
  
  //load first part
  loadChunk(v_x_prev_shared, v_x_prev, threadIdx.x, threadIdx.y, 0);
  loadChunk(v_y_prev_shared, v_y_prev, threadIdx.x, threadIdx.y, 0);
  loadChunk(p_prev_shared, p_prev, threadIdx.x, threadIdx.y, 0);
  loadChunk(aux_shared, aux, threadIdx.x, threadIdx.y, 0);
  loadChunk(sigma_shared, sigma, threadIdx.x, threadIdx.y, 0);
  
  //load second
  int offset = 256;
  if(threadIdx.x + threadIdx.y * 16 + offset < SHARED_N){
    loadChunk(v_x_prev_shared, v_x_prev, threadIdx.x, threadIdx.y, offset);
    loadChunk(v_y_prev_shared, v_y_prev, threadIdx.x, threadIdx.y, offset);
    loadChunk(p_prev_shared, p_prev, threadIdx.x, threadIdx.y, offset);
    loadChunk(aux_shared, aux, threadIdx.x, threadIdx.y, offset);
    loadChunk(sigma_shared, sigma, threadIdx.x, threadIdx.y, offset);
  }
  
  __syncthreads();

  int i = getIndexLocal(threadIdx.x, threadIdx.y);
  //PRESSURE------------------------------
  

  float p_current = pressureStep(v_x_prev_shared, v_y_prev_shared, p_prev_shared, aux_shared, sigma_shared, i);
  float p_right = pressureStep(v_x_prev_shared, v_y_prev_shared, p_prev_shared, aux_shared, sigma_shared, i + STRIDE_X_SHARED);
  float p_down = pressureStep(v_x_prev_shared, v_y_prev_shared, p_prev_shared, aux_shared, sigma_shared, i + STRIDE_Y_SHARED);
  
  p[i_global] = p_current;


  // //VB------------------------------
  // //TODO: not sure if this is supposed to be previous or next pressure
  float delta_p = max(P_MOUTH - p_prev[p_bore_index], 0.0f);
  float vb_x = 0;
  float vb_y = 0;
  int wall = getWall(aux_shared, i);
  int excitor = getExcitor(aux_shared, i);
  int wall_down = getWall(aux_shared, i + STRIDE_Y_SHARED);
  vb_x = excitor * (1 - delta_p / DELTA_P_MAX) * sqrt(2 * delta_p / RHO) * VB_COEFF / num_excite;
  vb_y = wall * ADMITTANCE * -1 * p_down + wall_down * ADMITTANCE * p_current;
  
  // //VELOCITY------------------------------
  int beta_current = getBeta(aux_shared, i);

  float beta_x = min(beta_current, getBeta(aux_shared, i + STRIDE_X_SHARED));
  float grad_x = p_right - p_current;
  float sigma_prime_dt_x = (1 - beta_x + sigma_shared[i]) * DT;
  v_x[i_global] = (beta_x * v_x_prev_shared[i] - beta_x * beta_x * COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x)/(beta_x + sigma_prime_dt_x);


  
  float beta_y = min(beta_current, getBeta(aux_shared, i + STRIDE_Y_SHARED));
  float grad_y = p_down - p_current;
  float sigma_prime_dt_y = (1 - beta_y + sigma_shared[i]) * DT;

  v_y[i_global] = (beta_y * v_y_prev_shared[i] - beta_y * beta_y * COEFF_GRADIENT * grad_y + sigma_prime_dt_y * vb_y)/(beta_y + sigma_prime_dt_y);

  if(i_global == listen_index){
    audioBuffer[iter] = p_current;
  }

}

#endif