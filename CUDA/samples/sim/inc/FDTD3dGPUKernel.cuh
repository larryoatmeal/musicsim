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
#include <cooperative_groups.h>

namespace cg = cooperative_groups;


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
  float *sigma
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
}


// Note: If you change the RADIUS, you should also change the unrolling below
#define RADIUS 4

__constant__ float stencil[RADIUS + 1];













__global__ void FiniteDifferencesKernel(float *output,
                                        const float *input,
                                        const int dimx,
                                        const int dimy,
                                        const int dimz)
{
    bool validr = true;
    bool validw = true;
    const int gtidx = blockIdx.x * blockDim.x + threadIdx.x;
    const int gtidy = blockIdx.y * blockDim.y + threadIdx.y;
    const int ltidx = threadIdx.x;
    const int ltidy = threadIdx.y;
    const int workx = blockDim.x;
    const int worky = blockDim.y;
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

    const int stride_y = dimx + 2 * RADIUS;
    const int stride_z = stride_y * (dimy + 2 * RADIUS);

    int inputIndex  = 0;
    int outputIndex = 0;

    // Advance inputIndex to start of inner volume
    inputIndex += RADIUS * stride_y + RADIUS;

    // Advance inputIndex to target element
    inputIndex += gtidy * stride_y + gtidx;

    float infront[RADIUS];
    float behind[RADIUS];
    float current;

    const int tx = ltidx + RADIUS;
    const int ty = ltidy + RADIUS;

    // Check in bounds
    if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
        validr = false;

    if ((gtidx >= dimx) || (gtidy >= dimy))
        validw = false;

    // Preload the "infront" and "behind" data
    for (int i = RADIUS - 2 ; i >= 0 ; i--)
    {
        if (validr)
            behind[i] = input[inputIndex];

        inputIndex += stride_z;
    }

    if (validr)
        current = input[inputIndex];

    outputIndex = inputIndex;
    inputIndex += stride_z;

    for (int i = 0 ; i < RADIUS ; i++)
    {
        if (validr)
            infront[i] = input[inputIndex];

        inputIndex += stride_z;
    }

    // Step through the xy-planes
#pragma unroll 9

    for (int iz = 0 ; iz < dimz ; iz++)
    {
        // Advance the slice (move the thread-front)
        for (int i = RADIUS - 1 ; i > 0 ; i--)
            behind[i] = behind[i - 1];

        behind[0] = current;
        current = infront[0];
#pragma unroll 4

        for (int i = 0 ; i < RADIUS - 1 ; i++)
            infront[i] = infront[i + 1];

        if (validr)
            infront[RADIUS - 1] = input[inputIndex];

        inputIndex  += stride_z;
        outputIndex += stride_z;
        cg::sync(cta);

        // Note that for the work items on the boundary of the problem, the
        // supplied index when reading the halo (below) may wrap to the
        // previous/next row or even the previous/next xy-plane. This is
        // acceptable since a) we disable the output write for these work
        // items and b) there is at least one xy-plane before/after the
        // current plane, so the access will be within bounds.

        // Update the data slice in the local tile
        // Halo above & below
        if (ltidy < RADIUS)
        {
            tile[ltidy][tx]                  = input[outputIndex - RADIUS * stride_y];
            tile[ltidy + worky + RADIUS][tx] = input[outputIndex + worky * stride_y];
        }

        // Halo left & right
        if (ltidx < RADIUS)
        {
            tile[ty][ltidx]                  = input[outputIndex - RADIUS];
            tile[ty][ltidx + workx + RADIUS] = input[outputIndex + workx];
        }

        tile[ty][tx] = current;
        cg::sync(cta);

        // Compute the output value
        float value = stencil[0] * current;
#pragma unroll 4

        for (int i = 1 ; i <= RADIUS ; i++)
        {
            value += stencil[i] * (infront[i-1] + behind[i-1] + tile[ty - i][tx] + tile[ty + i][tx] + tile[ty][tx - i] + tile[ty][tx + i]);
        }

        // Store the output value
        if (validw)
            output[outputIndex] = value;
    }
}
