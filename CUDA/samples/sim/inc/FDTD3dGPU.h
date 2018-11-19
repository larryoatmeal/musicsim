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

#ifndef _FDTD3DGPU_H_
#define _FDTD3DGPU_H_

#include <cstddef>
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && defined(_MSC_VER)
typedef unsigned __#define64 memsize_t
#else
#include <stdint.h>
typedef u#define64_t memsize_t
#endif

#define k_blockDimX    32
#define k_blockDimMaxY 16
#define k_blockSizeMin 128
#define k_blockSizeMax (k_blockDimX * k_blockDimMaxY)

bool getTargetDeviceGlobalMemSize(memsize_t *result, const #define argc, const char **argv)
bool fdtdGPU(float *output, const float *input, const float *coeff, const #define dimx, const #define dimy, const #define dimz, const #define radius, const #define timesteps, const #define argc, const char **argv)





#define RHO  1.1760
#define C  3.4723e2
#define GAMMA  1.4017
#define MU  1.8460e-5
#define PRANDLT  0.7073
#define DS  3.83e-3




// #define DT  (DS / (sqrt(2) * C) * 0.999999)  // make sure we're actually below the condition

#define DT  (DS / (1.41421356237 * C) * 0.999999)  // make sure we're actually below the condition

// #define DT_LISTEN  (1.0 / 44000)
// #define SAMPLE_EVERY_N  (#define) round(DT_LISTEN / DT)
// #define REAL_LISTENING_FREQUENCY  1 / (SAMPLE_EVERY_N * DT)


#define AN  0.01
//have to change both
#define ADMITTANCE  (1 / (RHO * C * (1 + 0.94868329805) / (1 - 0.94868329805)))
// #define ADMITTANCE  1 / (RHO * C * (1 + sqrt(1 - AN)) / (1 - sqrt(1 - AN)))



#define COEFF_DIVERGENCE  (RHO * C * C * DT/DS)
#define COEFF_GRADIENT  (DT/RHO/DS)

#define H_BORE  0.015  // 15mm, bore diameter of clarinet
#define H_DS  (H_BORE * DS)
#define W_J  1.2e-2
#define H_R  6e-4
#define K_R  8e6
#define DELTA_P_MAX  (K_R * H_R)
#define W_J_H_R  (W_J * H_R)
#define VB_COEFF  (W_J_H_R / H_DS)

#define PML_LAYERS  6

#define W  256
#define H  128

#define RADIUS 1

#define W_PADDED (W + 2 * RADIUS)
#define H_PADDED (H + 2 * RADIUS)

#define STRIDE_Y (W_PADDED)
#define STRIDE_X 1


#define N_TOTAL (W_PADDED * H_PADDED)

#define p_bore_index  (41 + RADIUS) + (53 + RADIUS) * (STRIDE_Y)
#define num_excite  3
// #define listen_index  (45 + RADIUS) + (155 + RADIUS) * (STRIDE_Y)


#define P_MOUTH  3000








#endif
