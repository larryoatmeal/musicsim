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


 #include <stdio.h>

 #include "Kernel3d.cuh"
 #include "constants.h"
 #include "Kernel3dGPU.h"
 #include <cooperative_groups.h>
 #include <helper_functions.h>
 #include <helper_cuda.h>
 namespace cg = cooperative_groups;
 

 // Note: If you change the RADIUS, you should also change the unrolling below

//  __constant__ float stencil[RADIUS + 1];
 
void writeDT(float val){
    checkCudaErrors(cudaMemcpyToSymbol(DT, &val, sizeof(float)));
}
void writeDS(float val){
    checkCudaErrors(cudaMemcpyToSymbol(DS, &val, sizeof(float)));
}
void writeZN(float val){
    checkCudaErrors(cudaMemcpyToSymbol(ZN, &val, sizeof(float)));
}
void writeNumExcitors(int val){
    checkCudaErrors(cudaMemcpyToSymbol(numExcitor, &val, sizeof(int)));
}
void writeExcitorMode(int val){
    checkCudaErrors(cudaMemcpyToSymbol(excitorMode, &val, sizeof(int)));
}

__device__ int getBitsCUDA(int val, int shift, int mask){
    return (val >> shift) & mask;
}

__device__ float getSigmaCUDA(int aux){
    float PML_SCALE = (0.5 / DT / PML);
    int sigmaN = getBitsCUDA(aux, SIGMA_SHIFT, THREE_BIT);
    return PML_SCALE * (sigmaN);
}
__device__ int getBetaCUDA(int aux){
    return getBitsCUDA(aux, BETA_SHIFT, ONE_BIT);
}

__device__ int getExcitorCUDA(int aux){
    return getBitsCUDA(aux, EXCITE_SHIFT, ONE_BIT);
}

__device__ float pressureStep(
    float p,
    float v_x,
    float v_x_left,
    float v_y,
    float v_y_up,
    float v_z,
    float v_z_behind,
    int beta,
    float sigma  
)
  {
    float COEFF_DIVERGENCE = RHO * Cs * Cs * DT/DS;

    float divergence = v_x + v_y + v_z - v_x_left - v_y_up - v_z_behind;
    float p_denom = 1 + (1 - beta + sigma) * DT;
    return (p - COEFF_DIVERGENCE * divergence)/p_denom;
  }


// __device__ float pressureStep2(
//     int i,
//     float4 *input,
//     int *aux
//     int stride_x,
//     int stride_y
// ){
//     float p = input[i].x;
//     float divergence = 
//         input[i].y - input[i - 1].y 
//         + input[i].z - input[i - stride_y].z 
//         + input[i].w - input[i - stride_z].w;
//     float sigma = getSigmaCUDA(aux[i]);
//     float beta = getBetaCUDA(aux[i]);
//     float sigma_prime = 1  - beta + sigma;
//     float p_next = (p - RHO * Cs * Cs * DT  * divergence)/(1 + sigma_prime * DT);

//     return p_next;
// }

// __device__ float vStep2(
//     int i,
//     float4 *input,
//     int *aux
//     int stride
// ){
//     float v = input[i].y;

//     float beta = min(getBetaCUDA(aux[i]), getBetaCUDA(aux[i + stride]));
//     float grad_x = 
// }




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
)
 {
    const float COEFF_GRADIENT = (DT/RHO/DS);
    const float ADMITTANCE = (1.0/ZN);

    
    // printf("DT %f\n", DT);
    // printf("DS %f\n", DS);
    // printf("ZN %f\n", ZN);

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
     __shared__ float4 tile[k_blockDimMaxY + 2 * RADIUS][k_blockDimX + 2 * RADIUS];

     const int stride_y = dimx + 2 * RADIUS;
     const int stride_z = stride_y * (dimy + 2 * RADIUS);
 
     int inputIndex  = 0;
     int outputIndex = 0;
 
     // Advance inputIndex to start of inner volume
     inputIndex += RADIUS * stride_y + RADIUS;
 
     // Advance inputIndex to target element
     inputIndex += gtidy * stride_y + gtidx;
 
     float4 infront[RADIUS];
     float4 behind[RADIUS];
     float4 current;
 
     const int tx = ltidx + RADIUS;
     const int ty = ltidy + RADIUS;
 
     // Check in bounds
 
     //can't read from here
     if ((gtidx >= dimx + RADIUS) || (gtidy >= dimy + RADIUS))
         validr = false;
 
     //can't read or write here 
     if ((gtidx >= dimx) || (gtidy >= dimy))
         validw = false;
 
     //outside dim + radius: can't read or write
     //outside dim but within: can't write, but can read!
 
     // Preload the "infront" and "behind" data
 
     //what's this for?
 
     //preload the behind at the beginning
     //advance z layer to where it actually starts
     
 
     //this loop runs RADIUS - 1 times
     //cuz we later do another += stride_z to make it RADIUS total
     for (int i = RADIUS - 2 ; i >= 0 ; i--)
     {
         if (validr) //if this is actually a valid read
             behind[i] = input[inputIndex];
 
         inputIndex += stride_z; //
     }
 
     if (validr)
         current = input[inputIndex];
 
    //Note at this point, current is at input index =
    //RADIUS * (STRIDE_X = 1) + RADIUS * strideY + (RADIUS - 1) * strideZ
    //It's at RADIUS - 1 for strideZ because the for loop starts off with an advance
    //so the first iteration will start at the right location!

     outputIndex = inputIndex;
     inputIndex += stride_z;
 
     //by here input index should be inputIndex += RADIUS * stride_z
 
     //now read the things that are inputIndex += stride_z ahead
     for (int i = 0 ; i < RADIUS ; i++)
     {
         if (validr)
             infront[i] = input[inputIndex];
 
         inputIndex += stride_z;
     }
 
     //inputIndex is now offset by 2 * RADIUS * stride_z
     //inputIndex represents the furthests most position of the infront 
 

     // Step through the xy-planes
 #pragma unroll 9
 
     for (int iz = 0 ; iz < dimz ; iz++)
     {
         // Advance the slice (move the thread-front)
         for (int i = RADIUS - 1 ; i > 0 ; i--)
             behind[i] = behind[i - 1];
 
         behind[0] = current;
         current = infront[0];
 #pragma unroll 2
 
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
 
    

         int aux = inputAux[outputIndex];
         int auxRight = inputAux[outputIndex + 1];
         int auxDown =  inputAux[outputIndex + stride_y];
         int auxInfront = inputAux[outputIndex + stride_z];

        
         float4 value = current;

         float4 valueLeft = tile[ty][tx - 1];
         float4 valueRight = tile[ty][tx + 1];
         float4 valueUp = tile[ty - 1][tx];
         float4 valueDown = tile[ty + 1][tx];
         float4 valueInfront = infront[0];
         float4 valueBehind = behind[0];

         float4 valueRightUp = tile[ty - 1][tx + 1];
         float4 valueRightBehind = input[outputIndex + 1 - stride_z]; //fix: expensive global memory load

         float4 valueDownLeft = tile[ty + 1][tx - 1];
         float4 valueDownBehind = input[outputIndex + stride_y - stride_z]; //fix: expensive global memory load
         



         float4 valueInfrontLeft = input[outputIndex - 1 + stride_z]; //fix
         float4 valueInfrontUp = input[outputIndex - stride_y + stride_z]; //fix

        float newPressure = pressureStep(
            value.x, 
            value.y, valueLeft.y, //don't be confused, v_x is y oops
            value.z, valueUp.z,
            value.w, valueBehind.w,
            getBetaCUDA(aux),
            getSigmaCUDA(aux)
        );

        float newPressureRight = pressureStep(
            valueRight.x, 
            valueRight.y, value.y, //don't be confused, v_x is y oops
            valueRight.z, valueRightUp.z,
            valueRight.w, valueRightBehind.w,
            getBetaCUDA(auxRight),
            getSigmaCUDA(auxRight)
        );

        float newPressureDown = pressureStep(
            valueDown.x, 
            valueDown.y, valueDownLeft.y, //don't be confused, v_x is y oops
            valueDown.z, value.z,
            valueDown.w, valueDownBehind.w,
            getBetaCUDA(auxDown),
            getSigmaCUDA(auxDown)
        );

        float newPressureInfront = pressureStep(
            valueInfront.x, 
            valueInfront.y, valueInfrontLeft.y, //don't be confused, v_x is y oops
            valueInfront.z, valueInfrontUp.z,
            valueInfront.w, value.w,
            getBetaCUDA(auxInfront),
            getSigmaCUDA(auxInfront)
        );
        


        int isExcitor = getExcitorCUDA(aux);
        int isBeta = getBetaCUDA(aux);

        float vb_x = 0;
        float vb_y = 0;
        float vb_z = 0;
        if(isExcitor == 1){
            // const int ubore_filter_position = 10 * 44100 - 1 -  4; //super hacky use audio buffer as global
            if(excitorMode == 0){
                float delta_p = max(p_mouth - input[i_global_p_bore].x, 0.0f);
                float u_bore =  W_J * H_R * max((1 - delta_p / DELTA_P_MAX), 0.0) * sqrt(2 * delta_p / RHO) ;
                float excitation = u_bore / (DS * DS * numExcitor); //hashtag units!  
                
                float in = excitation;


                //use audio buffer as extra stroage
                int zIndex = 13 * 44100 * 5;

                float z1 = 0;
                float z2 = 0;
                float out = 0;
                if(iter % 2 == 0){
                    z1 = audioBuffer[zIndex];
                    z2 = audioBuffer[zIndex + 1];
                    out = in * a0 + z1;
                    audioBuffer[zIndex + 2] = in * a1 + z2 - b1 * out;
                    audioBuffer[zIndex + 3]    = in * a2 - b2 * out;
                }

                else{
                    z1 = audioBuffer[zIndex + 2];
                    z2 = audioBuffer[zIndex + 3];
                    out = in * a0 + z1;
                    audioBuffer[zIndex] = in * a1 + z2 - b1 * out;
                    audioBuffer[zIndex + 1]    = in * a2 - b2 * out;
                }

                //filtered output
                vb_z = out;

                vb_y = 0;
                vb_x = 0;
            }else if(excitorMode == 1){
                vb_z = sin(iter * DT * 2 * 3.14159265359 * 440);
                vb_y = 0;
                vb_x = 0;
            }else{

            }
        }

        float sigma = getSigmaCUDA(aux);

        int beta_x = min(isBeta, getBetaCUDA(auxRight)); //if beta_vx_dir = -1 or 1, then beta_vx_dir = 0 as expected
        float grad_x = newPressureRight - newPressure;
        float sigma_prime_dt_x = (1 - beta_x + sigma) * DT;
        float current_vx = value.y;
        // float new_vx = beta_x * ( current_vx  - COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x)/(beta_x + sigma_prime_dt_x);

        float new_vx = (beta_x * current_vx - beta_x * COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x)/(beta_x + sigma_prime_dt_x);

        int beta_y = min(isBeta, getBetaCUDA(auxDown)); //if beta_vx_dir = -1 or 1, then beta_vx_dir = 0 as expected
        float grad_y = newPressureDown - newPressure;
        float sigma_prime_dt_y = (1 - beta_y + sigma) * DT;
        float current_vy = value.z;

        float new_vy = (beta_y * current_vy - beta_y * COEFF_GRADIENT * grad_y + sigma_prime_dt_y * vb_y)/(beta_y + sigma_prime_dt_y);


        int beta_z = min(isBeta, getBetaCUDA(auxInfront)); //if beta_vx_dir = -1 or 1, then beta_vx_dir = 0 as expected
        float grad_z = newPressureInfront - newPressure;
        float sigma_prime_dt_z = (1 - beta_z + sigma) * DT;
        float current_vz = value.w;

        float new_vz = (beta_z * current_vz  - beta_z * COEFF_GRADIENT * grad_z + sigma_prime_dt_z * vb_z)/(beta_z + sigma_prime_dt_z);

        value.x = newPressure;
        value.y = new_vx;
        value.z = new_vy;
        value.w = new_vz;


        if(validw)
            output[outputIndex] = value;

        if(outputIndex == i_global_listener){
            audioBuffer[iter] = newPressure;
        }
     }
 }
 