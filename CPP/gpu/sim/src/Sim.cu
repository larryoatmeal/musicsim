#include "Sim.h"


#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Kernel.cuh"
#include <stdlib.h>

dim3              dimBlock;
dim3              dimGrid;

void SimState::step(){

};

float SimState::read_pressure(){
  
}
SimState::SimState(float *sigma, int * aux_data, int argc, char *argv[]){
  
}
SimState::~SimState(){
  checkCudaErrors(cudaFree(bufferP_in));
  checkCudaErrors(cudaFree(bufferVx_in));
  checkCudaErrors(cudaFree(bufferVy_in));
  checkCudaErrors(cudaFree(bufferP_out));
  checkCudaErrors(cudaFree(bufferVx_out));
  checkCudaErrors(cudaFree(bufferVy_out));
  checkCudaErrors(cudaFree(bufferAux_in));
};
int SimState::GetWidth(){

};
int SimState::GetHeight(){

};
float SimState::GetPressure(int x, int y){

};


void SimState::gpu_step(){
  // Launch the kernel
  // printf("launch kernel\n");
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

}

void SimState::read_back(){
    // float * output_from_gpu = (float *) calloc(N_TOTAL, sizeof(float));
    // // Wait for the kernel to complete
    // checkCudaErrors(cudaDeviceSynchronize());
    // // Read the result back, result is in bufferP_in (after final toggle)
    // checkCudaErrors(cudaMemcpy(output_from_gpu, bufferP_in, size, cudaMemcpyDeviceToHost));
}

void SimState::init(float *sigma, int * aux_data, int argc, char *argv[]){
  float * empty = (float *) calloc(N_TOTAL, sizeof(float));

  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(argc, 0);

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
};


