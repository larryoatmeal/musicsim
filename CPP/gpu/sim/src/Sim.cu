#include "Sim.h"

#include <iostream>
#include <algorithm>
#include <helper_functions.h>
#include <helper_cuda.h>
#include "Kernel.cuh"
#include <stdlib.h>
#include <vector>
#include "sim_constants.h"

dim3              dimBlock;
dim3              dimGrid;
int MAX_AUDIO_SIZE = 44100 * 10;
void SimStateGPU::step(){
  gpu_step();
};

float SimStateGPU::read_pressure(){
  return 0;
}
SimStateGPU::SimStateGPU(float *sigma, int * aux_data, int argc, char *argv[]){
  init(sigma, aux_data, argc, argv);
  iter = 0;
}
SimStateGPU::~SimStateGPU(){
  checkCudaErrors(cudaFree(bufferP_in));
  checkCudaErrors(cudaFree(bufferVx_in));
  checkCudaErrors(cudaFree(bufferVy_in));
  checkCudaErrors(cudaFree(bufferP_out));
  checkCudaErrors(cudaFree(bufferVx_out));
  checkCudaErrors(cudaFree(bufferVy_out));
  checkCudaErrors(cudaFree(bufferAux_in));
};
int SimStateGPU::GetWidth(){
  return WIDTH;
};
int SimStateGPU::GetHeight(){
  return HEIGHT;
};

void SimStateGPU::gpu_step(){
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
    buffersSigma_in,
    bufferAudio,
    iter
  );
  // check for error
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }
  //for debug 
  // cudaDeviceSynchronize();
  std::swap<float *>(bufferP_in, bufferP_out);
  std::swap<float *>(bufferVx_in, bufferVx_out);
  std::swap<float *>(bufferVy_in, bufferVy_out);
  iter += 1;
}

std::vector<float> SimStateGPU::read_back(){
    float * output_from_gpu = (float *) calloc(MAX_AUDIO_SIZE, sizeof(float));
    // Wait for the kernel to complete
    checkCudaErrors(cudaDeviceSynchronize());
    // Read the result back
    checkCudaErrors(cudaMemcpy(output_from_gpu, bufferAudio, sizeof(float) * MAX_AUDIO_SIZE, cudaMemcpyDeviceToHost));

    std::cout << "Samples: " << iter << std::endl;
    std::vector<float> v(iter);

    for(int i = 0; i < iter; i++){
      v[i] = output_from_gpu[i];
    }
    
    return v;

}

void SimStateGPU::clear(){
  iter = 0;
}

int getIndexGlobalPrivate(int x, int y){
  return x + (y + 2) * STRIDE_Y;
}

void SimStateGPU::setAux(int x, int y, int val){
  int i = getIndexGlobalPrivate(x, y);
  checkCudaErrors(cudaMemcpy(&bufferAux_in[i], &val, sizeof(val), cudaMemcpyHostToDevice));
};



void SimStateGPU::init(float *sigma, int * aux_data, int argc, char *argv[]){
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


  checkCudaErrors(cudaMalloc((void **)&bufferAudio, MAX_AUDIO_SIZE * sizeof(float)));
  checkCudaErrors(cudaMemcpy(bufferAudio, (float *) calloc(MAX_AUDIO_SIZE, sizeof(float)), MAX_AUDIO_SIZE * sizeof(float), cudaMemcpyHostToDevice));

  
  dimBlock.x = 16;
  dimBlock.y = 16;
  dimGrid.x  = WIDTH/dimBlock.x;
  dimGrid.y  = HEIGHT/dimBlock.y; //for now assume this is perfect division
  printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
  printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);
  
  // int _p_bore_index = (41 + PAD_HALF) + (53 + PAD_HALF) * (STRIDE_Y);
  // int _num_excite = 4;
  // int _listen_index  = (155 + PAD_HALF) + (45 + PAD_HALF) * (STRIDE_Y);

  // checkCudaErrors(cudaMemcpyToSymbol(listen_index, &_listen_index, sizeof(listen_index)));
  // checkCudaErrors(cudaMemcpyToSymbol(num_excite, &_num_excite, sizeof(num_excite)));
  // checkCudaErrors(cudaMemcpyToSymbol(p_bore_index, &_p_bore_index, sizeof(p_bore_index)));
};


