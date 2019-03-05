#include "Sim3d.h"
#include "Kernel3d.cuh"
#include <helper_functions.h>
#include <helper_cuda.h>
#include "constants.h"
#include <algorithm> 
const int MAX_AUDIO_SIZE = 44100 * 10;

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

Sim3D::Sim3D(int dimx, int dimy, int dimz){
  m_dimx = dimx;
  m_dimy = dimy;
  m_dimz = dimz;
  m_scheduledWalls = std::vector< std::vector<int> >();
};


void Sim3D::init(){
  int dimx = m_dimx;
  int dimy = m_dimy;
  int dimz = m_dimz;
  
  const int         outerDimx  = dimx + 2 * RADIUS;
  const int         outerDimy  = dimy + 2 * RADIUS;
  const int         outerDimz  = dimz + 2 * RADIUS;
  const size_t      volumeSize = outerDimx * outerDimy * outerDimz;
  int               deviceCount  = 0;
  int               targetDevice = 0;
  float4            *bufferOut    = 0;
  float4            *bufferIn     = 0;
  int               *bufferInAux  = 0;
  float             *audioBuffer  = 0;
  dim3              dimBlock;
  dim3              dimGrid;



  // size_t volumeSize;
  // memsize_t memsize;

  // const float lowerBound = 0.0f;
  // const float upperBound = 1.0f;

  // Determine default dimensions
  printf("Set-up, based upon target device GMEM size...\n");
  // Get the memory size of the target device
  printf(" getTargetDeviceGlobalMemSize\n");
  getTargetDeviceGlobalMemSize(&memsize, argc, argv);

  // We can never use all the memory so to keep things simple we aim to
  // use around half the total memory
  // memsize /= 2;

  // Most of our memory use is taken up by the input and output buffers -
  // two buffers of equal size - and for simplicity the volume is a cube:
  //   dim = floor( (N/2)^(1/3) )
  // defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

  // By default, make the volume edge size an integer multiple of 128B to
  // improve performance by coalescing memory accesses, in a real
  // application it would make sense to pad the lines accordingly
  // int roundTarget = 128 / sizeof(float);
  // defaultDim = defaultDim / roundTarget * roundTarget;
  // defaultDim -= k_radius_default * 2;

  // // Check dimension is valid
  // if (defaultDim < k_dim_min)
  // {
  //     printf("insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
  //     exit(EXIT_FAILURE);
  // }
  // else if (defaultDim > k_dim_max)
  // {
  //     defaultDim = k_dim_max;
  // }

  // // For QA testing, override default volume size
  // if (checkCmdLineFlag(argc, argv, "qatest"))
  // {
  //     defaultDim = MIN(defaultDim, k_dim_qa);
  // }


  // Ensure that the inner data starts on a 128B boundary
  // Note volumesize already includes padding from the radius!
  const int padding = (128 / sizeof(float4)) - RADIUS;
  const size_t paddedVolumeSize = volumeSize + padding;


  // Get the number of CUDA enabled GPU devices
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));

  // Select target device (device 0 by default)
  targetDevice = findCudaDevice(0, 0);

  checkCudaErrors(cudaSetDevice(targetDevice));

  // Allocate memory buffers
  checkCudaErrors(cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float4)));
  checkCudaErrors(cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float4)));
  checkCudaErrors(cudaMalloc((void **)&bufferInAux, paddedVolumeSize * sizeof(int)));

  checkCudaErrors(cudaMalloc((void **)&audioBuffer, MAX_AUDIO_SIZE * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&audioBuffer, MAX_AUDIO_SIZE * sizeof(float)));

  int userBlockSize = k_blockSizeMax;

  // Check the device limit on the number of threads
  struct cudaFuncAttributes funcAttrib;
  checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, AudioKernel3D));

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

  // // Copy the input to the device input buffer
  // checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float4), cudaMemcpyHostToDevice));
  // // Copy the input to the device output buffer (actually only need the halo)
  // checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float4), cudaMemcpyHostToDevice));
  // //set audio buffer to zero
  // checkCudaErrors(cudaMemset(audioBuffer, 0, MAX_AUDIO_SIZE * sizeof(float)));


  // Execute the FDTD
  float4 *bufferSrc = bufferIn + padding;
  float4 *bufferDst = bufferOut + padding;
  int *bufferAux    = bufferInAux + padding;
  printf(" GPU FDTD loop\n");

  m_bufferSrc = bufferSrc;
  m_bufferDst = bufferDst;
  m_bufferIn = bufferIn;
  m_bufferOut = bufferOut;
  m_audioBuffer = audioBuffer;
  m_i = 0;
  m_dimGrid = dimGrid;
  m_dimBlock = dimBlock;
  m_volumeSize = volumeSize;

  m_bufferAux = bufferAux;
  m_bufferInAux = bufferInAux;
  reset();
};


void Sim3D::clean(){
    // Cleanup
    if (m_bufferIn)
    {
        checkCudaErrors(cudaFree(m_bufferIn));
    }

    if (m_bufferOut)
    {
        checkCudaErrors(cudaFree(m_bufferOut));
    }

    if (m_bufferInAux)
    {
        checkCudaErrors(cudaFree(m_bufferInAux));
    }

    if (m_audioBuffer)
    {
        checkCudaErrors(cudaFree(m_audioBuffer));
    }
}

void Sim3D::step(int n){
  for(int i = 0; i < n; i++){
    AudioKernel3D<<<m_dimGrid, m_dimBlock>>>(m_bufferDst, m_bufferSrc, m_bufferAux, m_audioBuffer, m_dimx, m_dimy, m_dimz, m_i,
    m_i_global_listener, m_i_global_p_bore, m_p_mouth);
    // Toggle the buffers
    // Visual Studio 2005 does not like std::swap
    //    std::swap<float *>(bufferSrc, bufferDst);
    float4 *tmp = m_bufferDst;
    m_bufferDst = m_bufferSrc;
    m_bufferSrc = tmp;
    m_i += 1;
  }
}




std::vector<float> Sim3D::readBackAudio(){
  float * output_from_gpu = (float *) calloc(MAX_AUDIO_SIZE, sizeof(float));
  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());
  // Read the result back
  checkCudaErrors(cudaMemcpy(output_from_gpu, m_audioBuffer, sizeof(float) * MAX_AUDIO_SIZE, cudaMemcpyDeviceToHost));
  std::cout << "Samples: " << m_i << std::endl;
  std::vector<float> v(m_i);
  for(int i = 0; i < m_i; i++){
    v[i] = output_from_gpu[i];
  }

  free(output_from_gpu);
  
  return v;
};

std::vector< std::vector<float> > Sim3D::readBackData(){
  float4 * output_from_gpu = (float4 *) calloc(m_volumeSize, sizeof(float4));
  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());
  // Read the result back
  checkCudaErrors(cudaMemcpy(output_from_gpu, m_bufferSrc, sizeof(float4) * m_volumeSize, cudaMemcpyDeviceToHost));
  std::cout << "Size: " << m_volumeSize << std::endl;
  std::vector< std::vector<float> > v(m_volumeSize);
  for(int i = 0; i < m_volumeSize; i++){
    std::vector<float> vec;
    float4 output = output_from_gpu[i];
    vec.push_back(output.x);
    vec.push_back(output.y);
    vec.push_back(output.z);
    vec.push_back(output.w);
    v[i] = vec;
  }

  free(output_from_gpu);
  return v;
};



std::vector<int> Sim3D::readBackAux(){
  int * output_from_gpu = (int *) calloc(m_volumeSize, sizeof(int));
  // Wait for the kernel to complete
  checkCudaErrors(cudaDeviceSynchronize());
  // Read the result back
  checkCudaErrors(cudaMemcpy(output_from_gpu, m_bufferAux, sizeof(int) * m_volumeSize, cudaMemcpyDeviceToHost));
  std::cout << "Size: " << m_volumeSize << std::endl;
  std::vector<int> v(m_volumeSize);
  for(int i = 0; i < m_volumeSize; i++){
    v[i] = output_from_gpu[i];
  }

  free(output_from_gpu);
  return v;
};



int Sim3D::getStrideY(){
  return m_dimx + 2 * RADIUS;
}

int Sim3D::getStrideZ(){
  return getStrideY() * (m_dimy +  2 * RADIUS);
}

int Sim3D::getGlobalIndex(int x, int y, int z){
  int startOfInnerVolume = 
    RADIUS 
  + RADIUS * getStrideY()  
  + RADIUS * getStrideZ();

  
  int target = startOfInnerVolume + 
    x 
  + y * getStrideY() 
  + z * getStrideZ();
  return target;
}

//first MSBs represent sigma value
//LSB represents beta



// int setBit(int p, int n){
//   (*p) |= 1UL << n;
// }

// int clearBit(int p, int n){
//   (*p) &= ~(1UL << n);
// }

int setValues(int original, int valueToInsert, int mask, int shift){

  int bit_mask = mask << shift;
  return (original & (~bit_mask)) | (valueToInsert << shift);
}

int getBit(int val, int shift){
  return (val >> shift) & 1;
}

int getBits(int val, int shift, int mask){
  return (val >> shift) & mask;
}

int getBetaValue(int val){
  return getBit(val, BETA_SHIFT);
}

void Sim3D::setSigma(int x, int y, int z, int level){
  int i = getGlobalIndex(x, y, z);

  int aux;
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&aux, &m_bufferAux[i], sizeof(int), cudaMemcpyDeviceToHost));

  int newAux = setValues(aux, level, THREE_BIT, SIGMA_SHIFT);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(&m_bufferAux[i], &newAux, sizeof(int), cudaMemcpyHostToDevice));
}
int* Sim3D::getAux(){
  int *aux = (int *) calloc(m_volumeSize, sizeof(int));

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(aux, m_bufferAux, m_volumeSize * sizeof(int), cudaMemcpyDeviceToHost));
  return aux;
}

void Sim3D::calculateAndUpdateAux(int* aux){
  int stride_x = 1;
  int stride_y = getStrideY();
  int stride_z = getStrideZ();

  for(int z = 0; z < m_dimz; z++){
    for(int y = 0; y < m_dimy; y++){
      for(int x = 0; x < m_dimx; x++){
        int i = getGlobalIndex(x, y, z);

        int currentAux = aux[i];

        int beta_vx_wall_after = getBetaValue(aux[i + stride_x]);
        int beta_vx_wall_before = getBetaValue(currentAux);

        int beta_vy_wall_after = getBetaValue(aux[i + stride_y]);
        int beta_vy_wall_before = getBetaValue(currentAux);

        int beta_vz_wall_after = getBetaValue(aux[i + stride_z]);
        int beta_vz_wall_before = getBetaValue(currentAux);

        int beta_vx = 0;
        if(beta_vx_wall_after == 0){
          beta_vx = 1;
        }
        if(beta_vx_wall_before == 0){
          beta_vx = -1;
        }
        beta_vx += 1; //adjust it to 0 1 2

        //---------------------DUP Y--------------------
        int beta_vy = 0;
        if(beta_vy_wall_after == 0){
          beta_vy = 1;
        }
        if(beta_vy_wall_before == 0){
          beta_vy = -1;
        }
        beta_vy += 1; //adjust it to 0 1 2

        //---------------------DUP Z--------------------

        int beta_vz = 0;
        if(beta_vz_wall_after == 0){
          beta_vz = 1;
        }
        if(beta_vz_wall_before == 0){
          beta_vz = -1;
        }
        beta_vz += 1; //adjust it to 0 1 2

        aux[i] = setValues(aux[i], beta_vx, TWO_BIT, BETA_VX_LEVEL);
        aux[i] = setValues(aux[i], beta_vy, TWO_BIT, BETA_VY_LEVEL);
        aux[i] = setValues(aux[i], beta_vz, TWO_BIT, BETA_VZ_LEVEL);

        if(beta_vx != 1){
          //wall is forward
          //check four corners around

          int base = i;
          if(beta_vx == 2){
            base = i;
          }else{
            base = i + stride_x;
          }

          int a = 1 - getBetaValue(aux[base - stride_y]);
          int b = 1 - getBetaValue(aux[base + stride_y]);
          int c = 1 - getBetaValue(aux[base - stride_z]);
          int d = 1 - getBetaValue(aux[base + stride_z]);
          
          int norm = 1 + std::max(a, b) + std::max(c, d);
          aux[i] = setValues(aux[i], norm, TWO_BIT, BETA_VX_NORMALIZE);
        }

        if(beta_vy != 1){
          //wall is forward
          //check four corners around

          int base = i;
          if(beta_vy == 2){
            base = i;
          }else{
            base = i + stride_y;
          }

          int a = 1 - getBetaValue(aux[base - stride_x]);
          int b = 1 - getBetaValue(aux[base + stride_x]);
          int c = 1 - getBetaValue(aux[base - stride_z]);
          int d = 1 - getBetaValue(aux[base + stride_z]);
          
          int norm = 1 + std::max(a, b) + std::max(c, d);
          aux[i] = setValues(aux[i], norm, TWO_BIT, BETA_VY_NORMALIZE);
        }

        if(beta_vz != 1){
          //wall is forward
          //check four corners around

          int base = i;
          if(beta_vz == 2){
            base = i;
          }else{
            base = i + stride_z;
          }

          int a = 1 - getBetaValue(aux[base - stride_x]);
          int b = 1 - getBetaValue(aux[base + stride_x]);
          int c = 1 - getBetaValue(aux[base - stride_z]);
          int d = 1 - getBetaValue(aux[base + stride_z]);
          
          int norm = 1 + std::max(a, b) + std::max(c, d);
          aux[i] = setValues(aux[i], norm, TWO_BIT, BETA_VZ_NORMALIZE);
        }
      }
    }
  }

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(m_bufferAux, aux, m_volumeSize * sizeof(int), cudaMemcpyHostToDevice));
}


void Sim3D::setWall(int x, int y, int z, int val){
  int g = getGlobalIndex(x, y, z);

  int *aux = getAux();
  aux[g] = setValues(aux[g], val, ONE_BIT, BETA_SHIFT);
  calculateAndUpdateAux(aux);
  free(aux);
}

void Sim3D::setListener(int x_l, int y_l, int z_l){
  m_i_global_listener = getGlobalIndex(x_l, y_l, z_l);
  // int *aux = getAux();

  // //clear previous listener
  // for(int z = 0; z < m_dimz; z++){
  //   for(int y = 0; y < m_dimy; y++){
  //     for(int x = 0; x < m_dimx; x++){
  //       int g = getGlobalIndex(x, y, z);
  //       aux[g] = setValues(aux[g], 0, ONE_BIT, LISTENER_SHIFT);
  //     }
  //   }
  // }
  // int loc = getGlobalIndex(x_l, y_l, z_l);
  // aux[loc] = setValues(aux[loc], 1, ONE_BIT, LISTENER_SHIFT);
  // calculateAndUpdateAux(aux); //TODO don't need to actually update
  // free(aux);
}

void Sim3D::setPBore(int x_l, int y_l, int z_l){
  m_i_global_p_bore = getGlobalIndex(x_l, y_l, z_l);
  // int *aux = getAux();

  // //clear previous listener
  // for(int z = 0; z < m_dimz; z++){
  //   for(int y = 0; y < m_dimy; y++){
  //     for(int x = 0; x < m_dimx; x++){
  //       int g = getGlobalIndex(x, y, z);
  //       aux[g] = setValues(aux[g], 0, ONE_BIT, LISTENER_SHIFT);
  //     }
  //   }
  // }

  // int loc = getGlobalIndex(x_l, y_l, z_l);
  // aux[loc] = setValues(aux[loc], 1, ONE_BIT, LISTENER_SHIFT);
  // calculateAndUpdateAux(aux); //TODO don't need to actually update
  // free(aux);
}

void Sim3D::setPressureMouth(float pressure){
  m_p_mouth = pressure;
}



void Sim3D::setExcitor(int x, int y, int z, int val){
  int *aux = getAux();
  int loc = getGlobalIndex(x, y, z);
  aux[loc] = setValues(aux[loc], val, ONE_BIT, EXCITE_SHIFT);
  aux[loc] = setValues(aux[loc], 0, ONE_BIT, BETA_SHIFT);
  calculateAndUpdateAux(aux); 
  free(aux);
}

void Sim3D::reset(){
  // Toggle the buffers
  // Visual Studio 2005 does not like std::swap
  //    std::swap<float *>(bufferSrc, bufferDst);
  m_i = 0;

  printf("CLEARING TO ZERO \n");
  checkCudaErrors(cudaMemset(m_audioBuffer, 0, MAX_AUDIO_SIZE * sizeof(float)));
  checkCudaErrors(cudaMemset(m_bufferDst, 0, m_volumeSize * sizeof(float4)));
  checkCudaErrors(cudaMemset(m_bufferSrc, 0, m_volumeSize * sizeof(float4)));
  checkCudaErrors(cudaMemset(m_bufferAux, 0, m_volumeSize * sizeof(int)));

  
  int *aux = (int *) calloc(m_volumeSize, sizeof(int));
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(aux, m_bufferAux, m_volumeSize * sizeof(int), cudaMemcpyDeviceToHost));
  
  printf("SETTING BETA DEFAULT TO 1 \n");
  for(int i = 0; i < m_volumeSize; i++){
    aux[i] = setValues(aux[i], 1, ONE_BIT, BETA_SHIFT); // beta init to one
  }

  printf("SETTING SIGMA \n");

  for(int i = 0; i < PML + 1; i++){
    for(int x = i; x < m_dimx - i; x++){
      for(int y = i; y < m_dimy - i; y++){
        for(int z = i; z < m_dimz - i; z++){
          int g = getGlobalIndex(x, y, z);
          int sigma = PML - i;
          aux[g] = setValues(aux[g], sigma, THREE_BIT, SIGMA_SHIFT);
        }
      }
    }
  }

  printf("REUPLOADING \n");

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaMemcpy(m_bufferAux, aux, m_volumeSize * sizeof(int), cudaMemcpyHostToDevice));
  
}

void Sim3D::scheduleWall(int x, int y, int z, int val){
  std::vector<int> v;
  v.push_back(x);
  v.push_back(y);
  v.push_back(z);
  v.push_back(val);

  m_scheduledWalls.push_back(v);
}

void Sim3D::writeWalls(){
  int* aux = getAux();

  for(int i = 0; i < m_scheduledWalls.size(); i++){
    int x = m_scheduledWalls[i][0];
    int y = m_scheduledWalls[i][1];
    int z = m_scheduledWalls[i][2];
    int val = m_scheduledWalls[i][3];
    int g = getGlobalIndex(x, y, z);
    aux[g] = setValues(aux[g], val, ONE_BIT, BETA_SHIFT);
  }

  calculateAndUpdateAux(aux);

  free(aux);
}
