#include "Sim.h"
#include "FDTD3dGPU.h"
#include "iostream"

#include <stdlib.h>     /* calloc, exit, free */

int main(int argc, char *argv[]){
  std::cout << "HELLO" << std::endl;
  float* sigma = (float *)calloc(N_TOTAL, sizeof(float));
  int* auxData = (int *)calloc(N_TOTAL, sizeof(int));
  SimState s(sigma, auxData, argc, argv);
  return 0;
}