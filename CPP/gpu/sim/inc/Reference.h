#ifndef REFERENCE_H
#define REFERENCE_H

#include "FDTD3dGPU.h"
#include <cmath>
#include <algorithm> 
typedef float number;

namespace ReferenceSim{

int getBeta(
  int *aux,
  int i
){
  return std::min(aux[i] & (1 << 0), 1);
}

int getExcitor(
  int *aux,
  int i
){
  return std::min(aux[i] & (1 << 1), 1);
}

int getWall(
  int *aux,
  int i
){
  return std::min(aux[i] & (1 << 2), 1);
}

float pressureStep(
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

number* alloc_grid(){
  return (number *)calloc(N_TOTAL, sizeof(float));
}

int index_of_padded(int w, int h){
  return (w + PAD_HALF) + (h + PAD_HALF) * (STRIDE_Y);
}

void init(number *walls, number *excitor, number *beta, number *sigma, int *aux_cells){
    //walls

    for(int i = 40; i < 150; i++){
      walls[index_of_padded(i, 50)] = 1; //
      walls[index_of_padded(i, 55)] = 1;
    }

    walls[index_of_padded(55, 130)] = 0;
    walls[index_of_padded(55, 100)] = 0;

    //excitor
    for(int i = 51; i < 55; i++){
      excitor[index_of_padded(40, i)] = 1;
    }

    //beta
    for(int i = 0; i < N_TOTAL; i++){
      beta[i] = 1 - (walls[i] + excitor[i]);
    }

    //sigma
    for(int i = 0; i < PML_LAYERS + 1; i++){
      for(int x = i; x < W - i; x++){
        for(int y = i; y < H - i; y++){
          sigma[index_of_padded(x, y)] = 0.5 / DT * (PML_LAYERS - i)/PML_LAYERS;
        }
      }
    }

    for(int i = 0; i < N_TOTAL; i++){
      aux_cells[i] = ( (int) walls[i] << 2) + ( (int) excitor[i] << 1) + (int) beta[i];
    }
}


void genInitialState(
number **walls ,
number **excitor ,
number **beta ,
number **sigma ,
int **aux_cells,
number **p ,
number **v_x ,
number **v_y ,
number **p_prev ,
number **v_x_prev ,
number **v_y_prev 
){
    *walls = alloc_grid();
    *excitor = alloc_grid();
    *beta = alloc_grid();
    *sigma = alloc_grid();
    *aux_cells = (int *)calloc(N_TOTAL, sizeof(int));

    init(*walls, *excitor, *beta, *sigma, *aux_cells);

    *p = alloc_grid();
    *v_x = alloc_grid();
    *v_y = alloc_grid();

    *p_prev = alloc_grid();
    *v_x_prev = alloc_grid();
    *v_y_prev = alloc_grid();
}



void AudioKernel(
  float *v_x_prev,
  float *v_y_prev,
  float *p_prev,
  float *v_x,
  float *v_y,
  float *p,
  int *aux,
  float *sigma,
  int idx,
  int idy
)
{
  // int idx=blockIdx.x*blockDim.x+threadIdx.x;
  // int idy=blockIdx.y*blockDim.y+threadIdx.y;
  int i = (idx + PAD_HALF) + STRIDE_Y * (idy + PAD_HALF);

  //PRESSURE------------------------------

  float p_current = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i);
  float p_right = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i + STRIDE_X);
  float p_down = pressureStep(v_x_prev, v_y_prev, p_prev, aux, sigma, i + STRIDE_Y);
  p[i] = p_current;

  //VB------------------------------
  //TODO: not sure if this is supposed to be previous or next pressure
  float delta_p = std::max(P_MOUTH - p_prev[p_bore_index], 0.0f);
  float vb_x = 0;
  float vb_y = 0;
  int wall = getWall(aux, i);
  int excitor = getExcitor(aux, i);
  int wall_down = getWall(aux, i + STRIDE_Y);
  vb_x = excitor * (1 - delta_p / DELTA_P_MAX) * sqrt(2 * delta_p / RHO) * VB_COEFF / num_excite;
  vb_y = wall * ADMITTANCE * p_current + wall_down * -ADMITTANCE * p_down;

  //VELOCITY------------------------------
  int beta_current = getBeta(aux, i);

  float beta_x = std::min(beta_current, getBeta(aux, i + STRIDE_X));
  float grad_x = p_right - p_current;
  float sigma_prime_dt_x = (1 - beta_x + sigma[i]) * DT;
  v_x[i] = beta_x * v_x_prev[i] - beta_x * beta_x * COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x;

  float beta_y = std::min(beta_current, getBeta(aux, i + STRIDE_Y));
  float grad_y = p_down - p[i];
  float sigma_prime_dt_y = (1 - beta_y + sigma[i]) * DT;
  v_y[i] = beta_y * v_y_prev[i] - beta_y * beta_y * COEFF_GRADIENT * grad_y + sigma_prime_dt_y * vb_y;
}

void Reference(
  float *v_x_prev,
  float *v_y_prev,
  float *p_prev,
  float *v_x,
  float *v_y,
  float *p,
  int *aux,
  float *sigma
){
    for(int x = 0; x < W; x++){
        for(int y = 0; y < H; y++){
            AudioKernel(
                v_x_prev,
                v_y_prev,
                p_prev,
                v_x,
                v_y,
                p,
                aux,
                sigma,
                x,
                y
            );
        }
    }
}

}

#endif /* REFERENCE_H */
