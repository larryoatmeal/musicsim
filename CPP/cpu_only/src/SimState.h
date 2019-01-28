#ifndef SIM_STATE_H
#define SIM_STATE_H


#include <cmath>
#include <algorithm> 
#include "sim_constants.h"


float* alloc_grid(){
  return (float *)calloc(N_TOTAL, sizeof(float));
}

int index_of_padded(int w, int h){
  return (w + PAD_HALF) + (h + PAD_HALF) * (STRIDE_Y);
}

class SimState{
    public:
        SimState(){
            walls = alloc_grid();
            excitor = alloc_grid();
            beta = alloc_grid();
            sigma = alloc_grid();
            aux_data = (int *)calloc(N_TOTAL, sizeof(int));

            init();

            p = alloc_grid();
            v_x = alloc_grid();
            v_y = alloc_grid();

            p_prev = alloc_grid();
            v_x_prev = alloc_grid();
            v_y_prev = alloc_grid();
        }


        void step(){
            for(int x = 0; x < W; x++){
                for(int y = 0; y < H; y++){
                    AudioKernel(
                        x,
                        y
                    );
                }
            }
            std::swap<float *>(p_prev, p);
            std::swap<float *>(v_x_prev, v_x);
            std::swap<float *>(v_y_prev, v_y);
        }

        float read_pressure(){
            return p[listen_index];
        }

        ~SimState(){
            free(walls);
            free(excitor);
            free(beta);
            free(sigma);
            free(aux_data);
            free(p);
            free(v_x);
            free(v_y);
            free(p_prev);
            free(v_x_prev);
            free(v_y_prev);
        }

    int GetWidth(){
        return W;
    }
    int GetHeight(){
        return H;
    }
    float GetPressure(int x, int y){
        return p[index_of_padded(x, y)];
    }

    float* GetSigma(){
        return sigma;
    }
    int* GetAuxData(){
        return aux_data;
    }
    float *walls = 0;
    float *excitor = 0;
    float *beta = 0;
    float *sigma = 0;
    int *aux_data = 0;
    float *p = 0;
    float *v_x = 0;
    float *v_y = 0;
    float *p_prev = 0;
    float *v_x_prev = 0;
    float *v_y_prev = 0;
    private:
        


    void init(){
    //walls
        for(int i = 40; i < 150; i++){
        walls[index_of_padded(i, 50)] = 1; //
        walls[index_of_padded(i, 55)] = 1;
        }

        // walls[index_of_padded(130, 55)] = 0;
        // walls[index_of_padded(100, 55)] = 0;

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
        aux_data[i] = ( (int) walls[i] << 2) + ( (int) excitor[i] << 1) + (int) beta[i];
        }
    }

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
        int i
    )
    {
        float divergence = v_x_prev[i] - v_x_prev[i - STRIDE_X] + v_y_prev[i] - v_y_prev[i - STRIDE_Y];
        float p_denom = 1 + (1 - getBeta(aux_data, i) + sigma[i]) * DT;
        return (p_prev[i] - COEFF_DIVERGENCE * divergence)/p_denom;
    }


    void AudioKernel(
        int idx,
        int idy
        )
        {
        // int idx=blockIdx.x*blockDim.x+threadIdx.x;
        // int idy=blockIdx.y*blockDim.y+threadIdx.y;
        int i = (idx + PAD_HALF) + STRIDE_Y * (idy + PAD_HALF);

        //PRESSURE------------------------------

        float p_current = pressureStep(i);
        float p_right = pressureStep(i + STRIDE_X);
        float p_down = pressureStep(i + STRIDE_Y);
        p[i] = p_current;

        //VB------------------------------
        //TODO: not sure if this is supposed to be previous or next pressure
        float delta_p = std::max(P_MOUTH - p_prev[p_bore_index], 0.0f);
        float vb_x = 0;
        float vb_y = 0;
        int wall = getWall(aux_data, i);
        int excitor = getExcitor(aux_data, i);
        int wall_down = getWall(aux_data, i + STRIDE_Y);
        vb_x = excitor * (1 - delta_p / DELTA_P_MAX) * sqrt(2 * delta_p / RHO) * VB_COEFF / num_excite;
        vb_y = wall * ADMITTANCE * -1 * p_down + wall_down * ADMITTANCE * p_current;

        // if(idx == 40 && idy == 50){
        //     std::cout << wall << std::endl;
        //     std::cout << wall_down << std::endl;
        //     std::cout << p_current << std::endl;
        //     std::cout << p_down << std::endl;
        //     std::cout << vb_y << std::endl;
        // }

        //VELOCITY------------------------------
        int beta_current = getBeta(aux_data, i);

        float beta_x = std::min(beta_current, getBeta(aux_data, i + STRIDE_X));
        float grad_x = p_right - p_current;
        float sigma_prime_dt_x = (1 - beta_x + sigma[i]) * DT;
        v_x[i] = (beta_x * v_x_prev[i] - beta_x * beta_x * COEFF_GRADIENT * grad_x + sigma_prime_dt_x * vb_x)/(beta_x + sigma_prime_dt_x);

        float beta_y = std::min(beta_current, getBeta(aux_data, i + STRIDE_Y));
        float grad_y = p_down - p[i];
        float sigma_prime_dt_y = (1 - beta_y + sigma[i]) * DT;
        v_y[i] = (beta_y * v_y_prev[i] - beta_y * beta_y * COEFF_GRADIENT * grad_y + sigma_prime_dt_y * vb_y)/(beta_y + sigma_prime_dt_y);
    }

};
#endif // SIM_STATE_H
