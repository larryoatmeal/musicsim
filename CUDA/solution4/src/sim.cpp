#include <iostream>
#include <string>
#include <math.h>
#include <fstream>
using namespace std;

typedef float number;

number RHO = 1.1760;
number C = 3.4723e2;
number GAMMA = 1.4017;
number MU = 1.8460e-5;
number PRANDLT = 0.7073;
number DS = 3.83e-3;
number DT = DS / (sqrt(2) * C) * 0.999999;  // make sure we're actually below the condition;
number DT_LISTEN = 1.0 / 44000;
int SAMPLE_EVERY_N = (int) round(DT_LISTEN / DT);
number REAL_LISTENING_FREQUENCY = 1 / (SAMPLE_EVERY_N * DT);
number AN = 0.01;
number ADMITTANCE = 1 / (RHO * C * (1 + sqrt(1 - AN)) / (1 - sqrt(1 - AN)));

number COEFF_DIVERGENCE = RHO * C * C * DT/DS;
number COEFF_GRADIENT = DT/RHO/DS;

number H_BORE = 0.015;  // 15mm, bore diameter of clarinet
number H_DS = H_BORE * DS;
number W_J = 1.2e-2;
number H_R = 6e-4;
number K_R = 8e6;
number DELTA_P_MAX = K_R * H_R;
number W_J_H_R = W_J * H_R;
number VB_COEFF = W_J_H_R / H_DS;

int PML_LAYERS = 6;

int W = 220;
int H = 110;
int STRIDE_X = 1;
int STRIDE_Y = W;





float P_MOUTH = 3000;

int p_bore_index = 0;
int num_excite = 3;
int listen_index = 0;


int SIMULATION_SIZE = W * H;
int PADDING_HALF = STRIDE_Y + STRIDE_X;
int PADDING = PADDING_HALF * 2;
// int PADDED_SIZE = (W + 2) * (H + 2);
// int PAD_ONE_END = (PADDED_SIZE - SIMULATION_SIZE) / 2;


int index_of(int w, int h){
  return w + h * STRIDE_Y;
}

int index_of_padded(int w, int h){
  return index_of(w, h) + PADDING_HALF;
}



int N = 10000;

number* alloc_grid(){
  return (number *)calloc(SIMULATION_SIZE + PADDING, sizeof(number));
}

void init(number *walls, number *excitor, number *beta, number *sigma){
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

    p_bore_index = index_of_padded(41, 53);
    listen_index = index_of_padded(45, 155);


    //beta
    for(int i = PADDING_HALF; i < SIMULATION_SIZE + PADDING_HALF; i++){
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

}

void swapPtr(number** p1, number **p2)
{
  number *temp = *p1;
  *p1 = *p2;
  *p2 = temp;
}


int main() {
  cout << "Hello world!" << endl;

  number *walls = alloc_grid();
  number *excitor = alloc_grid();
  number *beta = alloc_grid();
  number *sigma = alloc_grid();

  init(walls, excitor, beta, sigma);


  number *p = alloc_grid();
  number *v_x = alloc_grid();
  number *v_y = alloc_grid();

  number *p_prev = alloc_grid();
  number *v_x_prev = alloc_grid();
  number *v_y_prev = alloc_grid();


  for(int n = 0; n < N; n++){
    //note the way this works with the padding:
    //for x derivatives, will WRAP to next line
    //probably okay since in PML layer anyway

    for(int i = PADDING_HALF; i < SIMULATION_SIZE + PADDING_HALF; i++){
      // gradient
      number divergence = v_x_prev[i] - v_x_prev[i - STRIDE_X] + v_y_prev[i] - v_y_prev[i - STRIDE_Y];
      number p_denom = 1 + (1 - beta[i] + sigma[i]) * DT;
      p[i] = (p_prev[i] - COEFF_DIVERGENCE * divergence)/p_denom;
    }

    number delta_p = P_MOUTH - p[p_bore_index];
    for(int i = PADDING_HALF; i < SIMULATION_SIZE + PADDING_HALF; i++){

      number vb_x = 0;
      number vb_y = 0;

      //check if wall
      if(excitor[i] && delta_p > 0){
        vb_x = (1 - delta_p / DELTA_P_MAX) * sqrt(2 * delta_p / RHO) * VB_COEFF / num_excite;
      }
      else if(walls[i]){
        vb_y = ADMITTANCE * p[i];
      }
      else if(walls[i + STRIDE_Y]){
        vb_y = -ADMITTANCE * p[i + STRIDE_Y];
      }

      number beta_x = min(beta[i], beta[i + STRIDE_X]);
      number grad_x = p[i + STRIDE_X] - p[i];
      number sigma_prime_dt_x = (1 - beta_x + sigma[i]) * DT;
      v_x[i] = beta_x * v_x_prev[i] - beta_x * beta_x * COEFF_DIVERGENCE * grad_x + sigma_prime_dt_x * vb_x;

      number beta_y = min(beta[i], beta[i + STRIDE_Y]);
      number grad_y = p[i + STRIDE_Y] - p[i];
      number sigma_prime_dt_y = (1 - beta_y + sigma[i]) * DT;
      v_y[i] = beta_y * v_y_prev[i] - beta_y * beta_y * COEFF_DIVERGENCE * grad_y + sigma_prime_dt_y * vb_y;
    }

    if(n % 1000 == 0){
      cout << "ITER: " << n << endl;
      ofstream myfile;
      myfile.open ("example.txt");
      myfile << "Writing this to a file.\n";
      for(int y = 0; y < H; y++){
        for(int x = 0; x < W; x++){
          myfile << sigma[index_of_padded(x, y)] << " ";
        }
        myfile << endl;
      }
      myfile.close();
    }



    for(int i = PADDING_HALF; i < SIMULATION_SIZE + PADDING_HALF; i++){
      // gradient

      // p[i] = (p_prev[i] - COEFF_DIVERGENCE * divergence)/p_denom;
      cout << p[i] << endl;
    }
    // cout << p[listen_index] << endl;
    //swap
    swapPtr(&p, &p_prev);
    swapPtr(&v_x, &v_x_prev);
    swapPtr(&v_y, &v_y_prev);
  }




  free(beta);
  free(sigma);
  free(p);
  free(v_x);
  free(v_y);
  free(p_prev);
  free(v_x_prev);
  free(v_y_prev);


  return 0;
}
