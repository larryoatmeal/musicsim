#define RHO  1.1760
#define C  3.4723e2
#define GAMMA  1.4017
#define MU  1.8460e-5
#define PRANDLT  0.7073
#define DS  3.83e-3

// #define DT  (DS / (sqrt(2) * C) * 0.999999)  // make sure we're actually below the condition

#define DT  (DS / (1.41421356237 * C) * 0.999999)  // make sure we're actually below the condition

#define DT_LISTEN  (1.0 / 44100)
#define SAMPLE_EVERY_N  round(DT_LISTEN / DT)
#define REAL_LISTENING_FREQUENCY  1 / (SAMPLE_EVERY_N * DT)



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

#define PAD_HALF 1

#define W_PADDED (W + 2 * PAD_HALF)
#define H_PADDED (H + 2 * PAD_HALF)

#define STRIDE_Y (W_PADDED)
#define STRIDE_X 1


#define N_TOTAL (W_PADDED * H_PADDED)

#define p_bore_index  (41 + PAD_HALF) + (53 + PAD_HALF) * (STRIDE_Y)
#define num_excite  3
#define listen_index  (155 + PAD_HALF) + (45 + PAD_HALF) * (STRIDE_Y)




#define P_MOUTH  3000