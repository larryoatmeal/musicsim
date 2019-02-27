#ifndef _CONSTANTS_
#define _CONSTANTS_

#define RADIUS 2
#define PML 5

#define SAMPLING_HZ (176400.0)

#define DT (1/SAMPLING_HZ)

#define Cs 3.4723e2 //speed of sound

#define SQRT_3 1.73205080757
#define DS (DT * Cs * SQRT_3)

#define ZN 7357
#define ADMITTANCE (1/ZN)


#define RHO  1.1760
#define GAMMA  1.4017
#define MU  1.8460e-5
#define PRANDLT  0.7073

#define COEFF_DIVERGENCE  (RHO * Cs * Cs * DT/DS)
#define COEFF_GRADIENT (DT/RHO/DS)


#define H_BORE  0.015  // 15mm, bore diameter of clarinet
#define H_DS  (H_BORE * DS)
#define W_J  1.2e-2
#define H_R  6e-4
#define K_R  8e6
#define DELTA_P_MAX  (K_R * H_R)
#define W_J_H_R  (W_J * H_R)
#define VB_COEFF  (W_J_H_R / H_DS)


#define PML_SCALE (0.5 / DT / PML_LAYERS)



#endif