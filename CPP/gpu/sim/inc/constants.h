#ifndef _CONSTANTS_
#define _CONSTANTS_

#define RADIUS 2
#define PML 5

#define SAMPLING_HZ (176400.0)

#define DT (1/SAMPLING_HZ)

#define Cs 3.4723e2 //speed of sound

#define SQRT_3 1.73205080757
#define DS (DT * Cs * SQRT_3)

#define ZN 7357.0
#define ADMITTANCE (1.0/ZN)


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


#define PML_SCALE (0.5 / DT / PML)


#define ONE_BIT (1)
#define TWO_BIT (3) //0b11
#define THREE_BIT (7) //0b111
#define FOUR_BIT (15) //0b1111

#define SIGMA_SHIFT (0) //3
#define BETA_SHIFT (SIGMA_SHIFT + 3)//1
#define BETA_VX_LEVEL (BETA_SHIFT + 1) //2
#define BETA_VX_NORMALIZE (BETA_VX_LEVEL + 2) //2
#define BETA_VY_LEVEL (BETA_VX_NORMALIZE + 2) //2
#define BETA_VY_NORMALIZE (BETA_VY_LEVEL + 2) //2
#define BETA_VZ_LEVEL (BETA_VY_NORMALIZE + 2) //2
#define BETA_VZ_NORMALIZE (BETA_VZ_LEVEL + 2) //2
#define LISTENER_SHIFT (BETA_VZ_NORMALIZE +2)//1
#define EXCITE_SHIFT (LISTENER_SHIFT + 1)     //1

#endif