#ifndef _CONSTANTS_
#define _CONSTANTS_

#define RADIUS 2
#define PML 7

#define LISTEN_HZ (44100)


// __constant__ float CFL_MULTIPLIER;



#define OVERSAMPLE 13
#define SAMPLING_HZ ( LISTEN_HZ * OVERSAMPLE * 1.0)

#define Cs 3.4723e2 //speed of sound

#define SQRT_3 1.73205080757



#define a0 0.010369125065929718
#define a1 0.020738250131859436
#define a2 0.010369125065929718
#define b1 -1.691991340594157
#define b2 0.7334678408578759


// #define a0 0.00011827672500940037
// #define a1 0.00023655345001880073
// #define a2 0.00011827672500940037
// #define b1 -1.9690034772368936
// #define b2 0.9694765841369313


// #define a0 0.000029797435414526333
// #define a1 0.000059594870829052666
// #define a2 0.000029797435414526333
// #define b1  -1.9845008303721605
// #define b2  0.9846200201138187


// #define DT (1/SAMPLING_HZ)
// #define DS (DT * Cs * SQRT_3 * 1.75) 

// #define DS 1e-3
// #define DT (DS/Cs/SQRT_3/100)


// #define ZN 7357.0
// #define ZN 14000.0


#define RHO  1.1760
#define GAMMA  1.4017
#define MU  1.8460e-5
#define PRANDLT  0.7073


// #define COEFF_DIVERGENCE  (RHO * Cs * Cs * DT/DS)
// #define COEFF_GRADIENT (DT/RHO/DS)
// #define PML_SCALE (0.5 / DT / PML)
// #define ADMITTANCE (1.0/ZN)





// #define H_BORE  20  // 15mm, bore diameter of clarinet
// #define H_DS  (H_BORE * DS)
// #define W_J  1.2e-2
#define W_J  1.2e-2
#define H_R  6e-4
// #define H_R  6e-2


#define K_R  8e6
#define DELTA_P_MAX  (K_R * H_R)
// #define W_J_H_R  (W_J * H_R)
// #define VB_COEFF  (W_J_H_R / H_DS)




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