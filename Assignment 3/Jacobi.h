//
// Created by Usuário on 24/09/2025.
//

#ifndef HCP_FOR_EDP_JACOBI_H
#define HCP_FOR_EDP_JACOBI_H
float** Jacobi(unsigned int max_iter,float tolerancia, float* diagonal, float* b,
    float** resto, float** chute,  unsigned int dimensão_X, unsigned int dimensão_Y);
#endif //HCP_FOR_EDP_JACOBI_H