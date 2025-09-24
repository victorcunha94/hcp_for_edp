//
// Created by Usuário on 24/09/2025.
//

#include "Jacobi.h"

#include <stdlib.h>

float** Jacobi(unsigned int max_iter,float tolerancia, float* diagonal, float* b,
              float** resto, float** chute, unsigned int dimensão_X, unsigned int dimensão_Y) {

    float** resultado=malloc(dimensão_Y*sizeof(float));
    for(unsigned int i=0;i<dimensão_Y;i++) {
        resultado[i]=malloc(dimensão_X*sizeof(float));
    }
    for(unsigned int i=0;i<max_iter;i++) {
        for (unsigned int j=0;j<dimensão_Y;j++) {
            for (unsigned int k=0;k<dimensão_X;k++) {
                resultado[k][j]=(chute[k-1][j]+chute[k+1][j]+chute[k][j-1]+chute[k][j+1])/4;
            }
        }

    }

    return resultado;
}
