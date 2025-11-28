#define _GNU_SOURCE
#include <math.h>
#include "Sol_exata.h"
#include <petsc.h>



PetscReal F_exata(PetscReal x,PetscReal y,PetscReal z){
    return(cos(2*x*M_PI)*cos(2*y*M_PI)*cos(2*z*M_PI));
}

PetscErrorCode Sol_exata(DM dmda3d, Vec u_exact){
    PetscInt       i, j, k;
    PetscReal      passo_x, passo_y,passo_z, x,y,z, ***altu_exact;
    DMDALocalInfo  info;

    PetscCall(DMDAGetLocalInfo(dmda3d,&info));

    passo_x=1.0/(info.mx-1);passo_y=1.0/(info.my-1);passo_z=1.0/(info.mz-1);

    PetscCall(DMDAVecGetArray(dmda3d, u_exact, &altu_exact));//cria array auxiliar
    
    for (k=info.zs;k<info.zs+info.zm;z++){
        z=passo_z*k;
        for (j = info.ys; j < info.ys+info.ym; j++) {
            y = j * passo_y;
            for (i = info.xs; i < info.xs+info.xm; i++) {
                x = i * passo_x;
                altu_exact[k][j][i]=F_exata(x,y,z);
            }
        }
    }
    PetscCall(DMDAVecRestoreArray(dmda3d, u_exact, &altu_exact));//manda os conteúdos do array pro vec original
    return 0;
}