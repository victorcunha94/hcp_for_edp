#include <petscksp.h>
#include <petscdmda.h>
#include <math.h>



static char help[] = "Grade 3d DMDA.\n";





int main(int argc, char **argv) {
    PetscMPIInt nprocs,N_Pontos_x,N_Pontos_y,N_Pontos_z;
    N_Pontos_x=N_Pontos_y=N_Pontos_z=10;
    PetscReal   norm, tol = 1000. * PETSC_MACHINE_EPSILON; /* norm of solution error */ 
    DM dmda3d;//estrutura 3d
    Vec b,u_aprox,u_exact;
    Mat matriz;

    PC pre_condicionador;
    KSP krilov_sp;
    // Inicializa PETSc e MPI
   
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
         DM_BOUNDARY_NONE, 
         DM_BOUNDARY_NONE,
         DM_BOUNDARY_NONE,
         DMDA_STENCIL_STAR,
         N_Pontos_x,N_Pontos_y,N_Pontos_z,//dimensões
         PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
         1,1,NULL,NULL,NULL,&dmda3d 
        ));
    PetscCall(DMSetFromOptions(dmda3d));
    
    PetscCall(DMSetUp(dmda3d));//setando o objeto
    
    PetscCall(DMCreateMatrix(dmda3d,&matriz));
/*
    MatStencil linhas_dirichlet[4*N_Pontos_x*N_Pontos_z],colunas_dirichlet[4*N_Pontos_x*N_Pontos_z];
    PetscReal valores_0[4*N_Pontos_x*N_Pontos_z];
    for (int i=0;i<N_Pontos_x;i++){
        for (int j=0;j<N_Pontos_z;j++){
             int idx = 4 * (j + N_Pontos_x * i);//criando offset pra colocar todas as 4 paredes de uma vez
            linhas_dirichlet[idx]=colunas_dirichlet[idx]=(MatStencil){0,i,j,0};
            linhas_dirichlet[idx+1]=colunas_dirichlet[idx+1]=(MatStencil){N_Pontos_x-1,i,j,0};
            linhas_dirichlet[idx+2]=colunas_dirichlet[idx+2]=(MatStencil){i,0,j,0};
            linhas_dirichlet[idx+3]=colunas_dirichlet[idx+3]=(MatStencil){i,N_Pontos_y-1,j,0};
            for(int k=0;k<4;k++){
                valores_0[idx+k]=0;
            }
        }

    }
    PetscCall(MatSetValuesStencil(matriz,4*N_Pontos_x*N_Pontos_z,linhas_dirichlet,4*N_Pontos_x*N_Pontos_z,colunas_dirichlet,valores_0,INSERT_VALUES));
*/
    PetscCall(MatDestroy(&matriz));
    PetscCall(DMDestroy(&dmda3d));
    PetscCall(PetscFinalize());
}