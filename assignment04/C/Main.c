#include <petscksp.h>
#include <petscdmda.h>
static char help[] = "Grade 3d DMDA.\n";
int main(int argc, char **argv) {
    PetscMPIInt nprocs;
    PetscReal   norm, tol = 1000. * PETSC_MACHINE_EPSILON; /* norm of solution error */ 
    DM dmda3d;//estrutura 3d
    Mat matriz;
    // Inicializa PETSc e MPI
   
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));
    
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD,
         DM_BOUNDARY_NONE, 
         DM_BOUNDARY_NONE,
         DM_BOUNDARY_NONE,
         DMDA_STENCIL_STAR,
         100,100,100,
         PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
         1,1,NULL,NULL,NULL,&dmda3d 
        ));
    
    PetscCall(DMSetUp(dmda3d));//setando o objeto
    
    PetscCall(DMCreateMatrix(dmda3d,&matriz));


    PetscCall(PetscFinalize());
}