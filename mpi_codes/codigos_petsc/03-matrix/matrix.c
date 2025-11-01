#include <petsc.h>

static char help[] = "Create a matrix and compute a mat-vec product.\n";

int main(int argc,char **args) {
    Vec         u, f;
    Mat         A;
    PetscInt    m = 10, i, istart, iend, j[3];
    PetscReal   v[3], uval;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    // System dimension
    PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"options for matrix",NULL);
    PetscCall(PetscOptionsInt("-m","dimension of linear system","matrix.c",m,&m,NULL));
    PetscOptionsEnd();

    // Create vectors
    PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
    PetscCall(VecSetSizes(u,PETSC_DECIDE,m));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u,&f));

    // Create matrix
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(MatGetOwnershipRange(A,&istart,&iend));

    // Assemble tridiagonal matrix and exact vector
    for (i=istart; i<iend; i++) {
        if (i == 0) {
            v[0] = 2.0;  v[1] = -1.0;
            j[0] = 0;    j[1] = 1;
            PetscCall(MatSetValues(A,1,&i,2,j,v,INSERT_VALUES));
        } else {
            v[0] = -1.0;  v[1] = 2.0;  v[2] = -1.0;
            j[0] = i-1;   j[1] = i;    j[2] = i+1;
            if (i == m-1) {
                PetscCall(MatSetValues(A,1,&i,2,j,v,INSERT_VALUES));
            } else {
                PetscCall(MatSetValues(A,1,&i,3,j,v,INSERT_VALUES));
            }
        }
        uval = PetscExpReal(PetscCosReal(i));
        PetscCall(VecSetValues(u,1,&i,&uval,INSERT_VALUES));
    }

    // Finalize matrix and vector assembly
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));

    // Compute f = A*u
    PetscCall(MatMult(A,u,f));

    // Free memory
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));
    PetscCall(PetscFinalize());
    return 0;
}
