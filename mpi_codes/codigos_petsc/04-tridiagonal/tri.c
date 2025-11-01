#include <petsc.h>

static char help[] = "Solve a tridiagonal system.\n";

int main(int argc,char **args) {
    Vec         u, f, uexact;
    Mat         A;
    KSP         ksp;
    PetscInt    m = 10, i, istart, iend, j[3];
    PetscReal   v[3], uval, norm_exact, errnorm;

    PetscCall(PetscInitialize(&argc,&args,NULL,help));

    // System dimension
    PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"options for tri",NULL);
    PetscCall(PetscOptionsInt("-m","dimension of linear system","tri.c",m,&m,NULL));
    PetscOptionsEnd();

    // Create vectors
    PetscCall(VecCreate(PETSC_COMM_WORLD,&u));
    PetscCall(VecSetSizes(u,PETSC_DECIDE,m));
    PetscCall(VecSetFromOptions(u));
    PetscCall(VecDuplicate(u,&f));
    PetscCall(VecDuplicate(u,&uexact));

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
        PetscCall(VecSetValues(uexact,1,&i,&uval,INSERT_VALUES));
    }

    // Finalize matrix and exact vector assembly
    PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
    PetscCall(VecAssemblyBegin(uexact));
    PetscCall(VecAssemblyEnd(uexact));

    // Compute f = A*uexact
    PetscCall(MatMult(A,uexact,f));

    // Create and configure KSP solver
    PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
    PetscCall(KSPSetOperators(ksp,A,A));
    PetscCall(KSPSetFromOptions(ksp));
    PetscCall(KSPSolve(ksp,f,u));  // solve A*u = f

    // Compute error u - uexact
    PetscCall(VecNorm(uexact,NORM_2,&norm_exact)); // ||uexact||_2
    PetscCall(VecAXPY(u,-1.0,uexact));       // u = u - uexact
    PetscCall(VecNorm(u,NORM_2,&errnorm));   // compute 2-norm of error
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
        "relative error for m = %d system is |u-uexact|_2/|uexact|_2 = %.3e\n",m,errnorm/norm_exact));

    // Free memory
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&A));
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));
    PetscCall(VecDestroy(&uexact));
    PetscCall(PetscFinalize());
    return 0;
}
