#include <petsc.h>

static char help[] = "Vector operations with PETSc.\n";

int main(int argc, char **argv) {
    Vec u, f;
    PetscInt i, n = 50;
    PetscReal dot, norm_u, norm_f;
    PetscInt istart, iend;
    PetscScalar val;

    // Initialize PETSc and MPI
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    // Create vector u
    PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
    PetscCall(VecSetSizes(u, PETSC_DECIDE, n));
    PetscCall(VecSetFromOptions(u));

    // Set prefix "u_"
    PetscCall(VecSetOptionsPrefix(u, "u_"));

    // Get local ownership range of indices
    PetscCall(VecGetOwnershipRange(u, &istart, &iend));
    for (i = istart; i < iend; i++) {
        val = (PetscScalar)(i);
        PetscCall(VecSetValue(u, i, val, INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(u));
    PetscCall(VecAssemblyEnd(u));

    // Create vector f as a duplicate of u
    PetscCall(VecDuplicate(u, &f));
    PetscCall(VecSetOptionsPrefix(f, "f_"));
    for (i = istart; i < iend; i++) {
        PetscCall(VecSetValue(f, i, (PetscScalar)(2 * i), INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(f));
    PetscCall(VecAssemblyEnd(f));

    // Dot product
    PetscCall(VecDot(u, f, &dot));

    // Norms
    PetscCall(VecNorm(u, NORM_2, &norm_u));
    PetscCall(VecNorm(f, NORM_2, &norm_f));

    // Sum u + f
    PetscCall(VecAXPY(u, 1.0, f)); // u = u + 1*f

    // Print results
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nDot product <u,f> = %g\n", dot));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of u = %g\n", norm_u));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of f = %g\n", norm_f));

    // Destroy vectors
    PetscCall(VecDestroy(&u));
    PetscCall(VecDestroy(&f));

    // Finalize PETSc
    PetscCall(PetscFinalize());
    return 0;
}

