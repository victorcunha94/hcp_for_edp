#include <petsc.h>

static char help[] = "Hello World usando PETSc.\n";

int main(int argc, char **argv) {
    PetscMPIInt rank, size;

    // Inicializa PETSc e MPI
    PetscCall(PetscInitialize(&argc, &argv, NULL, help));

    // Obtem rank e tamanho do comunicador
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

    // Cada processo imprime seu rank
    PetscCall(PetscPrintf(PETSC_COMM_SELF,
    "Hello from rank %d of %d processes!\n", rank, size));

    // Finaliza PETSc
    PetscCall(PetscFinalize());
    return 0;
}

