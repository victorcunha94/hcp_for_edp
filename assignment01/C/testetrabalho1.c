#include <stdio.h>
#include <stdlib.h>
#include <time.h>


double **alloc_matrix(int n) {
    double **mat = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        mat[i] = calloc(n, sizeof(double));
    return mat;
}

void fill_matrix(int n, double **mat) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = rand() % 10;
}

//modelo ijk
void mat_mult_ijk(int n, double **A, double **B, double **C) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

//modelo ikj
void mat_mult_ikj(int n, double **A, double **B, double **C) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            for (int j = 0; j < n; j++)
                C[i][j] += A[i][k] * B[k][j];
}

void zero_matrix(int n, double **mat) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mat[i][j] = 0.0;
}

void free_matrix(int n, double **mat) {
    for (int i = 0; i < n; i++)
        free(mat[i]);
    free(mat);
}

int main() {
    int sizes[] = {1000, 1500, 2000, 3000}; //tamanhos das matrizes
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    srand((unsigned)time(NULL));



    for (int s = 0; s < num_sizes; s++) {
        int n = sizes[s];
        printf("tamanho da matriz: %dx%d\n", n, n);
        double **A = alloc_matrix(n);
        double **B = alloc_matrix(n);
        double **C = alloc_matrix(n);

        fill_matrix(n, A);
        fill_matrix(n, B);

        zero_matrix(n, C);
        clock_t start_ijk = clock();
        mat_mult_ijk(n, A, B, C);
        clock_t end_ijk = clock();
        double elapsed_ijk = (double)(end_ijk - start_ijk) / CLOCKS_PER_SEC;
        printf("modelo ijk: tempo = %.3f segundos\n", elapsed_ijk);

        zero_matrix(n, C);
        clock_t start_ikj = clock();
        mat_mult_ikj(n, A, B, C);
        clock_t end_ikj = clock();
        double elapsed_ikj = (double)(end_ikj - start_ikj) / CLOCKS_PER_SEC;
        printf("modelo ikj: tempo = %.3f segundos\n", elapsed_ikj);

        free_matrix(n, A);
        free_matrix(n, B);
        free_matrix(n, C);

        printf("\n");
    }

    return 0;
}

