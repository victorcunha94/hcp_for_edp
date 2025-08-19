#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#define N 1000

void initialize(double **matrix) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            matrix[i][j] = (double)rand()/RAND_MAX;
        }
    }
}

double ijk_multiply(double **A, double **B, double **C) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceCounter(&start);
    
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            double sum = 0;
            for(int k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&freq);
    return (double)(end.QuadPart - start.QuadPart)/freq.QuadPart;
}

double ikj_multiply(double **A, double **B, double **C) {
    LARGE_INTEGER start, end, freq;
    QueryPerformanceCounter(&start);
    
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++) {
            double r = A[i][k];
            for(int j = 0; j < N; j++) {
                C[i][j] += r * B[k][j];
            }
        }
    }
    
    QueryPerformanceCounter(&end);
    QueryPerformanceFrequency(&freq);
    return (double)(end.QuadPart - start.QuadPart)/freq.QuadPart;
}

int main() {
    // Alocação
    double **A = (double**)malloc(N * sizeof(double*));
    double **B = (double**)malloc(N * sizeof(double*));
    double **C1 = (double**)malloc(N * sizeof(double*));
    double **C2 = (double**)malloc(N * sizeof(double*));
    
    for(int i = 0; i < N; i++) {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C1[i] = (double*)malloc(N * sizeof(double));
        C2[i] = (double*)malloc(N * sizeof(double));
    }
    
    // Inicialização
    srand(time(NULL));
    initialize(A);
    initialize(B);
    
    // Teste IJK
    double time_ijk = ijk_multiply(A, B, C1);
    
    // Teste IKJ
    double time_ikj = ikj_multiply(A, B, C2);
    
    printf("Tempo IJK: %.6f segundos\n", time_ijk);
    printf("Tempo IKJ: %.6f segundos\n", time_ikj);
    printf("IKJ foi %.2fx mais rápido\n", time_ijk/time_ikj);
    
    // Verificação
    double checksum = 0;
    for(int i = 0; i < N; i++) checksum += C1[i][i] - C2[i][i];
    printf("Diferença checksum: %e (deve ser próximo de 0)\n", checksum);
    
    // Liberação
    for(int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(C1[i]); free(C2[i]);
    }
    free(A); free(B); free(C1); free(C2);
    
    return 0;
}