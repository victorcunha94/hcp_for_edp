#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define N 1000

int main() {
    // Alocação e inicialização
    double **A = (double**)malloc(N * sizeof(double*));
    double **B = (double**)malloc(N * sizeof(double*));
    double **C = (double**)malloc(N * sizeof(double*));
    
    for(int i = 0; i < N; i++) {
        A[i] = (double*)malloc(N * sizeof(double));
        B[i] = (double*)malloc(N * sizeof(double));
        C[i] = (double*)malloc(N * sizeof(double));
        for(int j = 0; j < N; j++) {
            A[i][j] = (double)rand()/RAND_MAX;
            B[i][j] = (double)rand()/RAND_MAX;
        }
    }

    // Medição de tempo de alta precisão
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);

    // Multiplicação real
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
    double time = (end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
    
    // Verificação
    double checksum = 0;
    for(int i = 0; i < N; i++) checksum += C[i][i];
    
    printf("Tempo: %.6f segundos\n", time);
    printf("Checksum: %f\n", checksum);

    // Liberação de memória
    for(int i = 0; i < N; i++) {
        free(A[i]); free(B[i]); free(C[i]);
    }
    free(A); free(B); free(C);
    
    return 0;
}