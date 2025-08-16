#include <stdio.h>
#include <time.h>

#define N 100

int main() 
{
    int i, j, k;
    double A[N][N], B[N][N], C[N][N];
    
    // Inicialização (opcional)
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = 1.0;  // Exemplo: preenche com 1.0
            B[i][j] = 1.0;  // Exemplo: preenche com 1.0
        }
    }
    
    clock_t start = clock();
    
    // Multiplicação
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            double sum = 0.0;
            for (k = 0; k < N; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tempo: %f segundos\n", time_spent);
    
    return 0;
}