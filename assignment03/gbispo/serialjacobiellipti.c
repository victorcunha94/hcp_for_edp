#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>

inline static double complex ftest(double x, double y, int m, int n) {
    return sin(m*M_PI*x)*sin(n*M_PI*y);
}

inline static double complex uexact(double x, double y, int m, int n) {
    return ftest(x,y,m,n)/(pow(M_PI,2)*(pow(m,2)+pow(n,2)));
}

inline static double complex uborder(double x, double y) {
    return 0.0;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s <N> <maxit> <m> <n> <omega> <method> <tol>\n", argv[0]);
        fprintf(stderr, "Methods: 'j' for Jacobi, 'jj' for modified Jacobi, 's' for SOR\n");
        return 1;
    }

    int N = atoi(argv[1]);
    int maxit = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    double omega = atof(argv[5]);
    char method = argv[6][0];
    double tol = atof(argv[7]);

    double h = 1.0/(N-1);
    double h2 = h*h;

    double complex *u = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *uold = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *f = (double complex*) malloc(N*N*sizeof(double complex));

    // Monta f e condicao de contorno
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            double x = j*h;
            double y = i*h;
            f[i*N+j] = ftest(x,y,m,n);
            if (i==0 || i==N-1 || j==0 || j==N-1)
                u[i*N+j] = uborder(x,y);
        }
    }

    // Loop principal de iteracoes
    int it;
    for (it=0; it<maxit; it++) {
        // Copia a solucao atual para uold
        memcpy(uold, u, N*N*sizeof(double complex));

        // Escolhe o metodo de iteracao
        switch (method) {
            case 'j': // Jacobi
                for (int i=1; i<N-1; i++) {
                    for (int j=1; j<N-1; j++) {
                        u[i*N+j] = 0.25 * ( uold[(i-1)*N+j] + uold[(i+1)*N+j] +
                                             uold[i*N+(j-1)] + uold[i*N+(j+1)]
                                             + h2 * f[i*N+j] );
                    }
                }
                break;
            case 'jj': // Jacobi sem o uold. Talvez seja o Gauss-Seidel
                for (int i=1; i<N-1; i++) {
                    for (int j=1; j<N-1; j++) {
                        u[i*N+j] = 0.25 * ( u[(i-1)*N+j] + u[(i+1)*N+j] +
                                             u[i*N+(j-1)] + u[i*N+(j+1)]
                                             + h2 * f[i*N+j] );
                    }
                }
                break;
            case 's': // SOR
                for (int i=1; i<N-1; i++) {
                    for (int j=1; j<N-1; j++) {
                        double complex u_jacobi_update = 0.25 * ( uold[(i-1)*N+j] + uold[(i+1)*N+j] +
                                                                   uold[i*N+(j-1)] + uold[i*N+(j+1)]
                                                                   + h2 * f[i*N+j] );
                        u[i*N+j] = omega*u_jacobi_update + (1.0-omega)*uold[i*N+j];
                    }
                }
                break;
            default:
                fprintf(stderr, "Metodo invalido.\n");
                    free(u);
                    free(uold);
                    free(f);
                return 1;
        }

        //  Criterio de parada 
        double res_norm_sq = 0.0;
        for (int i=1; i<N-1; i++) {
            for (int j=1; j<N-1; j++) {
                double complex r = f[i*N+j] + (u[(i-1)*N+j] + u[(i+1)*N+j] +
                                               u[i*N+(j-1)] + u[i*N+(j+1)] - 4.0*u[i*N+j])/h2;
                res_norm_sq += creal(r*conj(r));
            }
        }
        double res_norm = sqrt(res_norm_sq);

        if (res_norm < tol) {
            printf("Convergencia atingida em %d iteracoes. Residuo = %.6e\n", it+1, res_norm);
            break;
        }
    }

    if (it == maxit) {
        printf("Numero maximo de iteracoes atingido sem convergir.\n");
    }

    // Erro em relacao a solucao exata
    double err2 = 0.0, norm2 = 0.0;
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            double x = j*h;
            double y = i*h;
            double complex ue = uexact(x,y,m,n);
            double complex diff = u[i*N+j] - ue;
            err2  += creal(diff*conj(diff));
            norm2 += creal(ue*conj(ue));
        }
    }

    printf("Erro relativo = %.6e\n", sqrt(err2/norm2));
 // --- Salvando os dados para plotagem com Gnuplot ---
    FILE *fp = fopen("solucao_serial.dat", "w");
    if (fp == NULL) {
        fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
        // Lembre-se de liberar a memória antes de sair em caso de erro.
        free(u);
        free(uold);
        free(f);
        return 1;
    }

    //'u' é o array com a solução final
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = j * h;
            double y = i * h;
            // Para complexos, use 'creal' para a parte real.
            // Para a parte imaginária, use 'cimag'.
            fprintf(fp, "%f %f %f\n", x, y, creal(u[i * N + j]));
        }
        fprintf(fp, "\n"); // Linha em branco para Gnuplot (separador de 'linhas' do grid)
    }

    fclose(fp);
    printf("Solucao salva em 'solucao_serial.dat'\n");
    free(u);
    free(uold);
    free(f);

    return 0;
}