/*
Metodo de Jacobi matrix-free para resolver uma edp eliptica em [0,1]x[0,1]
com condicao de contorno u na fronteira = 0.
Usamos f = sin(m*pi*x)*sin(n*pi*y) para obter uma solucao exata,
a fim de poder calcular o erro.
Solucao exata: u = -f/(pi^2*(m^2+n^2)).
*/

#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include <time.h>

inline static double complex ftest(double x, double y, int m, int n) {
    return sin(m*M_PI*x)*sin(n*M_PI*y);
}

inline static double complex uexact(double x, double y, int m, int n) {
    return -ftest(x,y,m,n)/(pow(M_PI,2)*(pow(m,2)+pow(n,2)));
}

inline static double complex uborder(double x, double y) {
    if (x==0) return 0.0;
    if (x==1) return 0.0;
    if (y==0) return 0.0;
    if (y==1) return 0.0;
    //Deveria colocar um teste de compatibilidade aqui, 
    //tipo uborder(0,0) == uborder(0,1)
}


int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "Usage: %s <N> <maxit> <m> <n> <omega>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int maxit = atoi(argv[2]); 
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    double omega = atoi(argv[5]); 

    double h = 1.0/(N-1);
    double h2 = h*h;

    double complex *u    = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *uold = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *f    = (double complex*) malloc(N*N*sizeof(double complex));

    // monta f e condicao de contorno
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            double x = j*h;
            double y = i*h;
            f[i*N+j] = ftest(x,y,m,n);
            if (i==0 || i==N-1 || j==0 || j==N-1)
                u[i*N+j] = uborder(x,y);
        }
    }

    // Jacobi
    for (int it=0; it<maxit; it++) {
        // provavelmente ha um jeito mais inteligente de copiar u em uold
        for (int i=0; i<N*N; i++) uold[i] = u[i];

        for (int i=1; i<N-1; i++) {
            for (int j=1; j<N-1; j++) {
                u[i*N+j] = 0.25 * ( uold[(i-1)*N+j] + uold[(i+1)*N+j] +
                                    uold[i*N+(j-1)] + uold[i*N+(j+1)]
                                    - h2 * f[i*N+j] );
                u[i*N+j] = omega*u[i*N+j] + (1-omega)*uold[i*N+j]; //SOR

            }
        }
    }

    // Erro
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

    free(u);
    free(uold);
    free(f);

    return 0;
}
