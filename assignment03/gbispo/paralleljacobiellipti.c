#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <mpi.h>

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
    
    if (argc < 8) {
        fprintf(stderr, "Usage: %s <N> <maxit> <m> <n> <omega> <method> <tol> <comm_freq>\n", argv[0]);
        fprintf(stderr, "Methods: 'j' for Jacobi, 'g' for modified Jacobi, 's' for SOR\n");
        MPI_Finalize();
        return 1;
    }
    
    int N = atoi(argv[1]);
    int maxit = atoi(argv[2]);
    int m = atoi(argv[3]);
    int n = atoi(argv[4]);
    double omega = atof(argv[5]);
    char method = argv[6][0];
    double tol = atof(argv[7]);
    int comm_freq = atoi(argv[8]);
    
    double h = 1.0/(N-1);
    double h2 = h*h;

    double complex *u = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *uold = (double complex*) calloc(N*N, sizeof(double complex));
    double complex *f = (double complex*) malloc(N*N*sizeof(double complex));

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            double x = j*h;
            double y = i*h;
            f[i*N+j] = ftest(x,y,m,n);
            if (i==0 || i==N-1 || j==0 || j==N-1)
                u[i*N+j] = uborder(x,y);
            }
        }
    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);


    MPI_Status status;
        
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    int periods[2] = {0, 0}; 
    MPI_Comm comm_cart;

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1 , &comm_cart);
    
    int coords[2];
    MPI_Cart_coords(comm_cart, rank, 2, coords);
    int north, south, east, west;
    MPI_Cart_shift(comm_cart, 0, 1, &north, &south);
    MPI_Cart_shift(comm_cart, 1, 1, &west, &east);
    int local_Nx = N / dims[1];
    int local_Ny = N / dims[0];
    if (coords[1] < N % dims[1]) local_Nx++;
    if (coords[0] < N % dims[0]) local_Ny++;
    local_Nx += 2; // Para os halos
    local_Ny += 2; 

    u

    MPI_Scatter);
    
    // Loop principal de iteracoes
    int it;
    for (it=0; it<maxit; it++) {
        // Comunica os halos a cada comm_freq iteracoes
        if (it % comm_freq == 0) {
            MPI_Request requests[8];
            if (north != MPI_PROC_NULL) {
                MPI_Irecv(&u[0*N], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     north, 0, comm_cart, &requests[0]);
                MPI_Isend(&u[1*N], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     north, 1, comm_cart, &requests[1]);
            } else {
                requests[0] = MPI_REQUEST_NULL;
                requests[1] = MPI_REQUEST_NULL;
            }
            if (south != MPI_PROC_NULL) {
                MPI_Irecv(&u[(local_Ny-1)*N], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     south, 1, comm_cart, &requests[2]);
                MPI_Isend(&u[(local_Ny-2)*N], local_Nx  -2, MPI_C_DOUBLE_COMPLEX,
                     south, 0, comm_cart, &requests[3]);
            } else {
                requests[2] = MPI_REQUEST_NULL;
                requests[3] = MPI_REQUEST_NULL;
            }
            if (west != MPI_PROC_NULL) {
                MPI_Irecv(&u[0*N+0], 1, MPI_C_DOUBLE_COMPLEX,
                     west, 2, comm_cart, &requests[4]);
                MPI_Isend(&u[0*N+1], 1, MPI_C_DOUBLE_COMPLEX,
                     west, 3, comm_cart, &requests[5]);
            } else {        
                requests[4] = MPI_REQUEST_NULL;
                requests[5] = MPI_REQUEST_NULL;
            }
            if (east != MPI_PROC_NULL) {
                MPI_Irecv(&u[0*N+(local_Nx-1)], 1, MPI_C_DOUBLE_COMPLEX,
                     east, 3, comm_cart, &requests[6]);
                MPI_Isend(&u[0*N+(local_Nx-2)], 1, MPI_C_DOUBLE_COMPLEX,
                     east, 2, comm_cart, &requests[7]);
            } else {    
                requests[6] = MPI_REQUEST_NULL;
                requests[7] = MPI_REQUEST_NULL;
            }
            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
        }


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
            case 'g': // Jacobi sem Old. Talvez seja o Gauss-Seidel
                for (int i=1; i<N-1; i++) {
                    for (int j=1; j<N-1; j++) {
                        u[i*N+j] = 0.25 * ( u[(i-1)*N+j] + u[(i+1)*N+j] +
                                             u[i*N+(j-1)] + u[i*N+(j+1)]
                                             - h2 * f[i*N+j] );
                    }
                }
                break;
            case 's': // SOR
                for (int i=1; i<N-1; i++) {
                    for (int j=1; j<N-1; j++) {
                        double complex u_jacobi_update = 0.25 * ( uold[(i-1)*N+j] + uold[(i+1)*N+j] +
                                                                   uold[i*N+(j-1)] + uold[i*N+(j+1)]
                                                                   - h2 * f[i*N+j] );
                        u[i*N+j] = omega*u_jacobi_update + (1.0-omega)*uold[i*N+j];
                    }
                }
                break;
            default:
                fprintf(stderr, "Metodo invalido.\n");
                MPI_Finalize();
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
    FILE *fp = fopen("solucao_parallel.dat", "w");
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
    printf("Solucao salva em 'solucao_parallel.dat'\n");
    free(u);
    free(uold);
    free(f);
    MPI_Finalize();

    return 0;
}