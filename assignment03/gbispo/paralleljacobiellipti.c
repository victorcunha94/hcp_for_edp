#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <complex.h>
#include <string.h>
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
        fprintf(stderr, "Usage: %s <grid size> <maxit> <m> <n> <omega> <method> <tol> <comm_freq>\n", argv[0]);
        fprintf(stderr, "Methods: 'j' for Jacobi, 'g' for modified Jacobi, 's' for SOR\n");
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

    double inicio_comunicação,total_comunicação=0,inicio_execução,total_execução=0;

    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int convergencia_local=0;
    int convergencia_global=0;
        
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

    int global_i_offset=0;
    for (int i = 0; i < coords[0]; i++) {
        global_i_offset += N / dims[0];
        if (i < N % dims[0]) {
            global_i_offset++;
        }
    }
    int global_j_offset=0;
    for (int j = 0; j < coords[1]; j++) {
        global_j_offset += N / dims[1];
        if (j < N % dims[1]) {
            global_j_offset++;
        }
    }

    if (coords[1] < N % dims[1]) {
        local_Nx++;
    }
    if (coords[0] < N % dims[0]) {
        local_Ny++;
    }
    local_Nx += 2; // Para os halos
    local_Ny += 2;

    double complex *u_local = (double complex*) calloc(local_Ny * local_Nx, sizeof(double complex));
    double complex *uold_local = (double complex*) calloc(local_Ny * local_Nx, sizeof(double complex));
    double complex *f_local = (double complex*) calloc(local_Ny * local_Nx, sizeof(double complex));

    for (int i = 0; i < local_Ny; i++) {
        for (int j = 0; j < local_Nx; j++) {

            int global_i = global_i_offset + i - 1; // -1 pro halo
            int global_j = global_j_offset + j - 1;

            double x = global_j * h;
            double y = global_i * h;

            if (i > 0 && i < local_Ny - 1 && j > 0 && j < local_Nx - 1) {
                f_local[i * local_Nx + j] = ftest(x, y, m, n);
            }

            if (north == MPI_PROC_NULL && i == 1) {
                u_local[i * local_Nx + j] = uborder(x, y);
            }

            if (south == MPI_PROC_NULL && i == local_Ny - 2) {
                u_local[i * local_Nx + j] = uborder(x, y);
            }

            if (west == MPI_PROC_NULL && j == 1) {
                u_local[i * local_Nx + j] = uborder(x, y);
            }

            if (east == MPI_PROC_NULL && j == local_Nx - 2) {
                u_local[i * local_Nx + j] = uborder(x, y);
            }
        }
    }

    MPI_Datatype coluna;
    MPI_Type_vector(local_Ny-2,1,local_Nx,MPI_C_DOUBLE_COMPLEX, &coluna);
    MPI_Type_commit(&coluna);
    // Loop principal de iteracoes
    int it;
    for (it=0; it<maxit; it++) {
        // Comunica os halos a cada comm_freq iteracoes

        if (it % comm_freq == 0) {
            inicio_comunicação=MPI_Wtime();
            MPI_Request requests[8];

            if (north != MPI_PROC_NULL) {
                MPI_Irecv(&u_local[1], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     north, 0, comm_cart, &requests[0]);
                MPI_Isend(&u_local[1*local_Nx+1], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     north, 1, comm_cart, &requests[1]);
            } else {
                requests[0] = MPI_REQUEST_NULL;
                requests[1] = MPI_REQUEST_NULL;
            }
            if (south != MPI_PROC_NULL) {
                MPI_Irecv(&u_local[(local_Ny-1)*local_Nx+1], local_Nx-2, MPI_C_DOUBLE_COMPLEX,
                     south, 1, comm_cart, &requests[2]);
                MPI_Isend(&u_local[(local_Ny-2)*local_Nx+1], local_Nx  -2, MPI_C_DOUBLE_COMPLEX,
                     south, 0, comm_cart, &requests[3]);
            } else {
                requests[2] = MPI_REQUEST_NULL;
                requests[3] = MPI_REQUEST_NULL;
            }
            if (west != MPI_PROC_NULL) {
                MPI_Irecv(&u_local[local_Nx], 1, coluna,
                     west, 2, comm_cart, &requests[4]);
                MPI_Isend(&u_local[local_Nx+1], 1, coluna,
                     west, 3, comm_cart, &requests[5]);
            } else {        
                requests[4] = MPI_REQUEST_NULL;
                requests[5] = MPI_REQUEST_NULL;
            }
            if (east != MPI_PROC_NULL) {
                MPI_Irecv(&u_local[local_Nx+(local_Nx-1)], 1, coluna,
                     east, 3, comm_cart, &requests[6]);
                MPI_Isend(&u_local[local_Nx+(local_Nx-2)], 1, coluna,
                     east, 2, comm_cart, &requests[7]);
            } else {    
                requests[6] = MPI_REQUEST_NULL;
                requests[7] = MPI_REQUEST_NULL;
            }
            MPI_Waitall(8, requests, MPI_STATUSES_IGNORE);
            total_comunicação+=MPI_Wtime()-inicio_comunicação;
        }

        inicio_execução=MPI_Wtime();

        // Copia a solucao atual para uold
        memcpy(uold_local, u_local, local_Nx*local_Ny*sizeof(double complex));

        // Escolhe o metodo de iteracao
        switch (method) {
            case 'j': // Jacobi
                for (int i=1; i<local_Ny-1; i++) {
                    for (int j=1; j<local_Nx-1; j++) {
                        u_local[i*local_Nx+j] = 0.25 * ( uold_local[(i-1)*local_Nx+j] + uold_local[(i+1)*local_Nx+j] +
                                             uold_local[i*local_Nx+(j-1)] + uold_local[i*local_Nx+(j+1)]
                                             - h2 * f_local[i*local_Nx+j] );
                    }
                }
                break;
            case 's': // SOR
                for (int i=1; i<local_Ny-1; i++) {
                    for (int j=1; j<local_Nx-1; j++) {
                        double complex u_jacobi_update = 0.25 * ( uold_local[(i-1)*local_Nx+j] + uold_local[(i+1)*local_Nx+j] +
                                                                   uold_local[i*local_Nx+(j-1)] + uold_local[i*local_Nx+(j+1)]
                                                                   - h2 * f_local[i*local_Nx+j] );
                        u_local[i*local_Nx+j] = omega*u_jacobi_update + (1.0-omega)*uold_local[i*N+j];
                    }
                }
                break;
            default:
                fprintf(stderr, "Metodo invalido.\n");
                free(u_local);
                free(uold_local);
                free(f_local);
                MPI_Finalize();
                return 1;
        }

        //  Criterio de parada

        double res_norm_sq = 0.0;
        double res_norm_sq_global;
        for (int i=1; i<local_Ny-1; i++) {
            for (int j=1; j<local_Nx-1; j++) {
                double complex r = f_local[i*local_Nx+j] - (u_local[(i-1)*local_Nx+j] + u_local[(i+1)*local_Nx+j] +
                                               u_local[i*local_Nx+(j-1)] + u_local[i*local_Nx+(j+1)] - 4.0*u_local[i*local_Nx+j])/h2;
                res_norm_sq += creal(r*conj(r));
            }
        }
        MPI_Allreduce(&res_norm_sq,&res_norm_sq_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        double res_norm = sqrt(res_norm_sq_global);
        if (res_norm < tol && !convergencia_local) {
            convergencia_local=1;
            printf("Convergencia atingida em %d iteracoes. Residuo = %.6e\n", it+1, res_norm);
        }
        total_execução+=MPI_Wtime()-inicio_execução;
        inicio_comunicação=MPI_Wtime();
        MPI_Allreduce(&convergencia_local, &convergencia_global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        total_comunicação+=MPI_Wtime()-inicio_comunicação;
        if (convergencia_global) {
            break;
        }
    }

    if (it == maxit) {
        printf("Numero maximo de iteracoes atingido sem convergir.\n");
    }

    // Erro em relacao a solucao exata

    if (rank==0) {
        printf("tempo de execução: %lf\n tempo de comunicação: %lf ",total_execução,total_comunicação);
        double err2 = 0.0, norm2 = 0.0;
        for (int i=0; i<N; i++) {
            for (int j=0; j<N; j++) {
                double x = j*h;
                double y = i*h;
                double complex ue = uexact(x,y,m,n);
                double complex diff = u_local[i*N+j] - ue;
                err2  += creal(diff*conj(diff));
                norm2 += creal(ue*conj(ue));
            }
        }

        printf("Erro relativo = %.6e\n", sqrt(err2/norm2));
        // --- Salvando os dados para plotagem com Gnuplot ---
        FILE *fp = fopen("solucao_parallel.dat", "w");
        if (fp == NULL) {
            fprintf(stderr, "Erro ao abrir o arquivo para escrita.\n");
            free(u_local);
            free(uold_local);
            free(f_local);
            MPI_Finalize();
            return 1;
        }

        //'u' é o array com a solução final
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double x = j * h;
                double y = i * h;
                // Para complexos, use 'creal' para a parte real.
                // Para a parte imaginária, use 'cimag'.
                fprintf(fp, "%f %f %f\n", x, y, creal(u_local[i * N + j]));
            }
            fprintf(fp, "\n"); // Linha em branco para Gnuplot (separador de 'linhas' do grid)
        }

        fclose(fp);
        printf("Solucao salva em 'solucao_parallel.dat'\n");
    }
    free(u_local);
    free(uold_local);
    free(f_local);
    MPI_Type_free(&coluna);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}