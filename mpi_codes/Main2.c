#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int N_tarefas, rank, N_recebidos, cabeça=0;
    double inicio,fim;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N_tarefas);

    int carga=4032;
    int N_envios=N_recebidos=carga/N_tarefas;

    float* buffer_origem_A;
    float* buffer_origem_B;
    float* buffer_retorno_A;
    float* buffer_retorno_B;

    if(rank==cabeça) {
        buffer_origem_A = malloc(carga*sizeof(float));
        buffer_origem_B = malloc(carga*sizeof(float));
        buffer_retorno_A = malloc(carga*sizeof(float));
        buffer_retorno_B = malloc(carga*sizeof(float));
        for(int i=0; i<carga; i++) {
            buffer_origem_A[i] = rand()%10;
            buffer_origem_B[i] = rand()%10;
        }
    }

    float* buffer_destino_A = malloc((carga/N_tarefas)*sizeof(float));
    float* buffer_destino_B = malloc((carga/N_tarefas)*sizeof(float));

    inicio=MPI_Wtime();

    MPI_Scatter(buffer_origem_A,N_envios,MPI_FLOAT,buffer_destino_A,N_recebidos,MPI_FLOAT,cabeça,MPI_COMM_WORLD);
    MPI_Scatter(buffer_origem_B,N_envios,MPI_FLOAT,buffer_destino_B,N_recebidos,MPI_FLOAT,cabeça,MPI_COMM_WORLD);

    float* resultado_soma=malloc((N_recebidos)*sizeof(float));
    float resultado_produto=0.0;

    for(int i=0; i<N_recebidos; i++) {
        resultado_soma[i]=buffer_destino_A[i]+buffer_destino_B[i];
        resultado_produto+=buffer_destino_A[i]*buffer_destino_B[i];
    }

    MPI_Gather(buffer_destino_A,N_recebidos,MPI_FLOAT,buffer_retorno_A,N_envios,MPI_FLOAT,cabeça,MPI_COMM_WORLD);
    MPI_Gather(buffer_destino_B,N_recebidos,MPI_FLOAT,buffer_retorno_B,N_envios,MPI_FLOAT,cabeça,MPI_COMM_WORLD);

    fim=MPI_Wtime();

    if (rank==cabeça) {
        printf("tempo decorrido:%lf\ncarga total:%d\nNúmero de processadores utilizados:%d\n",fim-inicio,carga,N_tarefas);
        free(buffer_origem_A);
        free(buffer_origem_B);
        free(buffer_retorno_A);
        free(buffer_retorno_B);
    }

    free(buffer_destino_A);
    free(buffer_destino_B);
    free(resultado_soma);

    MPI_Finalize();

    return 0;
}
