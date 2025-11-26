import numpy as np
from mpi4py import MPI

def add(A, B):
    C = np.zeros(len(A))
    for i in range(len(A)):
        C[i] = A[i] + B[i]
    return C
    
    
def prod_int(A, B):
    C = np.zeros(len(A))
    for i in range(len(A)):
        C[i] = A[i] * B[i]
    S = 0
    for h in range(len(C)):
        S += C[h]
        
    return S

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        B = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        
        print(f"Vetor A: {A}")
        print(f"Vetor B: {B}")
        
        # Verificar se temos workers suficientes
        if size == 1:
            # Modo sequencial
            C_soma = add(A, B)
            print(f"Resultado final da soma: {C_soma}")
            return
        
        # Usar size-1 workers (processos 1 até size-1)
        num_workers = size - 1
        chunk_size = len(A) // num_workers
        remainder = len(A) % num_workers

        start = 0
        for i in range(1, size):  # i de 1 até size-1
            end = start + chunk_size
            if (i - 1) < remainder:  # i-1 porque i começa em 1
                end += 1

            A_chunk = A[start:end]
            B_chunk = B[start:end]

            comm.send(A_chunk, dest=i, tag=1)
            comm.send(B_chunk, dest=i, tag=2)

            print(f"Mestre enviou para worker {i}: A{start}:{end} = {A_chunk}")
            start = end

        # Coletar resultados
        C_soma = np.zeros(len(A))
        start = 0
        for i in range(1, size):
            C_chunk = comm.recv(source=i, tag=3)
            end = start + len(C_chunk)
            C_soma[start:end] = C_chunk
            start = end
            
        print(f"Resultado final da soma: {C_soma}") 

    else:
        # Processos workers (ranks 1, 2, 3, ...)
        if rank < size:  # Apenas processos válidos
            try:
                A_chunk = comm.recv(source=0, tag=1)
                B_chunk = comm.recv(source=0, tag=2)
                
                print(f"Worker {rank} recebeu: A={A_chunk}, B={B_chunk}")
                
                # Calcular soma local
                C_chunk = add(A_chunk, B_chunk)
                
                print(f"Worker {rank} calculou: {C_chunk}")
                
                # Enviar resultado de volta para o mestre
                comm.send(C_chunk, dest=0, tag=3)
            except Exception as e:
                print(f"Worker {rank} erro: {e}")

if __name__ == "__main__":
    main()



def main_pi():
	

