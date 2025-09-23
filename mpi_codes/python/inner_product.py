import numpy as np
from mpi4py import MPI
import random as rd
import time



def dot_product(A, B):
    """Calcula produto interno entre dois vetores"""
    result = 0.0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result
    
  
    

def gerar_vetores(tamanho):
    """Gera vetores de teste com dados realistas"""
    # Vetores com padrão mais realista que 1,2,3...
    A = np.random.rand(tamanho).astype(np.float64) * 1000  # 0-1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000  # 0-1000
    return A, B



def gerar_vetores_verificavel(tamanho):
    """Gera vetores onde podemos verificar o resultado"""
    A = np.arange(1, tamanho + 1, dtype=np.float64)
    B = np.arange(tamanho, 0, -1, dtype=np.float64) * 10  # Padrão reverso
    return A, B

    

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    

    inicio_total = None
    fim_total    = None


    #inicio_total = time.time()

    if rank == 0:
        # Vetores de entrada
        tamanho = 1000000  # Começar com 1 milhão
        A, B = gerar_vetores(tamanho)
        
        inicio_total = time.time()
        
        print(f"Vetor A: {A}")
        print(f"Vetor B: {B}")
        
        if size == 1:
            # Modo sequencial39
            result = dot_product(A, B)
            fim_total = time.time()
            print(f"Produto interno: {result}")
            print(f"Tempo total: {fim_total - inicio_total:.4f}s")
            print(f"Tamanho: {tamanho}, Processos: {size}")

            return
        
        # Dividir trabalho
        num_workers = size - 1
        chunk_size = tamanho // num_workers
        remainder = tamanho % num_workers

        start = 0
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1

            A_chunk = A[start:end]
            B_chunk = B[start:end]

            # Enviar dados para worker
            comm.send(('dot_product', A_chunk, B_chunk), dest=i, tag=1)
            print(f"Mestre enviou para worker {i}: {len(A_chunk)} elementos")
            start = end

        # Coletar resultados parciais
        total_result = 0.0
        for i in range(1, size):
            partial_result = comm.recv(source=i, tag=2)
            total_result += partial_result
            print(f"Mestre recebeu do worker {i}: {partial_result}")
        
        fim_total = time.time()

        print(f"Produto interno final: {total_result}") 

    else:
        # Workers
        print(f"Worker {rank}: Aguardando dados...")
        
        # Receber operação e dados
        operation, A_chunk, B_chunk = comm.recv(source=0, tag=1)
        
        if operation == 'dot_product':
            result = dot_product(A_chunk, B_chunk)
            print(f"Worker {rank}: calculou produto parcial = {result}")
            comm.send(result, dest=0, tag=2)
        else:
            print(f"Worker {rank}: Operação desconhecida")
            
            
    fim_total = time.time()
    print(f"Tempo total: {fim_total - inicio_total:.4f}s")
    print(f"Tamanho: {tamanho}, Processos: {size}")

if __name__ == "__main__":
    main()
