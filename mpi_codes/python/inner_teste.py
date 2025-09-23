import numpy as np
from mpi4py import MPI
import time

def dot_product(A, B):
    """Calcula produto interno entre dois vetores"""
    result = 0.0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result

def gerar_vetores(tamanho):
    """Gera vetores de teste com dados realistas"""
    np.random.seed(42)  # Para resultados reproduzíveis
    A = np.random.rand(tamanho).astype(np.float64) * 1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000
    return A, B

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Estrutura mais robusta para evitar conflitos
    if rank == 0:
        # ⏱️ Iniciar cronômetro APENAS no rank 0
        inicio_total = time.time()
        
        # Vetores de entrada - 1 milhão de elementos
        tamanho = 50000000
        A, B = gerar_vetores(tamanho)
        
        print(f"Rank 0: Vetor A - primeiros 3: {A[:3]} ... últimos 3: {A[-3:]}")
        print(f"Rank 0: Vetor B - primeiros 3: {B[:3]} ... últimos 3: {B[-3:]}")
        
        if size == 1:
            # Modo sequencial
            result = dot_product(A, B)
            fim_total = time.time()
            print(f"Rank 0: Produto interno: {result}")
            print(f"Rank 0: Tempo total: {fim_total - inicio_total:.4f}s")
            print(f"Rank 0: Tamanho: {tamanho}, Processos: {size}")
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

            comm.send(('dot_product', A_chunk, B_chunk), dest=i, tag=1)
            print(f"Rank 0: Enviou para worker {i}: {len(A_chunk)} elementos")
            start = end

        # Coletar resultados parciais
        total_result = 0.0
        for i in range(1, size):
            partial_result = comm.recv(source=i, tag=2)
            total_result += partial_result
            print(f"Rank 0: Recebeu do worker {i}: {partial_result}")
        
        fim_total = time.time()
            
        print(f"Rank 0: Produto interno final: {total_result}") 
        print(f"Rank 0: Tempo total: {fim_total - inicio_total:.4f}s")
        print(f"Rank 0: Tamanho: {tamanho}, Processos: {size}")
        
    else:
        # Workers - código completamente separado
        try:
            print(f"Worker {rank}: Aguardando dados...")
            
            # Receber operação e dados
            operation, A_chunk, B_chunk = comm.recv(source=0, tag=1)
            
            if operation == 'dot_product':
                # Medir tempo de cálculo do worker
                inicio_worker = time.time()
                result = dot_product(A_chunk, B_chunk)
                fim_worker = time.time()
                
                print(f"Worker {rank}: Calculou produto parcial = {result} em {fim_worker - inicio_worker:.4f}s")
                comm.send(result, dest=0, tag=2)
                print(f"Worker {rank}: Resultado enviado para mestre")
            else:
                print(f"Worker {rank}: Operação desconhecida")
                
        except Exception as e:
            print(f"Worker {rank}: Erro - {e}")

if __name__ == "__main__":
    main()
