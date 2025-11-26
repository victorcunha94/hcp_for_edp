import numpy as np
from mpi4py import MPI
import time
import sys



def dot_product(A, B):
    """Calcula produto interno entre dois vetores"""
    result = 0.0
    print(len(A))
    for i in range(len(A)):
        result += A[i] * B[i]
    return result
 

def gerar_vetores(tamanho):
    """Gera vetores de teste com dados realistas"""
    np.random.seed(42)
    A = np.random.rand(tamanho).astype(np.float64) * 1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000
    return A, B




def run_serial(comm, A, B, tamanho):
    """Versão serial"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    inicio = time.time()
    resultado = dot_product(A, B)
    tempo = time.time() - inicio
    
        
    return resultado, tempo


def run_point_to_point(comm, A, B, tamanho):
    """Versão com comunicação ponto-a-ponto bloqueante"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        inicio = time.time()
        
        if size == 1:
            resultado = dot_product(A, B)
            tempo = time.time() - inicio
            return resultado, tempo
        
        # Divisão de trabalho
        num_workers = size - 1
        chunk_size = tamanho // num_workers
        remainder = tamanho % num_workers
        
        # Enviar tamanhos primeiro para evitar problemas de memória
        start = 0
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1
            
            chunk_len = end - start
            
            # Enviar tamanho primeiro
            comm.send(chunk_len, dest=i, tag=1)
            
            # Enviar dados
            A_chunk = A[start:end].copy()  # Usar copy() para evitar referências
            B_chunk = B[start:end].copy()
            
            comm.send(A_chunk, dest=i, tag=2)
            comm.send(B_chunk, dest=i, tag=3)
            
            start = end
        
        # Coletar resultados
        total_result = 0.0
        for i in range(1, size):
            partial = comm.recv(source=i, tag=4)
            total_result += partial
        
        tempo = time.time() - inicio
        return total_result, tempo
        
    else:
        # Workers - receber de forma mais segura
        try:
            chunk_len = comm.recv(source=0, tag=1)
            A_chunk = comm.recv(source=0, tag=2)
            B_chunk = comm.recv(source=0, tag=3)
            
            # Verificar integridade dos dados
            if len(A_chunk) != chunk_len or len(B_chunk) != chunk_len:
                raise ValueError("Tamanho dos dados inconsistente")
            
            result = dot_product(A_chunk, B_chunk)
            comm.send(result, dest=0, tag=4)
            return None, None
            
        except Exception as e:
            print(f"Worker {rank}: Erro - {e}")
            comm.send(0.0, dest=0, tag=4)  # Enviar 0 em caso de erro
            return None, None

def run_non_blocking(comm, A, B, tamanho):
    """Versão com comunicação não-bloqueante"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        inicio = time.time()
        
        if size == 1:
            resultado = dot_product(A, B)
            tempo = time.time() - inicio
            return resultado, tempo
        
        # Divisão de trabalho
        num_workers = size - 1
        chunk_size = tamanho // num_workers
        remainder = tamanho % num_workers
        
        send_requests = []
        start = 0
        
        # Primeiro enviar todos os tamanhos
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1
            
            chunk_len = end - start
            comm.send(chunk_len, dest=i, tag=10)  # Envio bloqueante para tamanhos
            start = end
        
        # Agora enviar dados não-bloqueante
        start = 0
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1
            
            A_chunk = A[start:end].copy()
            B_chunk = B[start:end].copy()
            
            req1 = comm.isend(A_chunk, dest=i, tag=11)
            req2 = comm.isend(B_chunk, dest=i, tag=12)
            send_requests.extend([req1, req2])
            
            start = end
        
        # Esperar envios completarem
        MPI.Request.Waitall(send_requests)
        
        # Recepção não-bloqueante
        recv_requests = []
        for i in range(1, size):
            req = comm.irecv(source=i, tag=13)
            recv_requests.append(req)
        
        # Coletar resultados
        total_result = 0.0
        for req in recv_requests:
            partial = req.wait()
            total_result += partial
        
        tempo = time.time() - inicio
        return total_result, tempo
        
    else:
        # Workers
        try:
            # Receber tamanho primeiro
            chunk_len = comm.recv(source=0, tag=10)
            
            # Receber dados
            A_chunk = comm.recv(source=0, tag=11)
            B_chunk = comm.recv(source=0, tag=12)
            
            # Verificar integridade
            if len(A_chunk) != chunk_len or len(B_chunk) != chunk_len:
                raise ValueError("Tamanho dos dados inconsistente")
            
            result = dot_product(A_chunk, B_chunk)
            comm.isend(result, dest=0, tag=13).wait()
            return None, None
            
        except Exception as e:
            print(f"Worker {rank}: Erro - {e}")
            comm.isend(0.0, dest=0, tag=13).wait()
            return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Configurações mais conservadoras
    tamanhos = [10000000, 50000000, 100000000]  # Começar com tamanhos menores
    repeticoes = 2  # Menos repetições inicialmente
    
    if rank == 0:
        print("=" * 70)
        print("SISTEMA DE COMPARAÇÃO MPI - VERSÃO SEGURA")
        print("=" * 70)
        print(f"Processos: {size}, Repetições: {repeticoes}")
        print("=" * 70)
    
    for tamanho in tamanhos:
        if rank == 0:
            print(f"\n--- TESTE: {tamanho:,} elementos ---")
            A, B = gerar_vetores(tamanho)
        else:
            A, B = None, None
        
        # Sincronizar
        comm.Barrier()
        
        # Testar Ponto-a-Ponto
        tempos_p2p = []
        tempos_nb = []
        tempos_sl = []
        for i in range(repeticoes):
            if rank == 0:
                A, B = gerar_vetores(tamanho)
            
            comm.Barrier()
            resultado_P2P, tempo_P2P = run_point_to_point(comm, A, B, tamanho)
            resultado_NB, tempo_NB = run_non_blocking(comm, A, B, tamanho)
            resultado_sl, tempo_sl = run_serial(comm, A, B, tamanho)
            
            if rank == 0:
                tempos_p2p.append(tempo_P2P)
                print(f"P2P - Exec {i+1}: {tempo_P2P:.4f}s")
                tempos_nb.append(tempo_NB)
                print(f"NB  - Exec {i+1}: {tempo_NB:.4f}s")
                tempos_sl.append(tempo_sl)
                print(f"NB  - Exec {i+1}: {tempo_sl:.4f}s")
        
        
        # Resultados
        if rank == 0 and tempos_p2p and tempos_nb:
            avg_p2p = np.mean(tempos_p2p)
            avg_nb = np.mean(tempos_nb)
            avg_serial = np.mean(tempos_sl)
            
            
            print(f"\nRESUMO {tamanho:,}:")
            print(f"P2P: {avg_p2p:.4f}s, NB: {avg_nb:.4f}s")
            print(f"Speedup_NB: {avg_serial/avg_nb:.2f}x")
            print(f"Speedup_P2P: {avg_serial/avg_p2p:.2f}x")
            #print(f"Speedup: {avg_p2p/avg_nb:.2f}x")
            print("-" * 40)

if __name__ == "__main__":
    main()
