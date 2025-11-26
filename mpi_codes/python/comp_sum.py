import numpy as np
from mpi4py import MPI
import time
import sys

def vector_sum(A, B):
    C = np.zeros(len(A))
    for i in range(len(A)):
        C[i] = A[i] + B[i]
    return C

def gerar_vetores(tamanho):
    """Gera vetores de teste"""
    np.random.seed(42)
    A = np.random.rand(tamanho).astype(np.float64) * 1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000
    return A, B

def run_serial(A, B):
    """Versão serial"""
    inicio = time.time()
    resultado = vector_sum(A, B)
    tempo = time.time() - inicio
    return resultado, tempo

def run_parallel_simple(comm, A, B, tamanho, mode='p2p'):
    """Versão paralela simplificada"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        inicio = time.time()
        
        if size == 1:
            resultado = vector_sum(A, B)
            return resultado, time.time() - inicio
        
        # Divisão simples: rank 0 faz metade, outros processos fazem o resto
        metade = tamanho // 2
        resto = tamanho - metade
        
        if mode == 'p2p':
            # Ponto-a-ponto bloqueante
            comm.send(resto, dest=1, tag=1)
            comm.send(A[metade:].copy(), dest=1, tag=2)
            comm.send(B[metade:].copy(), dest=1, tag=3)
            
            # Rank 0 processa sua parte
            parte_rank0 = vector_sum(A[:metade], B[:metade])
            
            # Receber do worker
            parte_worker = comm.recv(source=1, tag=4)
            
            # Combinar resultados
            resultado = np.concatenate([parte_rank0, parte_worker])
            
        else:  # Não-bloqueante
            # Envio não-bloqueante
            req1 = comm.isend(resto, dest=1, tag=10)
            req2 = comm.isend(A[metade:].copy(), dest=1, tag=11)
            req3 = comm.isend(B[metade:].copy(), dest=1, tag=12)
            
            # Rank 0 processa enquanto envia
            parte_rank0 = vector_sum(A[:metade], B[:metade])
            
            # Esperar envios completarem
            req1.wait()
            req2.wait()
            req3.wait()
            
            # Receber resultado
            parte_worker = comm.recv(source=1, tag=13)
            resultado = np.concatenate([parte_rank0, parte_worker])
        
        tempo = time.time() - inicio
        return resultado, tempo
        
    else:  # Worker (rank 1)
        if mode == 'p2p':
            tamanho_worker = comm.recv(source=0, tag=1)
            A_worker = comm.recv(source=0, tag=2)
            B_worker = comm.recv(source=0, tag=3)
            
            resultado_worker = vector_sum(A_worker, B_worker)
            comm.send(resultado_worker, dest=0, tag=4)
            
        else:  # Não-bloqueante
            tamanho_worker = comm.recv(source=0, tag=10)
            A_worker = comm.recv(source=0, tag=11)
            B_worker = comm.recv(source=0, tag=12)
            
            resultado_worker = vector_sum(A_worker, B_worker)
            comm.send(resultado_worker, dest=0, tag=13)
        
        return None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Para este teste, vamos usar apenas 2 processos
    if size != 2:
        if rank == 0:
            print("Este teste requer exatamente 2 processos. Use: mpirun -np 2 python3 comp_sum.py")
        return
    
    # Tamanhos menores para teste inicial
    tamanhos = [100000, 500000, 1000000]  # 100K, 500K, 1M elementos
    repeticoes = 2
    
    if rank == 0:
        print("=" * 80)
        print("COMPARAÇÃO: SOMA DE VETORES (2 PROCESSOS)")
        print("=" * 80)
        print(f"Repetições: {repeticoes}")
        print("=" * 80)
        print(f"{'Tamanho':<10} {'Serial (s)':<10} {'P2P (s)':<10} {'NB (s)':<10} {'Speedup P2P':<12} {'Speedup NB':<12}")
        print("-" * 80)
    
    for tamanho in tamanhos:
        if rank == 0:
            print(f"Processando {tamanho:,} elementos...")
            A, B = gerar_vetores(tamanho)
        else:
            A, B = None, None
        
        comm.Barrier()
        
        tempos_serial, tempos_p2p, tempos_nb = [], [], []
        
        for i in range(repeticoes):
            if rank == 0:
                A, B = gerar_vetores(tamanho)
            
            comm.Barrier()
            
            # Serial (apenas rank 0)
            if rank == 0:
                resultado_serial, tempo_serial = run_serial(A, B)
                tempos_serial.append(tempo_serial)
            
            # Ponto-a-ponto
            resultado_p2p, tempo_p2p = run_parallel_simple(comm, A, B, tamanho, 'p2p')
            if rank == 0:
                tempos_p2p.append(tempo_p2p)
            
            # Não-bloqueante
            resultado_nb, tempo_nb = run_parallel_simple(comm, A, B, tamanho, 'nb')
            if rank == 0:
                tempos_nb.append(tempo_nb)
            
            # Pequena pausa entre execuções
            time.sleep(0.1)
        
        # Resultados no rank 0
        if rank == 0:
            avg_serial = np.mean(tempos_serial)
            avg_p2p = np.mean(tempos_p2p)
            avg_nb = np.mean(tempos_nb)
            
            speedup_p2p = avg_serial / avg_p2p if avg_p2p > 0 else 0
            speedup_nb = avg_serial / avg_nb if avg_nb > 0 else 0
            
            print(f"{tamanho:<10,} {avg_serial:<10.4f} {avg_p2p:<10.4f} {avg_nb:<10.4f} "
                  f"{speedup_p2p:<12.2f} {speedup_nb:<12.2f}")

if __name__ == "__main__":
    main()
