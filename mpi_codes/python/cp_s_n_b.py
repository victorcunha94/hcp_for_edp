import numpy as np
from mpi4py import MPI
import time
import sys

def dot_product(A, B):
    """Calcula produto interno entre dois vetores"""
    result = 0.0
    for i in range(len(A)):
        result += A[i] * B[i]
    return result


def gerar_vetores(tamanho):
    """Gera vetores de teste com dados realistas"""
    np.random.seed(42)
    A = np.random.rand(tamanho).astype(np.float64) * 1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000
    return A, B

def run_serial(A, B):
    """Versão serial - executada apenas no rank 0"""
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
        
        # Enviar tamanhos primeiro
        start = 0
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1
            
            chunk_len = end - start
            comm.send(chunk_len, dest=i, tag=1)
            A_chunk = A[start:end].copy()
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
        # Workers
        try:
            chunk_len = comm.recv(source=0, tag=1)
            A_chunk = comm.recv(source=0, tag=2)
            B_chunk = comm.recv(source=0, tag=3)
            
            result = dot_product(A_chunk, B_chunk)
            comm.send(result, dest=0, tag=4)
            return None, None
            
        except Exception as e:
            print(f"Worker {rank}: Erro - {e}")
            comm.send(0.0, dest=0, tag=4)
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
        
        # Enviar tamanhos primeiro (bloqueante)
        start = 0
        for i in range(1, size):
            end = start + chunk_size
            if (i - 1) < remainder:
                end += 1
            chunk_len = end - start
            comm.send(chunk_len, dest=i, tag=10)
            start = end
        
        # Enviar dados não-bloqueante
        send_requests = []
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
        
        MPI.Request.Waitall(send_requests)
        
        # Recepção não-bloqueante
        recv_requests = []
        for i in range(1, size):
            req = comm.irecv(source=i, tag=13)
            recv_requests.append(req)
        
        total_result = 0.0
        for req in recv_requests:
            partial = req.wait()
            total_result += partial
        
        tempo = time.time() - inicio
        return total_result, tempo
        
    else:
        # Workers
        try:
            chunk_len = comm.recv(source=0, tag=10)
            A_chunk = comm.recv(source=0, tag=11)
            B_chunk = comm.recv(source=0, tag=12)
            
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
    
    # Configurações - tamanhos realistas
    tamanhos = [1000000, 5000000, 10000000]  # 1M, 5M, 10M elementos
    repeticoes = 5
    
    if rank == 0:
        print("=" * 80)
        print("COMPARAÇÃO: SERIAL vs PONTO-A-PONTO vs NÃO-BLOQUEANTE")
        print("=" * 80)
        print(f"Processos: {size}, Repetições: {repeticoes}")
        print("=" * 80)
        
        # Cabeçalho da tabela
        print(f"{'Tamanho':<12} {'Serial (s)':<10} {'P2P (s)':<10} {'NB (s)':<10} {'Speedup P2P':<12} {'Speedup NB':<12} {'Efic. P2P':<10} {'Efic. NB':<10}")
        print("-" * 80)
    
    for tamanho in tamanhos:
        if rank == 0:
            A, B = gerar_vetores(tamanho)
        else:
            A, B = None, None
        
        comm.Barrier()
        
        # Executar testes
        tempos_serial, tempos_p2p, tempos_nb = [], [], []
        
        for i in range(repeticoes):
            if rank == 0:
                A, B = gerar_vetores(tamanho)  # Novos dados a cada repetição
            
            comm.Barrier()
            
            # Executar Serial (apenas rank 0)
            tempo_serial = 0.0
            if rank == 0:
                resultado_serial, tempo_serial = run_serial(A, B)
                tempos_serial.append(tempo_serial)
            
            # Executar P2P
            resultado_p2p, tempo_p2p = run_point_to_point(comm, A, B, tamanho)
            if rank == 0:
                tempos_p2p.append(tempo_p2p)
            
            # Executar Não-Bloqueante
            resultado_nb, tempo_nb = run_non_blocking(comm, A, B, tamanho)
            if rank == 0:
                tempos_nb.append(tempo_nb)
        
        # Calcular estatísticas no rank 0
        if rank == 0:
            avg_serial = np.mean(tempos_serial)
            avg_p2p = np.mean(tempos_p2p)
            avg_nb = np.mean(tempos_nb)
            
            # Calcular speedups
            speedup_p2p = avg_serial / avg_p2p
            speedup_nb = avg_serial / avg_nb
            
            # Calcular eficiências
            eficiencia_p2p = (speedup_p2p / (size)) * 100
            eficiencia_nb = (speedup_nb / (size)) * 100
            
            # Imprimir linha da tabela
            print(f"{tamanho:<12,} {avg_serial:<10.4f} {avg_p2p:<10.4f} {avg_nb:<10.4f} "
                  f"{speedup_p2p:<12.2f} {speedup_nb:<12.2f} "
                  f"{eficiencia_p2p:<10.1f}% {eficiencia_nb:<10.1f}%")
            
            # Verificar correção
            diff_p2p = abs(resultado_serial - resultado_p2p) / resultado_serial
            diff_nb = abs(resultado_serial - resultado_nb) / resultado_serial
            
            if diff_p2p > 1e-10 or diff_nb > 1e-10:
                print(f"  ⚠️  Diferença nos resultados: P2P={diff_p2p:.2e}, NB={diff_nb:.2e}")

if __name__ == "__main__":
    main()
