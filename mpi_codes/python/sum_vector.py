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

def run_scatter_gather(comm, A, B, tamanho):
    """Versão usando Scatterv e Gatherv - Mais eficiente"""
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    inicio = time.time()
    
    if size == 1:
        # Caso sequencial
        resultado = vector_sum(A, B)
        return resultado, time.time() - inicio
    
    # Calcular como dividir os dados
    chunk_size = tamanho // size
    remainder = tamanho % size
    
    # Calcular deslocamentos e tamanhos para cada processo
    displacements = []
    counts = []
    
    start = 0
    for i in range(size):
        count = chunk_size + (1 if i < remainder else 0)
        displacements.append(start)
        counts.append(count)
        start += count
    
    # Preparar buffers de recepção para cada processo
    local_count = counts[rank]
    local_A = np.zeros(local_count, dtype=np.float64)
    local_B = np.zeros(local_count, dtype=np.float64)
    
    # Scatter dos dados
    comm.Scatterv([A, counts, displacements, MPI.DOUBLE], local_A, root=0)
    comm.Scatterv([B, counts, displacements, MPI.DOUBLE], local_B, root=0)
    
    # Cada processo calcula sua parte local
    local_result = vector_sum(local_A, local_B)
    
    # Preparar buffer para Gather
    if rank == 0:
        global_result = np.zeros(tamanho, dtype=np.float64)
    else:
        global_result = None
    
    # Gather dos resultados
    comm.Gatherv(local_result, [global_result, counts, displacements, MPI.DOUBLE], root=0)
    
    tempo = time.time() - inicio
    
    if rank == 0:
        return global_result, tempo
    else:
        return None, tempo

def run_serial(A, B):
    """Versão serial"""
    inicio = time.time()
    resultado = vector_sum(A, B)
    tempo = time.time() - inicio
    return resultado, tempo

def verify_result(A, B, result):
    """Verifica se o resultado está correto"""
    expected = A + B
    diff = np.max(np.abs(result - expected))
    return diff < 1e-10, diff

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Tamanhos para teste
    tamanhos = [1000000, 5000000, 10000000, 50000000]  # Até 5M elementos
    repeticoes = 5
    
    if rank == 0:
        print("=" * 70)
        print("SOMA DE VETORES - SCATTERV/GATHERV vs SERIAL")
        print("=" * 70)
        print(f"Processos: {size}, Repetições: {repeticoes}")
        print("=" * 70)
        print(f"{'Tamanho':<10} {'Serial (s)':<10} {'MPI (s)':<10} {'Speedup':<12} {'Eficiência':<12} {'Status':<10}")
        print("-" * 70)
    
    for tamanho in tamanhos:
        if rank == 0:
            A, B = gerar_vetores(tamanho)
        else:
            A, B = None, None
        
        comm.Barrier()
        
        tempos_serial, tempos_mpi = [], []
        resultados_corretos = True
        
        for i in range(repeticoes):
            if rank == 0:
                A, B = gerar_vetores(tamanho)
            
            comm.Barrier()
            
            # Versão Serial (apenas rank 0)
            tempo_serial = 0.0
            if rank == 0:
                resultado_serial, tempo_serial = run_serial(A, B)
                tempos_serial.append(tempo_serial)
            
            # Versão MPI com Scatter/Gather
            resultado_mpi, tempo_mpi = run_scatter_gather(comm, A, B, tamanho)
            
            if rank == 0:
                tempos_mpi.append(tempo_mpi)
                
                # Verificar correção
                correto, diff = verify_result(A, B, resultado_mpi)
                if not correto:
                    resultados_corretos = False
                    print(f"  ⚠️  Diferença detectada: {diff:.2e}")
        
        # Calcular estatísticas no rank 0
        if rank == 0:
            avg_serial = np.mean(tempos_serial)
            avg_mpi = np.mean(tempos_mpi)
            
            speedup = avg_serial / avg_mpi if avg_mpi > 0 else 0
            eficiencia = (speedup / size) * 100 if speedup > 0 else 0
            
            status = "OK" if resultados_corretos else "ERRO"
            
            print(f"{tamanho:<10,} {avg_serial:<10.4f} {avg_mpi:<10.4f} "
                  f"{speedup:<12.2f} {eficiencia:<12.1f}% {status:<10}")

if __name__ == "__main__":
    main()
