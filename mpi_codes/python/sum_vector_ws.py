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
    np.random.seed(42)
    A = np.random.rand(tamanho).astype(np.float64) * 1000
    B = np.random.rand(tamanho).astype(np.float64) * 1000
    return A, B

def run_scatter_gather(comm, A, B, tamanho):
    rank = comm.Get_rank()
    size = comm.Get_size()
    inicio = time.time()

    if size == 1:
        resultado = vector_sum(A, B)
        return resultado, time.time() - inicio

    # Divisão dos dados
    chunk_size = tamanho // size
    remainder = tamanho % size
    displacements, counts = [], []
    start = 0
    for i in range(size):
        count = chunk_size + (1 if i < remainder else 0)
        displacements.append(start)
        counts.append(count)
        start += count

    # Buffers locais
    local_count = counts[rank]
    local_A = np.zeros(local_count, dtype=np.float64)
    local_B = np.zeros(local_count, dtype=np.float64)

    comm.Scatterv([A, counts, displacements, MPI.DOUBLE], local_A, root=0)
    comm.Scatterv([B, counts, displacements, MPI.DOUBLE], local_B, root=0)

    local_result = vector_sum(local_A, local_B)

    if rank == 0:
        global_result = np.zeros(tamanho, dtype=np.float64)
    else:
        global_result = None

    comm.Gatherv(local_result, [global_result, counts, displacements, MPI.DOUBLE], root=0)

    tempo = time.time() - inicio
    if rank == 0:
        return global_result, tempo
    else:
        return None, tempo




def run_experiment(tamanho, repeticoes):
    """Executa o cálculo de soma paralela com Scatterv/Gatherv"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divisão do vetor
    chunk_size = tamanho // size
    remainder = tamanho % size
    counts = [chunk_size + 1 if i < remainder else chunk_size for i in range(size)]
    displs = np.cumsum([0] + counts[:-1])

    # Vetor no rank 0
    if rank == 0:
        A = np.ones(tamanho, dtype=np.float64)
        B = np.ones(tamanho, dtype=np.float64)
    else:
        A = None
        B = None

    # Buffers locais
    local_A = np.zeros(counts[rank], dtype=np.float64)
    local_B = np.zeros(counts[rank], dtype=np.float64)

    # Scatterv
    comm.Scatterv([A, counts, displs, MPI.DOUBLE], local_A, root=0)
    comm.Scatterv([B, counts, displs, MPI.DOUBLE], local_B, root=0)

    # Cálculo local
    local_C = local_A + local_B

    # Gatherv
    if rank == 0:
        C = np.empty(tamanho, dtype=np.float64)
    else:
        C = None
    comm.Gatherv(local_C, [C, counts, displs, MPI.DOUBLE], root=0)

    return C
    
    
    
def run_serial(A, B):
    inicio = time.time()
    resultado = vector_sum(A, B)
    tempo = time.time() - inicio
    return resultado, tempo

def verify_result(A, B, result):
    expected = A + B
    diff = np.max(np.abs(result - expected))
    return diff < 1e-10, diff

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if len(sys.argv) < 2 or sys.argv[1] != "weak":
        if rank == 0:
            print("Uso: python3 sum_vector_ws.py weak")
        return

    repeticoes = 5
    base_size = 10**6   # tamanho base para 1 processo
    tamanho = base_size * size

    # =============================
    # Rodar baseline com np=1
    # =============================
    if rank == 0:
        # Simula baseline (np=1) localmente
        A = np.ones(base_size, dtype=np.float64)
        B = np.ones(base_size, dtype=np.float64)
        tempos = []
        for _ in range(repeticoes):
            t0 = time.time()
            C = A + B
            tempos.append(time.time() - t0)
        baseline_time = np.mean(tempos)

        print("="*70)
        print("SOMA DE VETORES - SCATTERV/GATHERV vs SERIAL (WEAK SCALING)")
        print("="*70)
        print(f"Processos: {size}, Repetições: {repeticoes}")
        print("="*70)
        print(f"{'Nprocs':<8}{'Tamanho':<12}{'MPI (s)':<12}{'Eficiência Weak':<18}{'Status':<10}")
        print("-"*70)
    else:
        baseline_time = None

    # Broadcast para todos conhecerem baseline_time
    baseline_time = comm.bcast(baseline_time, root=0)

    # =============================
    # Executar experimento paralelo
    # =============================
    tempos = []
    for _ in range(repeticoes):
        t0 = time.time()
        C = run_experiment(tamanho, repeticoes)
        tf = time.time() - t0
        tempos.append(tf)

    avg_mpi = np.mean(tempos)
    status = "OK" if C is not None and np.allclose(C, 2.0) else "ERRO"
    eficiencia_weak = (baseline_time / avg_mpi) * 100 if avg_mpi > 0 else 0

    if rank == 0:
        print(f"{size:<8}{tamanho:<12}{avg_mpi:<12.6f}{eficiencia_weak:<18.2f}{status:<10}")


if __name__ == "__main__":
    main()
