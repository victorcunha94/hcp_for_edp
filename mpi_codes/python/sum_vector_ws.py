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

    # Lê argumento de modo (default = strong)
    mode = "strong"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    if mode not in ("strong", "weak"):
        if rank == 0:
            print("Uso: python3 sum_vector_mn.py [strong|weak]")
        sys.exit(0)

    repeticoes = 5
    base_size = 1_000_000  # tamanho base por processo para Weak Scaling

    if rank == 0:
        print("=" * 70)
        print(f"SOMA DE VETORES - SCATTERV/GATHERV vs SERIAL ({mode.upper()} SCALING)")
        print("=" * 70)
        print(f"Processos: {size}, Repetições: {repeticoes}")
        print("=" * 70)
        if mode == "strong":
            print(f"{'Tamanho':<12} {'Serial (s)':<12} {'MPI (s)':<12} "
                  f"{'Speedup':<12} {'Eficiência':<12} {'Status':<10}")
        else:  # weak
            print(f"{'Nprocs':<8} {'Tamanho':<12} {'MPI (s)':<12} "
                  f"{'Eficiência Weak':<16} {'Status':<10}")
        print("-" * 70)

    # Para Weak Scaling precisamos medir T(1) como baseline
    baseline_time = None
    if mode == "weak" and size == 1:
        # rank 0 roda serial p/ baseline
        A, B = gerar_vetores(base_size)
        resultado_serial, t_serial = run_serial(A, B)
        baseline_time = t_serial

    if mode == "strong":
        tamanhos = [1_000_000, 5_000_000, 10_000_000, 50_000_000]
    else:  # weak
        tamanhos = [base_size * size]  # só o tamanho proporcional ao nº de processos

    for tamanho in tamanhos:
        if rank == 0:
            A, B = gerar_vetores(tamanho)
        else:
            A, B = None, None

        comm.Barrier()
        tempos_serial, tempos_mpi = [], []
        resultados_corretos = True

        for _ in range(repeticoes):
            if rank == 0:
                A, B = gerar_vetores(tamanho)

            comm.Barrier()

            tempo_serial = 0.0
            if mode == "strong" and rank == 0:
                resultado_serial, tempo_serial = run_serial(A, B)
                tempos_serial.append(tempo_serial)

            resultado_mpi, tempo_mpi = run_scatter_gather(comm, A, B, tamanho)
            if rank == 0:
                tempos_mpi.append(tempo_mpi)
                correto, diff = verify_result(A, B, resultado_mpi)
                if not correto:
                    resultados_corretos = False
                    print(f"  ⚠️ Diferença detectada: {diff:.2e}")

        if rank == 0:
            avg_mpi = np.mean(tempos_mpi)

            if mode == "strong":
                avg_serial = np.mean(tempos_serial)
                speedup = avg_serial / avg_mpi if avg_mpi > 0 else 0
                eficiencia = (speedup / size) * 100 if speedup > 0 else 0
                status = "OK" if resultados_corretos else "ERRO"
                print(f"{tamanho:<12,} {avg_serial:<12.4f} {avg_mpi:<12.4f} "
                      f"{speedup:<12.2f} {eficiencia:<12.1f}% {status:<10}")

            else:  # weak scaling
                if baseline_time is None:
                    # broadcast T(1) para todos os ranks
                    baseline_time = comm.bcast(avg_mpi if size == 1 else None, root=0)

                eficiencia_weak = (baseline_time / avg_mpi) * 100 if avg_mpi > 0 else 0
                status = "OK" if resultados_corretos else "ERRO"
                print(f"{size:<8} {tamanho:<12,} {avg_mpi:<12.4f} "
                      f"{eficiencia_weak:<16.1f}% {status:<10}")

if __name__ == "__main__":
    main()

