import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import *
from bdf_methods import *
from rk_methods import *
from adams_bashforth_moulton_methods import *
from pc_methods import *
import time
from numba import njit, prange, config, set_num_threads
from time import perf_counter
import threading
import os
import concurrent.futures

############ Paralelização ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '4'  # Define número de threads

tol = 1e-08
T = 2000
tipo = "PC-AB4-AM4"
incle = 200

xl = -4
xr = 0.5
yb = -2.5
yt = 2.5

total_points = (incle + 1) * (incle + 1)


# Função para processar um ponto individual (modificada para retornar tipo de ponto)
@njit(cache=True)
def process_single_point(h, k, tipo, xl, xr, yb, yt, incle, T, tol):
    real_z = xl + (h * (np.abs(xr - xl) / incle))
    img_z = yb + (k * (np.abs(yt - yb) / incle))
    z = np.array([real_z, img_z])

    # Inicialização
    Un = np.array([1, 0])
    Un1 = euler_implict(Un, z)
    Un2 = euler_implict(Un1, z)
    Un3 = euler_implict(Un2, z)

    is_stable = False
    is_unstable = False

    for n in range(T):
        Un4 = preditor_corrector_AB_AM(Un, Un1, Un2, Un3, z=z,
                                       preditor_order=4, corretor_order=4, n_correcoes=1)
        Un = Un1
        Un1 = Un2
        Un2 = Un3
        Un3 = Un4

        norm = linalg.norm(Un4, 2)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable


# Função wrapper para paralelização
def process_point_wrapper(args):
    h, k, tipo, xl, xr, yb, yt, incle, T, tol = args
    return process_single_point(h, k, tipo, xl, xr, yb, yt, incle, T, tol)


def run_with_threads(num_threads):
    """Executa o processamento com número específico de threads"""
    set_num_threads(num_threads)
    print(f"\n=== Testando com {num_threads} thread(s) ===")

    # Configuração do plot
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    ax.set_xlim(xl, xr)
    ax.set_ylim(yb, yt)
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'Região de Estabilidade - {tipo} - {num_threads} threads')

    start_time_total = perf_counter()

    # Preparar todos os pontos para processamento
    points_to_process = []
    for h in range(incle + 1):
        for k in range(incle + 1):
            points_to_process.append((h, k, tipo, xl, xr, yb, yt, incle, T, tol))

    # Processamento paralelo
    stable_points = []
    unstable_points = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_point_wrapper, points_to_process))

    # Processar resultados
    for real_z, img_z, is_stable, is_unstable in results:
        if is_stable:
            stable_points.append((real_z, img_z))
        elif is_unstable:
            unstable_points.append((real_z, img_z))

    # Plotar resultados
    if stable_points:
        stable_x, stable_y = zip(*stable_points)
        plt.plot(stable_x, stable_y, 'bo', markersize=0.5, label='Estável')

    if unstable_points:
        unstable_x, unstable_y = zip(*unstable_points)
        plt.plot(unstable_x, unstable_y, 'go', markersize=0.5, label='Instável')

    end_time_total = perf_counter()
    total_execution_time = end_time_total - start_time_total

    # Salvar informações
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{tipo}_{num_threads}threads.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvar informações de tempo
    with open(f'{tipo}_{num_threads}threads_time_info.txt', 'w') as f:
        f.write(f"Método: {tipo}\n")
        f.write(f"Threads: {num_threads}\n")
        f.write(f"Tempo total: {total_execution_time:.2f} segundos\n")
        f.write(f"Tempo total: {total_execution_time / 60:.2f} minutos\n")
        f.write(f"Total de pontos: {total_points}\n")
        f.write(f"Pontos estáveis: {len(stable_points)}\n")
        f.write(f"Pontos instáveis: {len(unstable_points)}\n")
        f.write(f"Tempo médio por ponto: {total_execution_time / total_points * 1000:.4f} ms\n")
        f.write(f"Parâmetros: T={T}, incle={incle}, tol={tol}\n")

    print(f"Tempo com {num_threads} threads: {total_execution_time:.2f}s")
    print(f"Speedup: {base_time / total_execution_time:.2f}x" if num_threads > 1 else "Referência")

    return total_execution_time


# Executar para diferentes números de threads
threads_to_test = [1, 2, 4, 8]
execution_times = {}

# Primeiro executa com 1 thread como referência
base_time = run_with_threads(1)
execution_times[1] = base_time

# Depois executa com mais threads
for threads in threads_to_test[1:]:
    try:
        time_taken = run_with_threads(threads)
        execution_times[threads] = time_taken
    except Exception as e:
        print(f"Erro com {threads} threads: {e}")

# Plotar comparação de performance
plt.figure(figsize=(10, 6))
threads_list = list(execution_times.keys())
times_list = list(execution_times.values())
speedup = [base_time / t for t in times_list]

plt.subplot(1, 2, 1)
plt.plot(threads_list, times_list, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Threads')
plt.ylabel('Tempo de Execução (s)')
plt.title('Tempo vs Threads')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(threads_list, speedup, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Número de Threads')
plt.ylabel('Speedup')
plt.title('Speedup vs Threads')
plt.grid(True)

plt.tight_layout()
plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Mostrar resumo final
print("\n=== RESUMO FINAL ===")
for threads, time_taken in execution_times.items():
    speedup = base_time / time_taken
    efficiency = (speedup / threads) * 100
    print(
        f"Threads: {threads:2d} | Tempo: {time_taken:6.2f}s | Speedup: {speedup:5.2f}x | Eficiência: {efficiency:5.1f}%")