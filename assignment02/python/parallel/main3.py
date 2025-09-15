import numpy as np
import matplotlib.pyplot as plt
from utils_tools import *
from pc_methods import *
from time import perf_counter
import threading
import os
import concurrent.futures
import colorsys
from numba import njit, config
import math

############ Configuração ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '1'  # Começar com 1 thread

# Parâmetros do problema - AUMENTADOS para ser mais compute-bound
tol = 1e-10  # Tolerância mais rigorosa
T = 8000  # 4x mais iterações - TORNA MAIS COMPUTE-BOUND
incle = 300  # Grid menor, mas mais iterações
xl, xr, yb, yt = -4, 0.5, -2.5, 2.5
total_points = (incle + 1) * (incle + 1)

# Criar pasta de output
os.makedirs('output', exist_ok=True)


# DESATIVAR CACHE e otimizações agressivas
@njit(parallel=False, cache=False, fastmath=False, inline='never')
def euler_implict_numba(Un, z):
    """Método de Euler implícito SEM otimizações agressivas"""
    one_zero = np.array([1.0, 0.0])
    den = complex_sub(one_zero, z)
    Un1 = complex_div(Un, den)
    return Un1


@njit(parallel=False, cache=False, fastmath=False, inline='never')
def preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, preditor_order, corretor_order, n_correcoes):
    """Preditor-Corretor SEM otimizações agressivas"""
    # Implementação manual sem otimizações
    history = [Un, Un1, Un2, Un3]

    valid_history = []
    for h in history:
        if not np.isnan(h[0]):
            valid_history.append(h)

    if len(valid_history) < 2:
        return Un3

    # PREDITOR: AB4 - Implementação manual lenta
    if preditor_order == 4 and len(valid_history) >= 4:
        u_nm3, u_nm2, u_nm1, u_n = valid_history

        # Cálculos manuais sem otimização
        f_n = complex_prod(z, u_n)
        f_nm1 = complex_prod(z, u_nm1)
        f_nm2 = complex_prod(z, u_nm2)
        f_nm3 = complex_prod(z, u_nm3)

        # Coeficientes calculados manualmente
        term1 = complex_scale(f_n, 55.0 / 24.0)
        term2 = complex_scale(f_nm1, -59.0 / 24.0)
        term3 = complex_scale(f_nm2, 37.0 / 24.0)
        term4 = complex_scale(f_nm3, -3.0 / 8.0)

        pred_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
        u_pred = complex_add(u_n, pred_sum)
    else:
        u_pred = Un3

    # CORRETOR: AM4 - Implementação manual
    if corretor_order == 4 and len(valid_history) >= 4:
        u_est = u_pred
        u_n, u_nm1, u_nm2 = valid_history[3], valid_history[2], valid_history[1]

        f_est = complex_prod(z, u_pred)

        for _ in range(n_correcoes):
            f_n = complex_prod(z, u_n)
            f_nm1 = complex_prod(z, u_nm1)
            f_nm2 = complex_prod(z, u_nm2)

            term1 = complex_scale(f_est, 9.0 / 24.0)
            term2 = complex_scale(f_n, 19.0 / 24.0)
            term3 = complex_scale(f_nm1, -5.0 / 24.0)
            term4 = complex_scale(f_nm2, 1.0 / 24.0)

            corr_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
            u_est = complex_add(u_n, corr_sum)

            f_est = complex_prod(z, u_est)

        return u_est

    return u_pred


# Adicionar operações compute-bound extras
@njit(cache=False, fastmath=False)
def additional_compute_bound_work(z, iterations=100):
    """Adiciona trabalho compute-bound extra"""
    result = z.copy()
    for i in range(iterations):
        # Operações matemáticas intensivas
        result = complex_add(result, complex_scale(z, math.sin(i * 0.1)))
        result = complex_prod(result, complex_scale(z, math.cos(i * 0.1)))
    return result


@njit(cache=False, fastmath=False)
def process_single_point_numba(h, k, xl, xr, yb, yt, incle, T, tol):
    """Processa um único ponto SEM otimizações agressivas"""
    real_z = xl + (h * np.abs(xr - xl) / incle)
    img_z = yb + (k * np.abs(yt - yb) / incle)
    z = np.array([real_z, img_z])

    # TRABALHO COMPUTE-BOUND EXTRA
    z = additional_compute_bound_work(z, 50)

    Un = np.array([1.0, 0.0])
    Un1 = euler_implict_numba(Un, z)
    Un2 = euler_implict_numba(Un1, z)
    Un3 = euler_implict_numba(Un2, z)

    is_stable = False
    is_unstable = False

    for n in range(T):
        Un4 = preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, 4, 4, 1)
        Un, Un1, Un2, Un3 = Un1, Un2, Un3, Un4

        norm = np.sqrt(Un4[0] ** 2 + Un4[1] ** 2)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1.0 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable


# Função principal - AGORA com warm-up e recompilação
def process_all_points_parallel(xl, xr, yb, yt, incle, T, tol, num_threads):
    """Processa todos os pontos forçando recompilação para cada número de threads"""

    # Forçar recompilação configurando environment
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    # Recriar a função com parallel=True dinamicamente
    @njit(parallel=True, cache=False, fastmath=False)
    def _process_all_points():
        total_points = (incle + 1) * (incle + 1)
        all_real_z = np.zeros(total_points)
        all_img_z = np.zeros(total_points)
        all_stable = np.zeros(total_points, dtype=np.bool_)
        all_unstable = np.zeros(total_points, dtype=np.bool_)

        for idx in prange(total_points):
            h = idx // (incle + 1)
            k = idx % (incle + 1)

            real_z, img_z, is_stable, is_unstable = process_single_point_numba(
                h, k, xl, xr, yb, yt, incle, T, tol
            )

            all_real_z[idx] = real_z
            all_img_z[idx] = img_z
            all_stable[idx] = is_stable
            all_unstable[idx] = is_unstable

        return all_real_z, all_img_z, all_stable, all_unstable

    return _process_all_points()


# Função principal modificada
def run_with_numba_prange(num_threads):
    """Executa com recompilação forçada para cada número de threads"""
    print(f"\n=== Testando com {num_threads} thread(s) ===")

    start_time_total = perf_counter()

    # Processar forçando recompilação
    real_coords, imag_coords, stable_flags, unstable_flags = \
        process_all_points_parallel(xl, xr, yb, yt, incle, T, tol, num_threads)

    end_time_total = perf_counter()
    total_execution_time = end_time_total - start_time_total

    print(f"Tempo com {num_threads} threads: {total_execution_time:.2f}s")
    print(f"Pontos estáveis: {np.sum(stable_flags)}, Instáveis: {np.sum(unstable_flags)}")

    return total_execution_time


# Execução principal com warm-up
if __name__ == "__main__":
    threads_to_test = [1, 2, 4, 8, 12, 16, 20]  # Testar até 20 threads

    print("=== WARM-UP (Compilação Inicial) ===")
    # Compilação inicial com problema pequeno
    os.environ['OMP_NUM_THREADS'] = '1'
    process_all_points_parallel(-4, 0.5, -2.5, 2.5, 10, 100, 1e-5, 1)
    print("Warm-up concluído!")

    execution_times = {}

    for n_threads in threads_to_test:
        time_taken = run_with_numba_prange(n_threads)
        execution_times[n_threads] = time_taken

    # Plotar resultados
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    threads_list = list(execution_times.keys())
    times_list = list(execution_times.values())
    plt.plot(threads_list, times_list, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Threads')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Tempo vs Threads (Compute-Bound)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    base_time = execution_times[1]
    speedup = [base_time / t for t in times_list]
    plt.plot(threads_list, speedup, 'ro-', linewidth=2, markersize=8)
    plt.plot(threads_list, threads_list, 'k--', alpha=0.5, label='Ideal')
    plt.xlabel('Número de Threads')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Threads')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('output/performance_comparison_detailed.png', dpi=300)
    plt.show()

    # Análise detalhada
    print("\n=== ANÁLISE DETALHADA ===")
    print(f"{'Threads':<8} {'Tempo(s)':<10} {'Speedup':<8} {'Eficiência':<10} {'Tempo/Thread':<12}")
    print("-" * 60)

    base_time = execution_times[1]
    for threads, time_val in execution_times.items():
        speedup_val = base_time / time_val
        efficiency = (speedup_val / threads) * 100
        time_per_thread = time_val / threads

        print(f"{threads:<8} {time_val:<10.2f} {speedup_val:<8.2f} {efficiency:<10.1f}% {time_per_thread:<12.4f}")