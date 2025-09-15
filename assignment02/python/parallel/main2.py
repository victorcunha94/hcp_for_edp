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

############ Configuração ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '8'

# Parâmetros do problema
tol = 1e-08
T = 2000
tipo = "PC-AB4-AM4"
incle = 200
xl, xr, yb, yt = -4, 0.5, -2.5, 2.5
total_points = (incle + 1) * (incle + 1)

# Criar pasta de output
os.makedirs('output', exist_ok=True)


# Implementação compatível com Numba do Euler implícito
@njit(cache=True)
def euler_implict_numba(Un, z):
    """Método de Euler implícito compatível com Numba"""
    one_zero = np.array([1.0, 0.0])
    den = complex_sub(one_zero, z)
    Un1 = complex_div(Un, den)
    return Un1


# Implementação correta do preditor-corretor
@njit(cache=True)
def preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, preditor_order, corretor_order, n_correcoes):
    """Preditor-Corretor AB-AM compatível com Numba seguindo a lógica exata"""
    # Montar histórico (Un é o mais antigo, Un3 é o mais recente)
    history = [Un, Un1, Un2, Un3]

    # Filtrar valores válidos
    valid_history = []
    for h in history:
        if not np.isnan(h[0]):
            valid_history.append(h)

    if len(valid_history) < 2:
        return Un3

    # --- PREDITOR: AB4 ---
    if preditor_order == 4:
        if len(valid_history) < 4:
            return Un3

        u_nm3 = valid_history[0]  # Mais antigo (Un)
        u_nm2 = valid_history[1]  # Un1
        u_nm1 = valid_history[2]  # Un2
        u_n = valid_history[3]  # Mais recente (Un3)

        # AB4: u_{n+1} = u_n + (55/24*f_n - 59/24*f_{n-1} + 37/24*f_{n-2} - 3/8*f_{n-3})
        f_n = complex_prod(z, u_n)
        f_nm1 = complex_prod(z, u_nm1)
        f_nm2 = complex_prod(z, u_nm2)
        f_nm3 = complex_prod(z, u_nm3)

        term1 = complex_scale(f_n, 55.0 / 24.0)
        term2 = complex_scale(f_nm1, -59.0 / 24.0)
        term3 = complex_scale(f_nm2, 37.0 / 24.0)
        term4 = complex_scale(f_nm3, -3.0 / 8.0)

        pred_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
        u_pred = complex_add(u_n, pred_sum)
    else:
        u_pred = Un3

    # --- CORRETOR: AM4 com n_correcoes iterações ---
    if corretor_order == 4:
        if len(valid_history) < 4:
            return u_pred

        u_est = u_pred
        u_n = valid_history[3]  # u_n (mais recente)
        u_nm1 = valid_history[2]  # u_{n-1}
        u_nm2 = valid_history[1]  # u_{n-2}

        # f_est inicial
        f_est = complex_prod(z, u_pred)

        for _ in range(n_correcoes):
            f_n = complex_prod(z, u_n)
            f_nm1 = complex_prod(z, u_nm1)
            f_nm2 = complex_prod(z, u_nm2)

            # AM4: u_est = u_n + (9/24*f_est + 19/24*f_n - 5/24*f_{n-1} + 1/24*f_{n-2})
            term1 = complex_scale(f_est, 9.0 / 24.0)
            term2 = complex_scale(f_n, 19.0 / 24.0)
            term3 = complex_scale(f_nm1, -5.0 / 24.0)
            term4 = complex_scale(f_nm2, 1.0 / 24.0)

            corr_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
            u_est = complex_add(u_n, corr_sum)

            # Re-avaliar f_est
            f_est = complex_prod(z, u_est)

        return u_est

    return u_pred


# Função para processar um ponto individual (sem timing)
@njit(cache=True)
def process_single_point_numba(h, k, xl, xr, yb, yt, incle, T, tol):
    """Processa um único ponto - compatível com Numba"""
    real_z = xl + (h * np.abs(xr - xl) / incle)
    img_z = yb + (k * np.abs(yt - yb) / incle)
    z = np.array([real_z, img_z])

    # Inicialização
    Un = np.array([1.0, 0.0])
    Un1 = euler_implict_numba(Un, z)
    Un2 = euler_implict_numba(Un1, z)
    Un3 = euler_implict_numba(Un2, z)

    is_stable = False
    is_unstable = False

    for n in range(T):
        Un4 = preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, 4, 4, 1)
        Un = Un1
        Un1 = Un2
        Un2 = Un3
        Un3 = Un4

        norm = np.sqrt(Un4[0] ** 2 + Un4[1] ** 2)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1.0 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable


# Função principal que processa TODOS os pontos em paralelo com prange
@njit(parallel=True, cache=True)
def process_all_points_parallel(xl, xr, yb, yt, incle, T, tol):
    """Processa todos os pontos em paralelo usando prange"""
    total_points = (incle + 1) * (incle + 1)

    # Arrays para resultados
    all_real_z = np.zeros(total_points)
    all_img_z = np.zeros(total_points)
    all_stable = np.zeros(total_points, dtype=np.bool_)
    all_unstable = np.zeros(total_points, dtype=np.bool_)

    # Paralelização REAL com prange - cada thread processa um subconjunto de pontos
    for idx in prange(total_points):
        # Calcular coordenadas (h, k) from linear index
        h = idx // (incle + 1)
        k = idx % (incle + 1)

        # Processar ponto
        real_z, img_z, is_stable, is_unstable = process_single_point_numba(
            h, k, xl, xr, yb, yt, incle, T, tol
        )

        # Armazenar resultados
        all_real_z[idx] = real_z
        all_img_z[idx] = img_z
        all_stable[idx] = is_stable
        all_unstable[idx] = is_unstable

    return all_real_z, all_img_z, all_stable, all_unstable


# Função para gerar cores das threads
def generate_thread_colors(n_threads):
    colors = []
    for i in range(n_threads):
        hue = i / n_threads
        saturation = 0.8
        lightness = 0.5 + (i % 3) * 0.15
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append((r, g, b))
    return colors


# Função para salvar estatísticas
def save_statistics(execution_time, total_points, n_threads, tipo):
    """Salva estatísticas simplificadas"""
    with open(f'output/{tipo}_{n_threads}threads_stats.txt', 'w') as f:
        f.write(f"ESTATÍSTICAS DE PARALELIZAÇÃO - {n_threads} THREADS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total de pontos: {total_points}\n")
        f.write(f"Tempo total de execução: {execution_time:.2f}s\n")
        f.write(f"Tempo médio por ponto: {execution_time / total_points * 1000:.4f}ms\n")
        f.write(f"Threads utilizadas: {n_threads}\n")
        f.write(f"Método: {tipo}\n")
        f.write(f"Tolerância: {tol}\n")
        f.write(f"Iterações máximas: {T}\n")


# Função para visualizar resultados
def visualize_results(real_coords, imag_coords, stable_flags, n_threads, execution_time, tipo):
    """Visualiza resultados sem informações de timing por thread"""

    # 1. Região de Estabilidade
    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    plt.xlim(xl, xr)
    plt.ylim(yb, yt)

    plt.scatter(real_coords[stable_flags], imag_coords[stable_flags],
                color='blue', s=1, alpha=0.7, label='Estável')
    plt.scatter(real_coords[~stable_flags], imag_coords[~stable_flags],
                color='red', s=1, alpha=0.7, label='Instável')

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'Região de Estabilidade - {tipo} - {n_threads} threads\nTempo: {execution_time:.2f}s')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'output/{tipo}_{n_threads}threads.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Ordem de Processamento (simulada)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(real_coords, imag_coords,
                          c=np.arange(len(real_coords)), cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(scatter, label='Ordem de Processamento')
    plt.title(f'Ordem de Processamento - {tipo} - {n_threads} threads')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'output/processing_order_{tipo}_{n_threads}threads.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Salvar estatísticas
    save_statistics(execution_time, len(real_coords), n_threads, tipo)


# Função principal com paralelização Numba
def run_with_numba_prange(num_threads):
    """Executa o processamento com paralelização Numba + prange"""
    print(f"\n=== Testando com {num_threads} thread(s) ===")
    print("Usando paralelização Numba + prange (ALTAMENTE EFICIENTE)")

    # Configurar número de threads para Numba
    os.environ['OMP_NUM_THREADS'] = str(num_threads)

    start_time_total = perf_counter()

    # Processar TODOS os pontos em paralelo (chamada única)
    real_coords, imag_coords, stable_flags, unstable_flags = \
        process_all_points_parallel(xl, xr, yb, yt, incle, T, tol)

    end_time_total = perf_counter()
    total_execution_time = end_time_total - start_time_total

    # Gerar visualizações
    visualize_results(real_coords, imag_coords, stable_flags,
                      num_threads, total_execution_time, tipo)

    print(f"Tempo com {num_threads} threads: {total_execution_time:.2f}s")
    print(f"Pontos estáveis: {np.sum(stable_flags)}")
    print(f"Pontos instáveis: {np.sum(unstable_flags)}")

    return total_execution_time


# Execução principal
if __name__ == "__main__":
    threads_to_test = [1, 2, 3, 4, 5, 6, 7, 8]
    execution_times = {}

    print("Iniciando processamento da região de estabilidade...")
    print(f"Total de pontos: {total_points}")
    print(f"Tipo de método: {tipo}")
    print("USANDO PARALELIZAÇÃO NUMBA + PRANGE (MÁXIMA EFICIÊNCIA)")

    # Executar para diferentes números de threads
    for n_threads in threads_to_test:
        try:
            time_taken = run_with_numba_prange(n_threads)
            execution_times[n_threads] = time_taken
        except Exception as e:
            print(f"Erro com {n_threads} threads: {e}")

    # Plotar comparação de performance
    if len(execution_times) > 1:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        threads_list = list(execution_times.keys())
        times_list = list(execution_times.values())
        plt.plot(threads_list, times_list, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Número de Threads')
        plt.ylabel('Tempo de Execução (s)')
        plt.title('Tempo vs Threads')
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
        plt.savefig('output/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # Mostrar resumo final
        print("\n=== RESUMO FINAL ===")
        print(f"{'Threads':<8} {'Tempo (s)':<10} {'Speedup':<8} {'Eficiência':<10}")
        print("-" * 40)

        base_time = execution_times[1]
        for threads, time_taken in execution_times.items():
            speedup_val = base_time / time_taken
            efficiency = (speedup_val / threads) * 100
            print(f"{threads:<8} {time_taken:<10.2f} {speedup_val:<8.2f} {efficiency:<10.1f}%")