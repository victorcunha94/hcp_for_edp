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
os.environ['OMP_NUM_THREADS'] = '4'

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
    # Para o problema y' = z*y, a solução implícita é: y_{n+1} = y_n / (1 - z)
    # Como z é complexo [real, imag], precisamos fazer a divisão complexa
    one_zero = np.array([1.0, 0.0])

    # den = 1 - z
    den = complex_sub(one_zero, z)

    # Un1 = Un / den
    Un1 = complex_div(Un, den)

    return Un1


# Implementação correta do preditor-corretor seguindo sua lógica exata
@njit(cache=True)
def preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, preditor_order, corretor_order, n_correcoes):
    """
    Preditor-Corretor AB-AM compatível com Numba seguindo a lógica exata da função original
    """
    # Montar histórico (Un é o mais antigo, Un3 é o mais recente)
    history = [Un, Un1, Un2, Un3]

    # Filtrar valores válidos (não NaN)
    valid_history = []
    for h in history:
        if not np.isnan(h[0]):
            valid_history.append(h)

    if len(valid_history) < 2:
        return Un3  # Fallback

    # --- PREDITOR: AB4 ---
    if preditor_order == 4:
        if len(valid_history) < 4:
            return Un3  # Não há histórico suficiente

        # Ordem: valid_history[0] = mais antigo, valid_history[-1] = mais recente
        u_nm3 = valid_history[0]  # Mais antigo (Un)
        u_nm2 = valid_history[1]  # Un1
        u_nm1 = valid_history[2]  # Un2
        u_n = valid_history[3]  # Mais recente (Un3)

        # AB4: u_{n+1} = u_n + (55/24*f_n - 59/24*f_{n-1} + 37/24*f_{n-2} - 3/8*f_{n-3})
        f_n = complex_prod(z, u_n)
        f_nm1 = complex_prod(z, u_nm1)
        f_nm2 = complex_prod(z, u_nm2)
        f_nm3 = complex_prod(z, u_nm3)

        # Calcular termos do preditor AB4
        term1 = complex_scale(f_n, 55.0 / 24.0)
        term2 = complex_scale(f_nm1, -59.0 / 24.0)
        term3 = complex_scale(f_nm2, 37.0 / 24.0)
        term4 = complex_scale(f_nm3, -3.0 / 8.0)  # -9/24 = -3/8

        # Combinar termos
        pred_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
        u_pred = complex_add(u_n, pred_sum)
    else:
        u_pred = Un3  # Fallback para outras ordens

    # --- ESTIMATIVA INICIAL: f_pred = z * u_pred ---
    f_est = complex_prod(z, u_pred)

    # --- CORRETOR: AM4 com n_correcoes iterações ---
    if corretor_order == 4:
        if len(valid_history) < 4:
            return u_pred  # Não há histórico suficiente para AM4

        u_est = u_pred
        u_n = valid_history[3]  # u_n (mais recente)
        u_nm1 = valid_history[2]  # u_{n-1}
        u_nm2 = valid_history[1]  # u_{n-2}

        for _ in range(n_correcoes):
            # Calcular f(u_n), f(u_{n-1}), f(u_{n-2})
            f_n = complex_prod(z, u_n)
            f_nm1 = complex_prod(z, u_nm1)
            f_nm2 = complex_prod(z, u_nm2)

            # AM4: u_est = u_n + (9/24*f_est + 19/24*f_n - 5/24*f_{n-1} + 1/24*f_{n-2})
            term1 = complex_scale(f_est, 9.0 / 24.0)
            term2 = complex_scale(f_n, 19.0 / 24.0)
            term3 = complex_scale(f_nm1, -5.0 / 24.0)  # -5/24
            term4 = complex_scale(f_nm2, 1.0 / 24.0)  # +1/24

            # Combinar termos
            corr_sum = complex_add(complex_add(term1, term2), complex_add(term3, term4))
            u_est = complex_add(u_n, corr_sum)

            # Re-avaliar f_est para próxima iteração
            f_est = complex_prod(z, u_est)

        return u_est

    return u_pred  # Retornar preditor se corretor não for AM4


# Função para processar um ponto individual
@njit(cache=True)
def process_single_point(h, k, tipo, xl, xr, yb, yt, incle, T, tol):
    real_z = xl + (h * (np.abs(xr - xl) / incle))
    img_z = yb + (k * (np.abs(yt - yb) / incle))
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


# Função wrapper para paralelização com tracking
def process_point_with_tracking(args):
    h, k, tipo, xl, xr, yb, yt, incle, T, tol, thread_id = args
    start_time = perf_counter()
    result = process_single_point(h, k, tipo, xl, xr, yb, yt, incle, T, tol)
    end_time = perf_counter()
    processing_time = end_time - start_time
    return (*result, processing_time, thread_id)


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
def save_statistics(thread_ids, processing_times, n_threads, execution_time, tipo):
    stats = []
    for thread_id in range(n_threads):
        mask = thread_ids == thread_id
        if np.any(mask):
            total_time = np.sum(processing_times[mask])
            avg_time = np.mean(processing_times[mask])
            points_count = np.sum(mask)
            stats.append((thread_id, points_count, total_time, avg_time))

    with open(f'output/{tipo}_{n_threads}threads_stats.txt', 'w') as f:
        f.write(f"ESTATÍSTICAS DE BALANCEAMENTO - {n_threads} THREADS\n")
        f.write("=" * 50 + "\n")
        f.write(f"{'Thread':<8} {'Pontos':<8} {'Tempo Total':<12} {'Tempo Médio':<12} {'% Carga':<8}\n")
        f.write("-" * 50 + "\n")

        total_processing_time = np.sum(processing_times)
        for thread_id, count, total_time, avg_time in stats:
            load_percentage = (total_time / total_processing_time) * 100
            f.write(
                f"{thread_id:<8} {count:<8} {total_time:.4f}s     {avg_time * 1000:.2f}ms      {load_percentage:.1f}%\n")

        if stats:
            total_times = [stat[2] for stat in stats]
            imbalance_ratio = max(total_times) / min(total_times)
            efficiency = min(total_times) / max(total_times) * 100
            f.write("-" * 50 + "\n")
            f.write(f"Taxa de desbalanceamento: {imbalance_ratio:.2f}\n")
            f.write(f"Eficiência: {efficiency:.1f}%\n")
            f.write(f"Tempo total de execução: {execution_time:.2f}s\n")


# Função para visualizar resultados
def visualize_balance_results(results, n_threads, execution_time, tipo):
    real_coords = np.array([r[0] for r in results])
    imag_coords = np.array([r[1] for r in results])
    stable_flags = np.array([r[2] for r in results])
    processing_times = np.array([r[4] for r in results])
    thread_ids = np.array([r[5] for r in results])

    thread_colors = generate_thread_colors(n_threads)

    # 1. Mapa de Cores por Thread
    plt.figure(figsize=(10, 8))
    for thread_id in range(n_threads):
        mask = thread_ids == thread_id
        if np.any(mask):
            plt.scatter(real_coords[mask], imag_coords[mask],
                        color=thread_colors[thread_id], s=2, alpha=0.7,
                        label=f'Thread {thread_id}')

    plt.title(f'Distribuição por Thread - {tipo} - {n_threads} threads')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'output/thread_distribution_{tipo}_{n_threads}threads.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Tempo de Processamento
    plt.figure(figsize=(10, 8))
    sizes = np.clip(processing_times * 5000, 1, 20)
    for thread_id in range(n_threads):
        mask = thread_ids == thread_id
        if np.any(mask):
            plt.scatter(real_coords[mask], imag_coords[mask],
                        color=thread_colors[thread_id], s=sizes[mask], alpha=0.7)

    plt.title(f'Tempo de Processamento - {tipo} - {n_threads} threads')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'output/processing_time_{tipo}_{n_threads}threads.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Ordem de Processamento
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(real_coords, imag_coords,
                          c=np.arange(len(results)), cmap='viridis', s=2, alpha=0.7)
    plt.colorbar(scatter, label='Ordem de Processamento')
    plt.title(f'Ordem de Processamento - {tipo} - {n_threads} threads')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'output/processing_order_{tipo}_{n_threads}threads.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    # Salvar estatísticas
    save_statistics(thread_ids, processing_times, n_threads, execution_time, tipo)


# Versão modificada da run_with_threads com tracking
def run_with_threads_tracking(num_threads):
    """Executa o processamento com tracking de threads"""
    print(f"\n=== Testando com {num_threads} thread(s) ===")

    start_time_total = perf_counter()

    # Preparar pontos com atribuição de thread
    points_to_process = []
    for h in range(incle + 1):
        for k in range(incle + 1):
            thread_id = (h * (incle + 1) + k) % num_threads
            points_to_process.append((h, k, tipo, xl, xr, yb, yt, incle, T, tol, thread_id))

    # Processamento paralelo
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch_size = 50  # Reduzido para melhor visualização do progresso
        for i in range(0, len(points_to_process), batch_size):
            batch = points_to_process[i:i + batch_size]
            batch_results = list(executor.map(process_point_with_tracking, batch))
            results.extend(batch_results)

            progress = (i + len(batch)) / len(points_to_process) * 100
            print(f"Progresso: {progress:.1f}%")

    end_time_total = perf_counter()
    total_execution_time = end_time_total - start_time_total

    # Gerar visualizações de balanceamento
    visualize_balance_results(results, num_threads, total_execution_time, tipo)

    # Também gerar plot original da região de estabilidade
    stable_points = [(r[0], r[1]) for r in results if r[2]]
    unstable_points = [(r[0], r[1]) for r in results if r[3]]

    plt.figure(figsize=(8, 8))
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
    plt.xlim(xl, xr)
    plt.ylim(yb, yt)

    if stable_points:
        stable_x, stable_y = zip(*stable_points)
        plt.plot(stable_x, stable_y, 'bo', markersize=0.5, label='Estável', alpha=0.7)
    if unstable_points:
        unstable_x, unstable_y = zip(*unstable_points)
        plt.plot(unstable_x, unstable_y, 'ro', markersize=0.5, label='Instável', alpha=0.7)

    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.title(f'Região de Estabilidade - {tipo} - {num_threads} threads')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'output/{tipo}_{num_threads}threads.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Tempo com {num_threads} threads: {total_execution_time:.2f}s")
    return total_execution_time

# Execução principal
if __name__ == "__main__":
    threads_to_test = [1, 2, 3, 4, 5, 6, 7, 8]
    #threads_to_test = [1, 2, 4]
    execution_times = {}

    print("Iniciando processamento da região de estabilidade...")
    print(f"Total de pontos: {total_points}")
    print(f"Tipo de método: {tipo}")

    # Executar para diferentes números de threads
    for n_threads in threads_to_test:
        try:
            time_taken = run_with_threads_tracking(n_threads)
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
        for threads, time_taken in execution_times.items():
            speedup_val = base_time / time_taken
            efficiency = (speedup_val / threads) * 100
            print(f"Threads: {threads:2d} | Tempo: {time_taken:6.2f}s | "
                  f"Speedup: {speedup_val:5.2f}x | Eficiência: {efficiency:5.1f}%")