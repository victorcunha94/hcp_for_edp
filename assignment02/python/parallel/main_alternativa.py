import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import time
from numba import njit, prange, config, set_num_threads
from time import perf_counter
import threading
import os
import concurrent.futures

############ Paralelização ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '4'

tol = 1e-08
T = 2000
tipo = "PC-AB4-AM4"
incle = 200

xl = -4
xr = 0.5
yb = -2.5
yt = 2.5

total_points = (incle + 1) * (incle + 1)


# Funções auxiliares para operações com números complexos como arrays [real, imag]
@njit(cache=True)
def complex_prod(a, b):
    """Multiplicação de números complexos representados como [real, imag]"""
    real = a[0] * b[0] - a[1] * b[1]
    imag = a[0] * b[1] + a[1] * b[0]
    return np.array([real, imag])


@njit(cache=True)
def complex_div(a, b):
    """Divisão de números complexos representados como [real, imag]"""
    denominator = b[0] * b[0] + b[1] * b[1]
    real = (a[0] * b[0] + a[1] * b[1]) / denominator
    imag = (a[1] * b[0] - a[0] * b[1]) / denominator
    return np.array([real, imag])


@njit(cache=True)
def complex_sub(a, b):
    """Subtração de números complexos representados como [real, imag]"""
    return np.array([a[0] - b[0], a[1] - b[1]])


@njit(cache=True)
def complex_norm(a):
    """Norma de número complexo representado como [real, imag]"""
    return np.sqrt(a[0] * a[0] + a[1] * a[1])


# Implementação correta do Euler implícito seguindo sua lógica
@njit(cache=True)
def euler_implict_numba(Un, z):
    """Método de Euler implícito conforme sua implementação original"""
    one_zero = np.array([1.0, 0.0])  # [1, 0] representa o número complexo 1 + 0i

    # den = [1,0] - prod([1,0], z)
    prod_result = complex_prod(one_zero, z)
    den = complex_sub(one_zero, prod_result)

    # Un1 = prod(div([1,0], den), Un)
    div_result = complex_div(one_zero, den)
    Un1 = complex_prod(div_result, Un)

    return Un1


@njit(cache=True)
def preditor_corrector_AB_AM_numba(Un, Un1, Un2, Un3, z, preditor_order, corretor_order, n_correcoes):
    """Método Preditor-Corretor AB4-AM4 para números complexos como arrays"""
    # Para o problema y' = z*y, f(y) = prod(z, y)
    h = np.array([1.0, 0.0])  # Passo de tempo como número complexo [1, 0]

    if preditor_order == 4 and corretor_order == 4:
        # Calcular f(y) = z * y
        f_n = complex_prod(z, Un)
        f_n1 = complex_prod(z, Un1)
        f_n2 = complex_prod(z, Un2)
        f_n3 = complex_prod(z, Un3)

        # Coeficientes como números complexos [real, 0]
        coeff_55 = np.array([55.0, 0.0])
        coeff_59 = np.array([59.0, 0.0])
        coeff_37 = np.array([37.0, 0.0])
        coeff_9 = np.array([9.0, 0.0])
        coeff_24 = np.array([24.0, 0.0])

        coeff_19 = np.array([19.0, 0.0])
        coeff_5 = np.array([5.0, 0.0])

        # Preditor AB4: Un4 = Un3 + h*(55*f_n3 - 59*f_n2 + 37*f_n1 - 9*f_n)/24
        term1 = complex_prod(coeff_55, f_n3)
        term2 = complex_prod(coeff_59, f_n2)
        term3 = complex_prod(coeff_37, f_n1)
        term4 = complex_prod(coeff_9, f_n)

        # Calcular 55*f_n3 - 59*f_n2 + 37*f_n1 - 9*f_n
        temp1 = complex_sub(term1, term2)
        temp2 = complex_sub(term3, term4)
        pred_sum = complex_sub(temp1, temp2)  # Note: temp1 - temp2 = (term1-term2) - (term3-term4)

        # Dividir por 24 e multiplicar por h
        pred_scaled = complex_prod(complex_div(pred_sum, coeff_24), h)
        U_pred = complex_add(Un3, pred_scaled)

        # Corretor AM4: Un4 = Un3 + h*(9*f_pred + 19*f_n3 - 5*f_n2 + f_n1)/24
        f_pred = complex_prod(z, U_pred)

        term1_corr = complex_prod(coeff_9, f_pred)
        term2_corr = complex_prod(coeff_19, f_n3)
        term3_corr = complex_prod(coeff_5, f_n2)

        # Calcular 9*f_pred + 19*f_n3 - 5*f_n2 + f_n1
        temp1_corr = complex_add(term1_corr, term2_corr)
        temp2_corr = complex_sub(term3_corr, f_n1)  # -5*f_n2 + f_n1 = -(5*f_n2 - f_n1)
        corr_sum = complex_sub(temp1_corr, temp2_corr)

        # Dividir por 24 e multiplicar por h
        corr_scaled = complex_prod(complex_div(corr_sum, coeff_24), h)
        U_corr = complex_add(Un3, corr_scaled)

        return U_corr

    return Un3  # Fallback


@njit(cache=True)
def complex_add(a, b):
    """Adição de números complexos representados como [real, imag]"""
    return np.array([a[0] + b[0], a[1] + b[1]])


# Função para processar um ponto individual
@njit(cache=True)
def process_single_point(h, k, xl, xr, yb, yt, incle, T, tol):
    real_z = xl + (h * (np.abs(xr - xl) / incle))
    img_z = yb + (k * (np.abs(yt - yb) / incle))
    z = np.array([real_z, img_z])  # z como [real, imag]

    # Inicialização
    Un = np.array([1.0, 0.0])  # [1, 0]
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

        norm = complex_norm(Un4)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1.0 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable


# Função wrapper para paralelização
def process_point_wrapper(args):
    h, k, xl, xr, yb, yt, incle, T, tol = args
    return process_single_point(h, k, xl, xr, yb, yt, incle, T, tol)


def run_with_threads(num_threads):
    """Executa o processamento com número específico de threads"""
    set_num_threads(num_threads)
    print(f"\n=== Testando com {num_threads} thread(s) ===")
    print(f"Threads ativas: {threading.active_count()}")

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
            points_to_process.append((h, k, xl, xr, yb, yt, incle, T, tol))

    # Processamento paralelo
    stable_points = []
    unstable_points = []
    other_points = []

    # Processar em lotes para melhor performance
    batch_size = 1000
    num_batches = (len(points_to_process) + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(points_to_process))
        batch = points_to_process[start_idx:end_idx]

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            batch_results = list(executor.map(process_point_wrapper, batch))

        for real_z, img_z, is_stable, is_unstable in batch_results:
            if is_stable:
                stable_points.append((real_z, img_z))
            elif is_unstable:
                unstable_points.append((real_z, img_z))
            else:
                other_points.append((real_z, img_z))

        # Progresso
        progress = (end_idx / len(points_to_process)) * 100
        print(f"Progresso: {progress:.1f}% - Lote {batch_idx + 1}/{num_batches}")

    end_time_total = perf_counter()
    total_execution_time = end_time_total - start_time_total

    # Plotar resultados
    if stable_points:
        stable_x, stable_y = zip(*stable_points)
        plt.plot(stable_x, stable_y, 'bo', markersize=0.5, label='Estável', alpha=0.7)

    if unstable_points:
        unstable_x, unstable_y = zip(*unstable_points)
        plt.plot(unstable_x, unstable_y, 'ro', markersize=0.5, label='Instável', alpha=0.7)

    if other_points:
        other_x, other_y = zip(*other_points)
        plt.plot(other_x, other_y, 'go', markersize=0.5, label='Indeterminado', alpha=0.3)

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
        f.write(f"Pontos indeterminados: {len(other_points)}\n")
        f.write(f"Tempo médio por ponto: {total_execution_time / total_points * 1000:.4f} ms\n")
        f.write(f"Parâmetros: T={T}, incle={incle}, tol={tol}\n")

    print(f"Tempo com {num_threads} threads: {total_execution_time:.2f}s")
    print(f"Pontos estáveis: {len(stable_points)}, instáveis: {len(unstable_points)}")

    return total_execution_time


# Executar para diferentes números de threads
if __name__ == "__main__":
    threads_to_test = [1, 2, 3, 4, 5, 6, 7, 8]
    execution_times = {}

    print("Iniciando processamento da região de estabilidade...")
    print(f"Total de pontos: {total_points}")
    print(f"Tipo de método: {tipo}")

    # Primeiro executa com 1 thread como referência
    try:
        base_time = run_with_threads(1)
        execution_times[1] = base_time
        print(f"Tempo de referência (1 thread): {base_time:.2f}s")
    except Exception as e:
        print(f"Erro com 1 thread: {e}")
        base_time = None

    # Depois executa com mais threads
    if base_time is not None:
        for threads in threads_to_test[1:]:
            try:
                time_taken = run_with_threads(threads)
                execution_times[threads] = time_taken
                speedup = base_time / time_taken
                print(f"Speedup com {threads} threads: {speedup:.2f}x")
            except Exception as e:
                print(f"Erro com {threads} threads: {e}")

    # Plotar comparação de performance se tiver dados
    if len(execution_times) > 1:
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
            speedup_val = base_time / time_taken
            efficiency = (speedup_val / threads) * 100
            print(
                f"Threads: {threads:2d} | Tempo: {time_taken:6.2f}s | Speedup: {speedup_val:5.2f}x | Eficiência: {efficiency:5.1f}%")
    else:
        print("Não foi possível gerar comparação de performance")