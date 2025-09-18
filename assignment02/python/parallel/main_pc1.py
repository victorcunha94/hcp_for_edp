import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from adams_bashforth_moulton_methods import*
import os
import colorsys
from utils_tools import*
from numba import njit, config, prange, set_num_threads, get_num_threads


############ Configuração ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '1'

# Parâmetros do problema
tol = 1e-10
T = 3000
incle = 150
xl, xr, yb, yt = -3.0, 1.0, -2.0, 2.0


method = 'ABM1'

# Criar pasta de output
os.makedirs(f'{method}', exist_ok=True)



@njit(cache=True)
def AB1_predict(u_n, z):
    # U^{n+1} = U^n + z*U^n
    f_n = prod(z, u_n)
    return np.array([u_n[0] + f_n[0], u_n[1] + f_n[1]])

@njit(cache=True)
def AB2_predict(u_n, u_nm1, z):
    # U^{n+1} = U^n + (1/2)*(3 f_n - f_{n-1})
    f_n   = prod(z, u_n)
    f_nm1 = prod(z, u_nm1)
    num0 = 3.0 * f_n[0] - 1.0 * f_nm1[0]
    num1 = 3.0 * f_n[1] - 1.0 * f_nm1[1]
    return np.array([u_n[0] + num0/2.0, u_n[1] + num1/2.0])

@njit(cache=True)
def AB3_predict(u_n, u_nm1, u_nm2, z):
    # U^{n+1} = U^n + (1/12)*(23 f_n - 16 f_{n-1} + 5 f_{n-2})
    f_n   = prod(z, u_n)
    f_nm1 = prod(z, u_nm1)
    f_nm2 = prod(z, u_nm2)
    num0 = 23.0*f_n[0] - 16.0*f_nm1[0] + 5.0*f_nm2[0]
    num1 = 23.0*f_n[1] - 16.0*f_nm1[1] + 5.0*f_nm2[1]
    return np.array([u_n[0] + num0/12.0, u_n[1] + num1/12.0])


@njit(cache=True)
def ABM1_step(u_n, u_nm1, z, n_corr=1):
    """
    Adams-Bashforth-Moulton de ordem 1 (ABM1)

    Convenção:
    u_n = U^n (mais recente), u_nm1 = U^{n-1}

    Esquema P(EC)ᴺ:
    P: Ypre = u_n + (1/2)*(3 f_n - 1 f_{n-1})     -> AB2 (precisamos de 2 passos)
    Mas para AB1 (ordem 1): Ypre = u_n + f_n

    Na verdade, para ordem 1 usamos:
    Preditor: Adams-Bashforth ordem 1
    Corretor: Adams-Moulton ordem 1

    Parameters:
    u_n: array [real, imag] - solução no tempo n
    u_nm1: array [real, imag] - solução no tempo n-1
    z: array [real, imag] - parâmetro complexo
    n_corr: int - número de iterações corretor

    Returns:
    Ypre: array - preditor (Adams-Bashforth ordem 1)
    Ycor: array - corretor (Adams-Moulton ordem 1)
    """
    # --- Preditor AB1: Ypre = u_n + f_n  ---
    # f_n = z * u_n
    f_n = complex_prod(z, u_n)
    f_nm1 = complex_prod(z, u_nm1)

    # Adams-Bashforth ordem 1: Ypre = u_n + f_n
    Ypre = complex_add(u_n, f_n)

    # --- Estimativa inicial E ---
    f_est = complex_prod(z, Ypre)

    # --- Corretor iterativo AM1: Ycor = u_n + f_est ---
    Ycor = Ypre.copy()
    for _ in range(n_corr):
        # Adams-Moulton ordem 1: Ycor = u_n + f_est
        Ycor = complex_add(u_n, f_est)

        # re-avaliação E após correção
        f_est = complex_prod(z, Ycor)

    return Ypre, Ycor


@njit(cache=True)
def process_single_point_numba(ix, iy, xl, xr, yb, yt, incle, T, tol, n_corr):
    """
    Processa um ponto com ABM1 (Adams-Bashforth-Moulton ordem 1)

    ABM1 precisa de 2 pontos no histórico: u_n e u_{n-1}
    """
    real_z = xl + (ix * (xr - xl) / incle)
    img_z = yb + (iy * (yt - yb) / incle)
    z = np.array([real_z, img_z])

    # --- BOOTSTRAP INICIAL (2 pontos) ---
    # Precisamos de u_{n-1} e u_n para o ABM1
    u0 = np.array([1.0, 0.0])  # u_{n-1} = u⁰
    u1 = euler_implict_numba(u0, z)  # u_n = u¹ (Euler implícito)

    is_stable = False
    is_unstable = False

    # --- ITERAÇÃO TEMPORAL ---
    for n_step in range(2, T):  # Começa do passo 2 (já temos u⁰ e u¹)
        # ABM1 precisa de: u_n (mais recente) e u_{n-1} (anterior)
        Ypre, Ycor = ABM1_step(u1, u0, z, n_corr)  # u1 = u_n, u0 = u_{n-1}

        # --- ATUALIZA HISTÓRICO ---
        u0 = u1  # u_{n-1} = u_n antigo
        u1 = Ycor  # u_n = novo valor

        norm = complex_norm(u1)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1.0 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable



# Função principal compilada
@njit(parallel=True, cache=True)
def process_all_points_compiled(xl_val, xr_val, yb_val, yt_val, incle_val, T_val, tol_val, num_threads):
    total_points = (incle_val + 1) * (incle_val + 1)
    all_real_z = np.zeros(total_points)
    all_img_z = np.zeros(total_points)
    all_stable = np.zeros(total_points, dtype=np.bool_)
    all_unstable = np.zeros(total_points, dtype=np.bool_)
    all_thread_ids = np.zeros(total_points, dtype=np.int32)

    linhas = incle_val + 1
    linhas_por_thread = linhas // num_threads
    if linhas_por_thread == 0:
        linhas_por_thread = 1



    for idx in prange(total_points):

        h = idx // (incle_val + 1) # índice da linha
        k = idx % (incle_val + 1)  # índice da coluna

        real_z, img_z, is_stable, is_unstable = process_single_point_numba(
            h, k, xl_val, xr_val, yb_val, yt_val, incle_val, T_val, tol_val, 1
        )

        all_real_z[idx] = real_z
        all_img_z[idx] = img_z
        all_stable[idx] = is_stable
        all_unstable[idx] = is_unstable
        # simulamos o thread_id, bloco de linhas contínuas
        # -> corresponde à ideia "cada thread cuida de linhas"
        tid = h // linhas_por_thread
        if tid >= num_threads:
            tid = num_threads - 1
        all_thread_ids[idx] = tid


    return all_real_z, all_img_z, all_stable, all_unstable, all_thread_ids


def plot_stability_region_by_thread(real_coords, imag_coords, stable_flags, unstable_flags, thread_ids, num_threads,
                                    filename, xl, xr, yb, yt):
    """Plota a região de estabilidade colorida por thread"""
    plt.figure(figsize=(10, 10))

    # Gerar cores distintas para cada thread
    colors = []
    for i in range(num_threads):
        hue = i / max(num_threads, 1)  # Distribui cores uniformemente
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)  # Cores vibrantes
        colors.append((r, g, b))

    # Plotar cada ponto com a cor da thread que o processou
    for i in range(len(real_coords)):
        if stable_flags[i]:
            color = colors[thread_ids[i] % num_threads]
            plt.plot(real_coords[i], imag_coords[i], 'o', color=color, markersize=1.5, alpha=0.9)
        elif unstable_flags[i]:
            plt.plot(real_coords[i], imag_coords[i], 'ko', markersize=0.0, alpha=0.3)

    # Adicionar legenda de cores
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=f'Thread {i}') for i in range(num_threads)]
    legend_elements.append(Patch(facecolor='black', label='Instável'))

    plt.legend(handles=legend_elements, loc='upper right')
    plt.xlabel('Re(z)')
    plt.ylabel('Img(z)')
    plt.title(f'Região de Estabilidade - Colorida por Thread ({num_threads} threads)')
    plt.grid(True, alpha=0.3)
    plt.xlim(xl, xr)
    plt.ylim(yb, yt)

    plt.gca().set_aspect('equal', adjustable='box')  # Mantém proporção 1:1
    plt.axhline(0, color='k', linewidth=1.0)  # Eixo X (real)
    plt.axvline(0, color='k', linewidth=1.0)  # Eixo Y (imaginário)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()



def plot_stability_region_basic(real_coords, imag_coords, stable_flags, unstable_flags, filename, xl, xr, yb, yt):
    """Plota a região de estabilidade básica"""
    plt.figure(figsize=(10, 10))

    # Separar pontos estáveis e instáveis
    stable_real = real_coords[stable_flags]
    stable_imag = imag_coords[stable_flags]
    unstable_real = real_coords[unstable_flags]
    unstable_imag = imag_coords[unstable_flags]

    plt.plot(stable_real, stable_imag, 'bo', markersize=2.5, alpha=0.6, label='Estável')
    plt.plot(unstable_real, unstable_imag, 'ro', markersize=0.0, alpha=0.3, label='Instável')

    plt.xlabel('Re(z)')
    plt.ylabel('Img(z)')
    plt.title('Região de Estabilidade do Método Numérico')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(xl, xr)
    plt.ylim(yb, yt)

    plt.gca().set_aspect('equal', adjustable='box')  # Mantém proporção 1:1
    plt.axhline(0, color='k', linewidth=1.0)  # Eixo X (real)
    plt.axvline(0, color='k', linewidth=1.0)  # Eixo Y (imaginário)

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


# Função wrapper para compatibilidade
def run_with_numba_prange(num_threads):
    print(f"\n=== Testando com {num_threads} thread(s) ===")

    os.environ['OMP_NUM_THREADS'] = str(num_threads)
    set_num_threads(num_threads)

    start_time_total = perf_counter()
    inicio = perf_counter()

    real_coords, imag_coords, stable_flags, unstable_flags, thread_ids = process_all_points_compiled(
        xl, xr, yb, yt, incle, T, tol, num_threads
    )

    fim = perf_counter()
    tempo_total = fim - inicio

    print(f"Tempo com {num_threads} threads: {tempo_total:.2f}s")
    print(f"Pontos estáveis: {np.sum(stable_flags)}, Instáveis: {np.sum(unstable_flags)}")

    #return tempo_total, thread_ids
    return tempo_total, real_coords, imag_coords, stable_flags, unstable_flags, thread_ids


# Execução principal
if __name__ == "__main__":
    #threads_to_test = [1, 2, 3, 4,5,6,7,8,9,10,11,12,13,14,15,16]
    threads_to_test = [1]

    print("=== WARM-UP (Compilação Inicial) ===")
    os.environ['OMP_NUM_THREADS'] = '1'
    set_num_threads(1)

    # Warm-up com problema pequeno
    try:
        result = process_all_points_compiled(-3.0, 1.0, -2.0, 2.0, 5, 10, 1e-5, 1)
        print("Warm-up concluído!")



        all_thread_ids = {}
        execution_times = {}

        for n_threads in threads_to_test:
            (time_taken, real_coords, imag_coords,
             stable_flags, unstable_flags, thread_ids) = run_with_numba_prange(n_threads)


            execution_times[n_threads] = time_taken
            all_thread_ids[n_threads] = thread_ids

            # # === CHAMAR OS PLOTS ===

            #
            #
            if n_threads % 1 == 0:
                plot_stability_region_by_thread(
                    real_coords, imag_coords, stable_flags, unstable_flags, thread_ids,
                    n_threads,
                    f"{method}/stability_by_thread_{n_threads}threads.png",
                    xl=-3.0, xr=1.0, yb=-2.0, yt=2.0)

            if n_threads == 0:
                plot_stability_region_basic(
                    real_coords, imag_coords, stable_flags, unstable_flags,
                    f"{method}/stability_basic_{n_threads}threads.png",
                    xl=-3.0, xr=1.0, yb=-2.0, yt=2.0)



        # Plotar resultados de performance
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
        plt.savefig(f'{method}/performance_comparison{method}.png', dpi=300)
        plt.show()


        # Análise de distribuição de trabalho
        print("\n=== DISTRIBUIÇÃO POR THREAD ===")
        for n_threads in threads_to_test:
            print(f"\nCom {n_threads} threads:")
            thread_ids = all_thread_ids[n_threads]
            unique, counts = np.unique(thread_ids, return_counts=True)
            for thread_id, count in zip(unique, counts):
                print(f"  Thread {thread_id}: {count} pontos ({count / len(thread_ids) * 100:.1f}%)")

    except Exception as e:
        print(f"Erro durante execução: {e}")
        import traceback

        traceback.print_exc()
