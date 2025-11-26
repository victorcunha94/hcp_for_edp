import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from adams_bashforth_moulton_methods import *
import os
from numba import njit, config, prange, set_num_threads
import matplotlib.animation as animation
import subprocess

############ Configuração ###################
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '1'

# Parâmetros do problema
tol = 1e-10
T = 1000
incle = 200
xl, xr, yb, yt = -3.0, 1.0, -3.0, 3.0

# Criar pasta de output
os.makedirs('animation', exist_ok=True)


@njit(cache=True)
def AB1_predict(u_n, z):
    f_n = complex_prod(z, u_n)
    return np.array([u_n[0] + f_n[0], u_n[1] + f_n[1]])


@njit(cache=True)
def AB2_predict(u_n, u_nm1, z):
    f_n = complex_prod(z, u_n)
    f_nm1 = complex_prod(z, u_nm1)
    num0 = 3.0 * f_n[0] - 1.0 * f_nm1[0]
    num1 = 3.0 * f_n[1] - 1.0 * f_nm1[1]
    return np.array([u_n[0] + num0 / 2.0, u_n[1] + num1 / 2.0])


@njit(cache=True)
def ABM3_step(u_n, u_nm1, u_nm2, z, n_corr=1):
    f_n = complex_prod(z, u_n)
    f_nm1 = complex_prod(z, u_nm1)
    f_nm2 = complex_prod(z, u_nm2)

    # Preditor AB3
    term1 = complex_scale(f_n, 23.0 / 12.0)
    term2 = complex_scale(f_nm1, -4.0 / 3.0)
    term3 = complex_scale(f_nm2, 5.0 / 12.0)
    Ypre = complex_add(u_n, complex_add(term1, complex_add(term2, term3)))

    # Corretor iterativo AM3
    f_est = complex_prod(z, Ypre)
    Ycor = Ypre.copy()

    for _ in range(n_corr):
        term1 = complex_scale(f_est, 5.0 / 12.0)
        term2 = complex_scale(f_n, 2.0 / 3.0)
        term3 = complex_scale(f_nm1, -1.0 / 12.0)
        Ycor = complex_add(u_n, complex_add(term1, complex_add(term2, term3)))
        f_est = complex_prod(z, Ycor)

    return Ypre, Ycor


@njit
def process_single_point_numba(ix, iy, xl, xr, yb, yt, incle, T, tol, n_corr):
    real_z = xl + (ix * (xr - xl) / incle)
    img_z = yb + (iy * (yt - yb) / incle)
    z = np.array([real_z, img_z])

    u0 = np.array([1.0, 0.0])
    u1 = AB1_predict(u0, z)
    u2 = AB2_predict(u1, u0, z)

    is_stable = False
    is_unstable = False

    for n_step in range(3, T):
        Ypre, Ycor = ABM3_step(u2, u1, u0, z, n_corr)
        u0, u1, u2 = u1, u2, Ycor

        norm = complex_norm(u2)
        if norm < tol:
            is_stable = True
            break
        elif norm > 1.0 / tol:
            is_unstable = True
            break

    return real_z, img_z, is_stable, is_unstable


@njit(parallel=True, cache=True)
def process_all_points_compiled(xl_val, xr_val, yb_val, yt_val, incle_val, T_val, tol_val, n_corr):
    total_points = (incle_val + 1) * (incle_val + 1)
    all_real_z = np.zeros(total_points)
    all_img_z = np.zeros(total_points)
    all_stable = np.zeros(total_points, dtype=np.bool_)
    all_unstable = np.zeros(total_points, dtype=np.bool_)

    for idx in prange(total_points):
        h = idx // (incle_val + 1)
        k = idx % (incle_val + 1)
        real_z, img_z, is_stable, is_unstable = process_single_point_numba(
            h, k, xl_val, xr_val, yb_val, yt_val, incle_val, T_val, tol_val, n_corr
        )
        all_real_z[idx] = real_z
        all_img_z[idx] = img_z
        all_stable[idx] = is_stable
        all_unstable[idx] = is_unstable

    return all_real_z, all_img_z, all_stable, all_unstable


def create_animation_mp4():
    print("=== INICIANDO CRIAÇÃO DA ANIMAÇÃO MP4 ===")

    # Números de correções para testar
    n_corrections = list(range(1, 101))  # 1 a 100 correções
    frames = []

    # Configuração da figura para alta qualidade
    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
    plt.tight_layout()

    for n_corr in n_corrections:
        print(f"Processando {n_corr} correções...")

        start_time = perf_counter()
        real_coords, imag_coords, stable_flags, unstable_flags = process_all_points_compiled(
            xl, xr, yb, yt, incle, T, tol, n_corr
        )
        end_time = perf_counter()

        print(
            f"Tempo: {end_time - start_time:.2f}s, Estáveis: {np.sum(stable_flags)}, Instáveis: {np.sum(unstable_flags)}")

        # Criar frame
        ax.clear()

        stable_real = real_coords[stable_flags]
        stable_imag = imag_coords[stable_flags]
        unstable_real = real_coords[unstable_flags]
        unstable_imag = imag_coords[unstable_flags]

        ax.plot(stable_real, stable_imag, 'bo', markersize=2.0, alpha=0.8, label='Estável')
        ax.plot(unstable_real, unstable_imag, 'ro', markersize=0.8, alpha=0.3, label='Instável')

        ax.set_xlabel('Re(z)', fontsize=12)
        ax.set_ylabel('Img(z)', fontsize=12)
        ax.set_title(f'Região de Estabilidade - ABM3 com {n_corr} correções', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect('equal')

        # Adicionar frame à animação
        frames.append([ax])

    # Criar animação MP4
    print("Criando animação MP4...")
    ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)

    # Salvar como MP4 com alta qualidade
    ani.save('animation/stability_animation.mp4',
             writer='ffmpeg',
             fps=5,
             dpi=150,
             bitrate=5000,  # Alta qualidade
             extra_args=['-crf', '18', '-preset', 'slow'])  # Parâmetros FFmpeg

    plt.close()
    print("Animação salva como 'animation/stability_animation.mp4'")


def create_animation_ffmpeg_direct():
    """Alternativa: gerar frames PNG e depois converter com FFmpeg"""
    print("=== MÉTODO ALTERNATIVO: FRAMES PNG + FFMPEG ===")

    n_corrections = list(range(1, 101))

    for frame_num, n_corr in enumerate(n_corrections):
        print(f"Processando frame {frame_num + 1}/100...")

        real_coords, imag_coords, stable_flags, unstable_flags = process_all_points_compiled(
            xl, xr, yb, yt, incle, T, tol, n_corr
        )

        # Criar e salvar frame
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

        stable_real = real_coords[stable_flags]
        stable_imag = imag_coords[stable_flags]
        unstable_real = real_coords[unstable_flags]
        unstable_imag = imag_coords[unstable_flags]

        ax.plot(stable_real, stable_imag, 'bo', markersize=2.0, alpha=0.8)
        ax.plot(unstable_real, unstable_imag, 'ro', markersize=0.8, alpha=0.3)

        ax.set_xlabel('Re(z)')
        ax.set_ylabel('Img(z)')
        ax.set_title(f'ABM3 com {n_corr} correções')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(xl, xr)
        ax.set_ylim(yb, yt)
        ax.set_aspect('equal')

        plt.tight_layout()
        plt.savefig(f'animation/frame_{frame_num:03d}.png', dpi=200, bbox_inches='tight')
        plt.close()

    # Converter para MP4 usando FFmpeg
    print("Convertendo frames para MP4...")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-r', '5', '-i', 'animation/frame_%03d.png',
            '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-vf', 'scale=1920:1080',
            'animation/stability_animation_hd.mp4'
        ], check=True)
        print("MP4 criado com sucesso!")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg não encontrado. Instale o FFmpeg ou use o método anterior.")


if __name__ == "__main__":
    # Warm-up
    print("=== WARM-UP ===")
    result = process_all_points_compiled(-3.0, 1.0, -2.0, 2.0, 10, 50, 1e-5, 1)
    print("Warm-up concluído!")

    # Escolha um método:
    create_animation_mp4()  # Método direto (requer ffmpeg no PATH)
    # create_animation_ffmpeg_direct()  # Método alternativo