#!/usr/bin/env python3
"""
Jacobi paralelo com decomposição cartesiana MPI + plot da decomposição por processo.
Uso: mpiexec -n 4 python jacobi_mpi_cart.py --N 50
"""

from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import matplotlib.pyplot as plt

def _compute_grid_dims(size):
    """Escolhe fatores (px,py) próximos do quadrado que multiplicam size."""
    for px in range(int(math.sqrt(size)), 0, -1):
        if size % px == 0:
            return [px, size // px]
    return [1, size]

def _partition_1d(n_interior, n_procs_dim, coord):
    """Particiona n_interior pontos entre n_procs_dim processos; retorna (start, end, local_n). start é índice global (>=1), end é exclusivo."""
    base = n_interior // n_procs_dim
    rem  = n_interior % n_procs_dim
    if coord < rem:
        local = base + 1
        start = 1 + coord * local
    else:
        local = base
        start = 1 + rem * (base + 1) + (coord - rem) * base
    end = start + local
    return start, end, local

def jacobi_mpi_cart(N, max_iter=10000, tol=1e-8, L=1.0):
    """
    Resolve o problema discreto com Jacobi em malha NxN (contorno Dirichlet zero).
    Retorna: (U_local_interior, niter, exec_time, final_error, start_x, end_x, start_y, end_y)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ========== topologia cartesiana de processos ==========
    dims = _compute_grid_dims(size)        # [Px, Py]
    periods = [False, False]
    cart = comm.Create_cart(dims, periods=periods, reorder=True)
    coords = cart.Get_coords(rank)
    left, right = cart.Shift(0, 1)   # retorna (vizinho_-x, vizinho_+x)
    down, up    = cart.Shift(1, 1)   # retorna (vizinho_-y, vizinho_+y)

    # ========== malha e particionamento ==========
    dx = L / (N - 1)
    dy = dx
    pts_x = N - 2   # pontos internos (excluem fronteiras globais)
    pts_y = N - 2

    start_x, end_x, local_nx = _partition_1d(pts_x, dims[0], coords[0])
    start_y, end_y, local_ny = _partition_1d(pts_y, dims[1], coords[1])

    # Se não há pontos internos neste processo, retorna rapidamente (participa das gathers)
    if local_nx == 0 or local_ny == 0:
        return np.empty((0, 0)), 0, 0.0, 0.0, start_x, end_x, start_y, end_y

    # arrays locais com halos (1 camada cada lado)
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    # função fonte (global)
    def f_global(i_global, j_global):
        x = i_global * dx
        y = j_global * dy
        return -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

    # preencher f_local (somente pontos internos)
    for i_local in range(1, local_nx + 1):
        i_global = start_x + (i_local - 1)
        for j_local in range(1, local_ny + 1):
            j_global = start_y + (j_local - 1)
            f_local[i_local, j_local] = f_global(i_global, j_global)

    comm.Barrier()
    t0 = time.time()
    final_error = None

    for iteration in range(1, max_iter + 1):
        max_err_loc = 0.0

        # ========== comunicar halos (Sendrecv com o MESMO vizinho) ==========
        # vertical (y): trocar linhas
        # topo: enviar linha interna superior (j=local_ny) -> up, receber do up no halo j=local_ny+1
        send_top = U[1:-1, local_ny].copy()
        if up != MPI.PROC_NULL:
            recv_top = np.empty(local_nx, dtype=np.float64)
            cart.Sendrecv(sendbuf=send_top, dest=up, recvbuf=recv_top, source=up)
            U[1:-1, local_ny+1] = recv_top
        else:
            U[1:-1, local_ny+1] = 0.0

        # baixo: enviar linha interna inferior (j=1) -> down, receber do down no halo j=0
        send_bot = U[1:-1, 1].copy()
        if down != MPI.PROC_NULL:
            recv_bot = np.empty(local_nx, dtype=np.float64)
            cart.Sendrecv(sendbuf=send_bot, dest=down, recvbuf=recv_bot, source=down)
            U[1:-1, 0] = recv_bot
        else:
            U[1:-1, 0] = 0.0

        # horizontal (x): trocar colunas
        send_right = U[local_nx, 1:-1].copy()
        if right != MPI.PROC_NULL:
            recv_right = np.empty(local_ny, dtype=np.float64)
            cart.Sendrecv(sendbuf=send_right, dest=right, recvbuf=recv_right, source=right)
            U[local_nx+1, 1:-1] = recv_right
        else:
            U[local_nx+1, 1:-1] = 0.0

        send_left = U[1, 1:-1].copy()
        if left != MPI.PROC_NULL:
            recv_left = np.empty(local_ny, dtype=np.float64)
            cart.Sendrecv(sendbuf=send_left, dest=left, recvbuf=recv_left, source=left)
            U[0, 1:-1] = recv_left
        else:
            U[0, 1:-1] = 0.0

        # ========== atualização Jacobi ==========
        for i in range(1, local_nx + 1):
            i_global = start_x + (i - 1)
            for j in range(1, local_ny + 1):
                j_global = start_y + (j - 1)
                # não atualiza a fronteira global (Dirichlet)
                if (i_global == 0 or i_global == N-1 or j_global == 0 or j_global == N-1):
                    Unew[i, j] = U[i, j]
                else:
                    Unew[i, j] = 0.25 * (U[i-1, j] + U[i+1, j] + U[i, j-1] + U[i, j+1] - (dx*dx) * f_local[i, j])
                    err = abs(Unew[i, j] - U[i, j])
                    if err > max_err_loc:
                        max_err_loc = err

        # swap
        U, Unew = Unew, U

        # convergência global
        max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)
        if max_err_glob < tol:
            final_error = max_err_glob
            if rank == 0:
                print(f"[rank 0] Convergido em {iteration} iterações; erro {max_err_glob:.3e}")
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    exec_time = time.time() - t0
    # retorna a matriz interior (sem halos) e índices globais do bloco
    return U[1:-1, 1:-1], iteration, exec_time, final_error, start_x, end_x, start_y, end_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50, help="Numero de pontos por eixo (inclui contorno)")
    parser.add_argument("--max_iter", type=int, default=20000)
    parser.add_argument("--tol", type=float, default=1e-8)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    U_local, iters, t_exec, err, sx, ex, sy, ey = jacobi_mpi_cart(args.N, max_iter=args.max_iter, tol=args.tol)

    # cada processo envia (start_x, end_x, start_y, end_y, rank) para root
    info = np.array([sx, ex, sy, ey, rank], dtype=int)
    all_info = comm.gather(info, root=0)

    if rank == 0:
        N = args.N
        owner = -1 * np.ones((N, N), dtype=int)   # -1: contorno ou não atribuído
        # preenche apenas pontos internos [1:N-2]x[1:N-2]
        for entry in all_info:
            sxi, exi, syi, eyi, r = entry.tolist()
            # entry pode conter subdomínios vazios (s==e)
            for i_global in range(sxi, exi):
                for j_global in range(syi, eyi):
                    owner[i_global, j_global] = r

        fig, ax = plt.subplots(figsize=(6,6))
        # Mostrar owners; origin='lower' para que j aumente para cima
        im = ax.imshow(owner.T, origin='lower', interpolation='nearest')
        ax.set_title(f"Decomposição de domínio (N={N})")
        ax.set_xlabel("i (x)")
        ax.set_ylabel("j (y)")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Rank proprietário")
        plt.tight_layout()
        plt.savefig("decomposition.png", dpi=150)
        print("[rank 0] Salvo decomposition.png")
    comm.Barrier()


if __name__ == "__main__":
    main()

