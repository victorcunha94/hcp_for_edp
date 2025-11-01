#!/usr/bin/env python3
"""
mpirun -np 4 python3 main_sor_mpi.py --N 514 --nx 2 --ny 2 --omega 1.7
Implementação paralela (MPI) do método SOR com ordering red-black (checkerboard).
Baseado no seu código original de Jacobi.
"""

from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import pandas as pd
import os

def exchange_halo(cart, U, send_up, send_down, send_right, send_left,
                  recv_up, recv_down, recv_right, recv_left,
                  up, down, right, left):
    """
    Realiza o halo exchange (non-blocking posts + Waitall) e aplica os halos em U.
    U tem shape (local_nx+2, local_ny+2).
    """
    reqs = []
    # preparar buffers de envio (colunas/linhas internas apropriadas)
    np.copyto(send_up,   U[1:-1, -2])   # linha imediatamente abaixo da borda superior local
    np.copyto(send_down, U[1:-1, 1])    # linha imediatamente acima da borda inferior local
    np.copyto(send_right,U[-2, 1:-1])   # coluna imediatamente à esquerda da borda direita local
    np.copyto(send_left, U[1, 1:-1])    # coluna imediatamente à direita da borda esquerda local

    # post Irecv
    if up != MPI.PROC_NULL:
        reqs.append(cart.Irecv(recv_up, source=up, tag=11))
    if down != MPI.PROC_NULL:
        reqs.append(cart.Irecv(recv_down, source=down, tag=10))
    if right != MPI.PROC_NULL:
        reqs.append(cart.Irecv(recv_right, source=right, tag=13))
    if left != MPI.PROC_NULL:
        reqs.append(cart.Irecv(recv_left, source=left, tag=12))

    # post Isend
    if up != MPI.PROC_NULL:
        reqs.append(cart.Isend(send_up, dest=up, tag=10))
    if down != MPI.PROC_NULL:
        reqs.append(cart.Isend(send_down, dest=down, tag=11))
    if right != MPI.PROC_NULL:
        reqs.append(cart.Isend(send_right, dest=right, tag=12))
    if left != MPI.PROC_NULL:
        reqs.append(cart.Isend(send_left, dest=left, tag=13))

    if reqs:
        MPI.Request.Waitall(reqs)

    # aplicar halos recebidos
    if up != MPI.PROC_NULL:
        U[1:-1, -1] = recv_up
    if down != MPI.PROC_NULL:
        U[1:-1, 0] = recv_down
    if right != MPI.PROC_NULL:
        U[-1, 1:-1] = recv_right
    if left != MPI.PROC_NULL:
        U[0, 1:-1] = recv_left


def sor_mpi_cart_optimized(N, nx, ny, max_iter=1000000, tol=1e-9, L=1.0, omega=None):
    """
    Implementação SOR (red-black) paralela.
    Parâmetros:
      - N: número de pontos na direção (N x N)
      - nx, ny: decomposição de processos (0 para auto)
      - max_iter, tol, L: como antes
      - omega: fator de relaxação (se None, calculado automaticamente)
    Retorna meta, comm_log, U (como antes)
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. determinar dims (nx, ny) similar ao seu código ---
    if rank == 0:
        if nx == 0 and ny == 0:
            dims = [0, 0]
            MPI.Compute_dims(size, 2, dims)
            nx, ny = dims
            error_flag = False
            print(f"MPI_Dims_create: {size} processos -> {nx}x{ny}")
        else:
            if nx * ny != size:
                print(f"ERRO: nx * ny = {nx} * {ny} = {nx * ny} != {size}")
                error_flag = True
            else:
                dims = [nx, ny]
                error_flag = False
    else:
        dims = None
        error_flag = None

    error_flag = comm.bcast(error_flag, root=0)
    if error_flag:
        if rank == 0:
            print("A configuração nx, ny não é válida para o número de processos.")
        comm.Abort(1)

    dims = comm.bcast(dims, root=0)
    nx, ny = dims

    # --- 2. criar cart ---
    if size == 1:
        cart = None
        coords = (0, 0)
        left = right = down = up = MPI.PROC_NULL
    else:
        cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        coords = cart.Get_coords(rank)
        left, right = cart.Shift(0, 1)
        down, up = cart.Shift(1, 1)

    # --- 3. particionamento ---
    dx = L / (N - 1)
    dx2 = dx * dx
    interior_x = N - 2
    interior_y = N - 2

    def part1d(n, p, c):
        base = n // p
        rem = n % p
        local = base + (1 if c < rem else 0)
        start = 1 + c * base + min(c, rem)
        end = start + local
        return start, end, local

    start_x, end_x, local_nx = part1d(interior_x, nx, coords[0])
    start_y, end_y, local_ny = part1d(interior_y, ny, coords[1])

    if local_nx == 0 or local_ny == 0:
        return dict(rank=rank, error="Empty subdomain"), [], None

    # --- 4. alocação e inicialização ---
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    f_local = np.zeros_like(U)

    # índice global (apenas para f e mask parity)
    i_global = np.arange(start_x, end_x).reshape(-1, 1)  # eixo x -> linhas locais
    j_global = np.arange(start_y, end_y).reshape(1, -1)  # eixo y -> colunas locais
    x = i_global * dx
    y = j_global * dx
    f_local[1:local_nx+1, 1:local_ny+1] = np.sin(4*np.pi * x) * np.sin(4*np.pi * y)

    # calcular máscara red/black (0 = red, 1 = black) baseada em índices globais
    parity = (i_global + j_global) % 2
    mask_red = (parity == 0)
    mask_black = (parity == 1)

    # buffers de comunicação (se paralelo)
    if size > 1:
        send_up   = np.empty(local_nx, dtype=np.float64)
        send_down = np.empty(local_nx, dtype=np.float64)
        send_right= np.empty(local_ny, dtype=np.float64)
        send_left = np.empty(local_ny, dtype=np.float64)

        recv_up   = np.empty(local_nx, dtype=np.float64)
        recv_down = np.empty(local_nx, dtype=np.float64)
        recv_right= np.empty(local_ny, dtype=np.float64)
        recv_left = np.empty(local_ny, dtype=np.float64)
    else:
        # placeholders para chamadas sem ramificação
        send_up = send_down = send_right = send_left = None
        recv_up = recv_down = recv_right = recv_left = None

    # --- 5. escolher omega se None (sugestão clássica) ---
    if omega is None:
        # heurística para Poisson 2D: omega ~ 2/(1+sin(pi*dx))
        # dx é h (malha física). Protege para dx pequeno.
        try:
            omega = 2.0 / (1.0 + math.sin(math.pi * dx))
        except Exception:
            omega = 1.5
    # limitações razoáveis
    if omega <= 0 or omega >= 2:
        omega = max(0.1, min(1.99, omega))

    # métricas/timers
    exec_time = 0.0
    comm_time_total = 0.0
    global_iteration = 0
    final_error = None

    # entradas para iteração
    max_iter = int(max_iter)
    tol = float(tol)

    comm.Barrier()
    t_total_start = time.perf_counter()

    # Array views para vizinhanças (mantidos e recalculados a cada iteração pois U muda)
    # Loop principal: em cada iteração fazemos duas varreduras: red e black.
    while global_iteration < max_iter:
        # --- halo antes da varredura RED (para ter vizinhos corretos das bordas) ---
        if size > 1:
            t_comm = time.perf_counter()
            exchange_halo(cart, U, send_up, send_down, send_right, send_left,
                          recv_up, recv_down, recv_right, recv_left,
                          up, down, right, left)
            comm_time_total += time.perf_counter() - t_comm

        # --- VARREDURA RED ---
        t_comp = time.perf_counter()
        # views alinhadas: center A, north,south,west,east
        A = U[1:-1, 1:-1]
        north = U[:-2, 1:-1]
        south = U[2:, 1:-1]
        west  = U[1:-1, :-2]
        east  = U[1:-1, 2:]

        rhs = (north + south + west + east + dx2 * f_local[1:-1, 1:-1]) * 0.25
        # atualização SOR (in-place) para os pontos RED apenas
        # U_new = (1-omega)*A_old + omega * rhs
        update_full = (1.0 - omega) * A + omega * rhs
        # calcular mudança apenas nos pontos RED
        changed = np.abs(update_full[mask_red] - A[mask_red])
        # aplicar atualização só nos RED
        A[mask_red] = update_full[mask_red]
        max_change_red = changed.max() if changed.size > 0 else 0.0

        exec_time += time.perf_counter() - t_comp

        # atualização de iteração não aumentada aqui; só conta after both colors
        # --- trocar halos novamente antes da varredura BLACK para que BLACK veja RED atualizado ---
        if size > 1:
            t_comm = time.perf_counter()
            exchange_halo(cart, U, send_up, send_down, send_right, send_left,
                          recv_up, recv_down, recv_right, recv_left,
                          up, down, right, left)
            comm_time_total += time.perf_counter() - t_comm

        # --- VARREDURA BLACK ---
        t_comp = time.perf_counter()
        # recomputar vistas pois U foi modificado
        A = U[1:-1, 1:-1]
        north = U[:-2, 1:-1]
        south = U[2:, 1:-1]
        west  = U[1:-1, :-2]
        east  = U[1:-1, 2:]

        rhs = (north + south + west + east + dx2 * f_local[1:-1, 1:-1]) * 0.25
        update_full = (1.0 - omega) * A + omega * rhs
        changed = np.abs(update_full[mask_black] - A[mask_black])
        A[mask_black] = update_full[mask_black]
        max_change_black = changed.max() if changed.size > 0 else 0.0

        exec_time += time.perf_counter() - t_comp

        # contabiliza como uma iteração (red + black = 1 SOR iteration)
        max_change_loc = max(max_change_red, max_change_black)
        # denominação para erro relativo
        denom_loc = np.max(np.abs(U[1:-1,1:-1]))
        err_loc = (max_change_loc / denom_loc) if denom_loc != 0 else max_change_loc

        # reduzir globalmente e checar convergência
        if size > 1:
            err_glob = comm.allreduce(err_loc, op=MPI.MAX)
        else:
            err_glob = err_loc

        global_iteration += 1

        if err_glob < tol:
            final_error = err_glob
            break

    # se não convergiu, calcula final_error (último err_glob)
    if final_error is None:
        # recomputa localmente
        denom_loc = np.max(np.abs(U[1:-1,1:-1]))
        final_error = err_glob if denom_loc != 0 else err_loc

    # --- final timings ---
    comm.Barrier()
    t_total_end = time.perf_counter()
    total_time_local = t_total_end - t_total_start
    overhead = comm_time_total / exec_time if exec_time > 0 else 0.0

    u_local_data = U[1:-1, 1:-1].flatten().tolist()

    meta = dict(
        rank=rank,
        start_x=start_x, end_x=end_x,
        start_y=start_y, end_y=end_y,
        local_nx=local_nx, local_ny=local_ny,
        iterations=global_iteration,
        exec_time=exec_time,
        comm_time=comm_time_total,
        total_time=total_time_local,
        overhead=overhead,
        final_error=final_error,
        omega=omega,
        U_data=u_local_data
    )

    # reduções para estatísticas globais (root)
    if size == 1:
        max_total_time = total_time_local
        max_exec_time = exec_time
        total_comm_time = comm_time_total
    else:
        max_total_time = comm.reduce(total_time_local, op=MPI.MAX, root=0)
        max_exec_time = comm.reduce(exec_time, op=MPI.MAX, root=0)
        total_comm_time = comm.reduce(comm_time_total, op=MPI.SUM, root=0)

    if rank == 0:
        print("--------------------------------------------------------")
        print(f"Método SOR (Red-Black) Paralelo Concluído (MPI {nx}x{ny})")
        print(f"Iterações = {global_iteration}")
        print(f"Erro Final = {final_error:.8e}")
        print(f"Omega usado = {omega:.6f}")
        print(f"Tempo Wall-Clock = {max_total_time:.4f}s")
        print(f"Tempo Máximo Computação = {max_exec_time:.4f}s")
        print(f"Tempo Total Comunicação = {total_comm_time:.4f}s")
        if max_exec_time > 0:
            print(f"Overhead = {total_comm_time/max_exec_time:.4f}")
        print("--------------------------------------------------------")

    comm_log = dict(rank=rank, comm_time=comm_time_total, exec_time=exec_time, total_time=total_time_local, iterations=global_iteration)

    return meta, comm_log, U


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256,
                        help="Tamanho total da grade (N x N pontos, N-2 pontos internos)")
    parser.add_argument("--nx", type=int, default=0,
                        help="Número de processos na direção X (0 para auto)")
    parser.add_argument("--ny", type=int, default=0,
                        help="Número de processos na direção Y (0 para auto)")
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--omega", type=float, default=None, help="Fator de relaxação SOR (opcional)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    meta, comm_log, U = sor_mpi_cart_optimized(
        args.N, args.nx, args.ny,
        max_iter=args.max_iter,
        tol=args.tol,
        omega=args.omega
    )

    # coleta metadados e grava resultados no root
    all_meta = comm.gather(meta, root=0)

    if rank == 0:

        if not os.path.exists('output'):
            os.makedirs('output')

        caminho_csv = f'output/results_sor_{args.nx}x{args.ny}.csv'

        valid_metas = [
            m for m in all_meta
            if (m is not None
                and 'start_x' in m
                and 'U_data' in m
                and m.get('local_nx', 0) > 0
                and m.get('local_ny', 0) > 0)
        ]

        if valid_metas:
            df = pd.DataFrame(valid_metas)
            df.to_csv(caminho_csv, index=False)
            print(f"Arquivo {caminho_csv} salvo com {len(df)} processos")

            print("Colunas salvas:", df.columns.tolist())

            total_points = 0
            for _, row in df.iterrows():
                expected_points = row['local_nx'] * row['local_ny']
                actual_points = len(row['U_data']) if isinstance(row['U_data'], list) else 0
                total_points += actual_points

                print(f"   Processo {row['rank']}: {row['local_nx']}x{row['local_ny']} "
                      f"= {expected_points} pontos | U_data: {actual_points} pontos")

            print(f"Total de pontos no domínio: {total_points}")
            print(f"Domínio esperado: {(args.N-2)**2} pontos")

        else:
            print("ERRO: Nenhum processo retornou dados válidos!")


if __name__ == "__main__":
    main()
