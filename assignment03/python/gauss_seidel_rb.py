#!/usr/bin/env python3

"""
mpirun -np 4 python3 main_redblack_gs.py --N 100 --nx 2 --ny 2 --tol 1e-10
"""
from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import pandas as pd
import os

def grid_dims(nx, ny, size):
    if nx * ny != size:
        raise ValueError(f"Erro: nx * ny = {nx} * {ny} = {nx * ny} != {size}")
    return [nx, ny]

def _partition_1d(n_interior, n_procs_dim, coord):
    base = n_interior // n_procs_dim
    rem = n_interior % n_procs_dim
    if coord < rem:
        local = base + 1
        start = 1 + coord * local
    else:
        local = base
        start = 1 + rem * (base + 1) + (coord - rem) * base
    end = start + local
    return start, end, local

def is_red_cell(i, j):
    """Define o padrão xadrez (red-black)"""
    return (i + j) % 2 == 0

def gauss_seidel_redblack_mpi(N, nx, ny, max_iter=1000000, tol=1e-10, L=1.0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        try:
            dims = grid_dims(nx, ny, size)
            error_flag = False
        except ValueError as e:
            print(f"ERRO: {e}")
            error_flag = True
    else:
        dims = None
        error_flag = None

    error_flag = comm.bcast(error_flag, root=0)
    if error_flag:
        comm.Abort(1)

    dims = grid_dims(nx, ny, size)
    cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
    coords = cart.Get_coords(rank)
    left, right = cart.Shift(0, 1)
    down, up = cart.Shift(1, 1)

    dx = L / (N - 1)
    dx2 = dx * dx
    pts_x = N - 2
    pts_y = N - 2

    start_x, end_x, local_nx = _partition_1d(pts_x, dims[0], coords[0])
    start_y, end_y, local_ny = _partition_1d(pts_y, dims[1], coords[1])

    if local_nx == 0 or local_ny == 0:
        meta = dict(rank=rank, error="Empty subdomain")
        return meta, [], None

    # Alocar U
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    f_local = np.zeros_like(U)

    def f_global(i, j):
        x = i * dx
        y = j * dx
        return 8.0 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

    # Pré-calcular f_local
    i_global = np.arange(start_x, end_x).reshape(-1, 1)
    j_global = np.arange(start_y, end_y).reshape(1, -1)
    x = i_global * dx
    y = j_global * dx
    f_local[1:local_nx+1, 1:local_ny+1] = 8.0 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

    # Buffers de comunicação
    send_buf_up     = np.empty(local_nx, dtype=np.float64)
    send_buf_down   = np.empty(local_nx, dtype=np.float64)
    send_buf_right  = np.empty(local_ny, dtype=np.float64)
    send_buf_left   = np.empty(local_ny, dtype=np.float64)

    recv_buf_up     = np.empty(local_nx, dtype=np.float64)
    recv_buf_down   = np.empty(local_nx, dtype=np.float64)
    recv_buf_right  = np.empty(local_ny, dtype=np.float64)
    recv_buf_left   = np.empty(local_ny, dtype=np.float64)

    exec_time = 0.0
    comm_time_total = 0.0
    comm.Barrier()
    final_error = None

    global_iteration = 0
    global_converged = False

    while global_iteration < max_iter and not global_converged:
        max_err_loc = 0.0
        
        # 🔥 FASE 1: CÉLULAS VERMELHAS (RED)
        t_comp_start = time.perf_counter()
        
        for i in range(1, local_nx + 1):
            for j in range(1, local_ny + 1):
                i_global = start_x + (i - 1)
                j_global = start_y + (j - 1)
                
                if is_red_cell(i_global, j_global):  # 🔴 Célula vermelha
                    old_val = U[i, j]
                    # 🔥 GAUSS-SEIDEL: usa valores já atualizados
                    U[i, j] = 0.25 * (
                        U[i-1, j] + U[i+1, j] +  # esquerda, direita
                        U[i, j-1] + U[i, j+1] +  # abaixo, acima
                        dx2 * f_local[i, j]
                    )
                    error = abs(U[i, j] - old_val)
                    max_err_loc = max(max_err_loc, error)
        
        exec_time += time.perf_counter() - t_comp_start

        # 🔄 COMUNICAÇÃO APÓS FASE VERMELHA
        t_comm_start = time.perf_counter()
        requests = []

        # Preparar buffers de envio
        np.copyto(send_buf_up, U[1:-1, -2])
        np.copyto(send_buf_down, U[1:-1, 1])
        np.copyto(send_buf_right, U[-2, 1:-1])
        np.copyto(send_buf_left, U[1, 1:-1])

        # Postar recepções
        if up != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_up, source=up, tag=1))
        if down != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_down, source=down, tag=0))
        if right != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))
        if left != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))

        # Postar envios
        if up != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_up, dest=up, tag=0))
        if down != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_down, dest=down, tag=1))
        if right != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_right, dest=right, tag=2))
        if left != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_left, dest=left, tag=3))

        MPI.Request.Waitall(requests)

        # Aplicar bordas recebidas
        if up != MPI.PROC_NULL:
            U[1:-1, -1] = recv_buf_up
        if down != MPI.PROC_NULL:
            U[1:-1, 0] = recv_buf_down
        if right != MPI.PROC_NULL:
            U[-1, 1:-1] = recv_buf_right
        if left != MPI.PROC_NULL:
            U[0, 1:-1] = recv_buf_left

        comm_time_total += time.perf_counter() - t_comm_start

        # 🔥 FASE 2: CÉLULAS PRETAS (BLACK)
        t_comp_start = time.perf_counter()
        
        for i in range(1, local_nx + 1):
            for j in range(1, local_ny + 1):
                i_global = start_x + (i - 1)
                j_global = start_y + (j - 1)
                
                if not is_red_cell(i_global, j_global):  # ⚫ Célula preta
                    old_val = U[i, j]
                    # 🔥 GAUSS-SEIDEL: usa valores já atualizados (incluindo vermelhos)
                    U[i, j] = 0.25 * (
                        U[i-1, j] + U[i+1, j] +  # esquerda, direita
                        U[i, j-1] + U[i, j+1] +  # abaixo, acima
                        dx2 * f_local[i, j]
                    )
                    error = abs(U[i, j] - old_val)
                    max_err_loc = max(max_err_loc, error)
        
        exec_time += time.perf_counter() - t_comp_start

        # 🔄 COMUNICAÇÃO APÓS FASE PRETA
        t_comm_start = time.perf_counter()
        requests = []

        # Preparar buffers de envio
        np.copyto(send_buf_up, U[1:-1, -2])
        np.copyto(send_buf_down, U[1:-1, 1])
        np.copyto(send_buf_right, U[-2, 1:-1])
        np.copyto(send_buf_left, U[1, 1:-1])

        # Postar recepções
        if up != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_up, source=up, tag=1))
        if down != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_down, source=down, tag=0))
        if right != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))
        if left != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))

        # Postar envios
        if up != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_up, dest=up, tag=0))
        if down != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_down, dest=down, tag=1))
        if right != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_right, dest=right, tag=2))
        if left != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_left, dest=left, tag=3))

        MPI.Request.Waitall(requests)

        # Aplicar bordas recebidas
        if up != MPI.PROC_NULL:
            U[1:-1, -1] = recv_buf_up
        if down != MPI.PROC_NULL:
            U[1:-1, 0] = recv_buf_down
        if right != MPI.PROC_NULL:
            U[-1, 1:-1] = recv_buf_right
        if left != MPI.PROC_NULL:
            U[0, 1:-1] = recv_buf_left

        comm_time_total += time.perf_counter() - t_comm_start

        global_iteration += 1

        # Verificar convergência
        if global_iteration % 10 == 0:  # Verificar a cada 10 iterações
            max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)
            
            if max_err_glob < tol:
                global_converged = True
                final_error = max_err_glob
                break

    if not global_converged:
        max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)
        final_error = max_err_glob
    
    if final_error is None:
        final_error = 0.0

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
        overhead=overhead,
        final_error=final_error,
        U_data=u_local_data,
        method="Gauss-Seidel-RedBlack"
    )

    if rank == 0:
        print(f"🎯 MÉTODO: Gauss-Seidel com Domínio Red-Black")
        print(f"🔴⚫ Padrão: Células Vermelhas + Pretas")
        print(f"📊 Processos: {nx*ny} ({nx}x{ny})")
        print(f"⏱️  Tempo execução = {exec_time}|Overhead = {overhead}|")
        print(f"🔄 Iterações = {global_iteration}|Tempo comunicação = {comm_time_total}|")
        print(f"📈 Erro final = {final_error:.2e}")

    return meta, [], U

def main():
    parser = argparse.ArgumentParser(description="Gauss-Seidel com Domínio Red-Black")
    parser.add_argument("--N", type=int, default=256, help="Tamanho do grid NxN")
    parser.add_argument("--nx", type=int, required=True, help="Processos na direção x")
    parser.add_argument("--ny", type=int, required=True, help="Processos na direção y")
    parser.add_argument("--max_iter", type=int, default=1000000, help="Máximo de iterações")
    parser.add_argument("--tol", type=float, default=1e-10, help="Tolerância de convergência")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    meta, comm_log, U = gauss_seidel_redblack_mpi(
        args.N, args.nx, args.ny,
        max_iter=args.max_iter,
        tol=args.tol
    )

    # Gather de todos os metadados
    all_meta = comm.gather(meta, root=0)

    if rank == 0:
        if not os.path.exists('output'):
            os.makedirs('output')
        
        caminho_csv = f'output/redblack_gs_{args.nx}x{args.ny}.csv'
        
        # Filtrar processos válidos
        valid_metas = [m for m in all_meta if m is not None and 'start_x' in m and 'U_data' in m]
        
        if valid_metas:
            df = pd.DataFrame(valid_metas)
            df.to_csv(caminho_csv, index=False)
            print(f"💾 Arquivo {caminho_csv} salvo com {len(df)} processos")
        else:
            print("🚨 ERRO: Nenhum processo com dados válidos!")

if __name__ == "__main__":
    main()
