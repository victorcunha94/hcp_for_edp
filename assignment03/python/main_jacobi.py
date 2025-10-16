#!/usr/bin/env python3
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

def jacobi_mpi_cart_optimized(N, nx, ny, max_iter=1000000, tol=1e-10, L=1.0, block_size=50):
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

    # Alocar U e Unew de uma vez
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    def f_global(i, j):
        x = i * dx
        y = j * dx
        return 8.0 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

    # Pré-calcular f_local de forma vetorizada
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
    check_interval = min(100, block_size * 5)
    global_converged = False

    while global_iteration < max_iter and not global_converged:
        # COMUNICAÇÃO ASSÍNCRONA OTIMIZADA
        t_comm_start = time.perf_counter()
        requests = []

        # Preparar buffers de envio
        np.copyto(send_buf_up, U[1:-1, -2])    # Linha abaixo da borda superior
        np.copyto(send_buf_down, U[1:-1, 1])   # Linha acima da borda inferior
        np.copyto(send_buf_right, U[-2, 1:-1]) # Coluna à esquerda da borda direita
        np.copyto(send_buf_left, U[1, 1:-1])   # Coluna à direita da borda esquerda

        # Postar recepções primeiro
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

        # Esperar pela comunicação
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

        # BLOCO DE ITERAÇÕES LOCAIS
        t_comp_start = time.perf_counter()
        max_err_loc = 0.0

        for local_iter in range(block_size):
            if global_iteration >= max_iter:
                break

            Unew[1:-1, 1:-1] = 0.25 * (
                U[:-2, 1:-1] + U[2:, 1:-1] +
                U[1:-1, :-2] + U[1:-1, 2:] +
                dx2 * f_local[1:-1, 1:-1]
            )

            # Calcular erro a cada iteração do bloco
            error_matrix = np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1])
            current_err = float(np.max(error_matrix))
            max_err_loc = max(max_err_loc, current_err)

            U, Unew = Unew, U
            global_iteration += 1

        exec_time += time.perf_counter() - t_comp_start


        # Todos os processos verificam juntos
        if global_iteration % check_interval == 0:
            max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)

            if max_err_glob < tol:
                global_converged = True
                final_error = max_err_glob
                break
        
        # A cada iteração do while, verificamos a convergência global
        if global_converged:
             break

    # Após o loop, garantir que o erro final seja o erro global máximo
    if not global_converged:
        # Se saiu do loop por max_iter, o erro final é o último erro local máximo
        max_err_loc = np.max(np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1]))
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)
    
    if final_error is None:
        # Caso o problema seja trivial e não tenha entrado no loop de erro
        final_error = 0.0

    overhead = comm_time_total / exec_time if exec_time > 0 else 0.0

    meta = dict(
        rank=rank,
        start_x=start_x, end_x=end_x,
        start_y=start_y, end_y=end_y,
        local_nx=local_nx, local_ny=local_ny,
        iterations=global_iteration,
        exec_time=exec_time,
        comm_time=comm_time_total,
        overhead=overhead,
        final_error=final_error
    )

    if rank == 0:
        print(f"Número de processos igual a {nx*ny}")
        print(f"Tempo de execução = {exec_time}|Overhead = {overhead}|")
        print(f"Iterações = {global_iteration}|Tempo total de comunicação = {comm_time_total}|")

    return meta, [], U

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=256)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--block_size", type=int, default=50)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    meta, comm_log, U = jacobi_mpi_cart_optimized(
        args.N, args.nx, args.ny,
        max_iter=args.max_iter,
        tol=args.tol,
        block_size=args.block_size
    )

    if rank == 0:
        if not os.path.exists('output'):
            os.makedirs('output')
        caminho_csv = f'output/results_{args.nx}x{args.ny}.csv'
        df = pd.DataFrame([meta])
        df.to_csv(caminho_csv, index=False)
        #print(f"[rank 0] Arquivo {caminho_csv} salvo.")

if __name__ == "__main__":
    main()
