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

def jacobi_mpi_cart(N, nx, ny, max_iter=10000, tol=1e-8, L=1.0, block_size=1):
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
    pts_x = N - 2
    pts_y = N - 2

    start_x, end_x, local_nx = _partition_1d(pts_x, dims[0], coords[0])
    start_y, end_y, local_ny = _partition_1d(pts_y, dims[1], coords[1])

    if local_nx == 0 or local_ny == 0:
        meta = dict(rank=rank, error="Empty subdomain")
        return meta, [], None

    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    def f_global(i, j):
        x = i * dx
        y = j * dx
        return 8.0 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

    for i in range(1, local_nx + 1):
        i_global = start_x + (i - 1)
        for j in range(1, local_ny + 1):
            j_global = start_y + (j - 1)
            f_local[i, j] = f_global(i_global, j_global)

    recv_buf_up     = np.empty(local_nx, dtype=np.float64)
    recv_buf_down   = np.empty(local_nx, dtype=np.float64)
    recv_buf_right  = np.empty(local_ny, dtype=np.float64)
    recv_buf_left   = np.empty(local_ny, dtype=np.float64)

    exec_time = 0.0
    comm.Barrier()
    final_error = None
    comm_log = []

    global_iteration = 0

    while global_iteration < max_iter:
        requests = []
        t1_comm = time.perf_counter()

        send_buf_up = U[1:-1, local_ny].copy()
        send_buf_down = U[1:-1, 1].copy()
        send_buf_right = U[local_nx, 1:-1].copy()
        send_buf_left = U[1, 1:-1].copy()

        if up != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_up, source=up, tag=1))
            requests.append(cart.Isend(send_buf_up, dest=up, tag=0))

        if down != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_down, source=down, tag=0))
            requests.append(cart.Isend(send_buf_down, dest=down, tag=1))

        if right != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))
            requests.append(cart.Isend(send_buf_right, dest=right, tag=2))

        if left != MPI.PROC_NULL:
            requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))
            requests.append(cart.Isend(send_buf_left, dest=left, tag=3))

        MPI.Request.Waitall(requests)
        dt = time.perf_counter() - t1_comm

        if up != MPI.PROC_NULL:
            U[1:-1, local_ny + 1] = recv_buf_up
        if down != MPI.PROC_NULL:
            U[1:-1, 0] = recv_buf_down
        if right != MPI.PROC_NULL:
            U[local_nx + 1, 1:-1] = recv_buf_right
        if left != MPI.PROC_NULL:
            U[0, 1:-1] = recv_buf_left

        comm_log.append([global_iteration, rank, dt])

        t0 = time.perf_counter()
        max_err_loc = 0.0

        for local_iter in range(block_size):
            if global_iteration >= max_iter:
                break
            global_iteration += 1

            Unew[1:local_nx+1, 1:local_ny+1] = 0.25 * (
                U[0:local_nx, 1:local_ny+1] +
                U[2:local_nx+2, 1:local_ny+1] +
                U[1:local_nx+1, 0:local_ny] +
                U[1:local_nx+1, 2:local_ny+2] +
                (dx*dx) * f_local[1:local_nx+1, 1:local_ny+1]
            )

            error_matrix = np.abs(Unew[1:local_nx+1, 1:local_ny+1] - U[1:local_nx+1, 1:local_ny+1])
            max_err_loc = max(max_err_loc, float(np.max(error_matrix)))

            U, Unew = Unew, U

        exec_time += time.perf_counter() - t0

        max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)

        if max_err_glob < tol:
            final_error = max_err_glob
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    comm_time_total = sum([c[2] for c in comm_log])
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
        data_length=len(u_local_data)
    )
    if rank == 0:
        print(f"Tempo de execução = {exec_time}|Overhead = {overhead}|")
    return meta, comm_log, U

def analytical(X, Y):
    return np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--local_iters", type=int, default=1)
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    meta, comm_log, U = jacobi_mpi_cart(
        args.N, args.nx, args.ny,
        max_iter=args.max_iter,
        tol=args.tol,
        block_size=args.local_iters
    )

    all_meta = comm.gather(meta, root=0)
    all_logs = comm.gather(comm_log, root=0)

    if rank == 0:
        rows = []
        for m, logs in zip(all_meta, all_logs):
            rows.append(m)
            for entry in logs:
                rows.append(dict(
                    rank=entry[1],
                    iteration=entry[0],
                    comm_time=entry[2]
                ))
        if not os.path.exists('output'):
            os.makedirs('output')
        caminho_csv = os.path.join(f'output/results_{args.nx}x{args.ny}.csv')
        df = pd.DataFrame(rows)
        df.to_csv(caminho_csv, index=False)
        print(f"[rank 0] Arquivo {caminho_csv} salvo.")

if __name__ == "__main__":
    main()
