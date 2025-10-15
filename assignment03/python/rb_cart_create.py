#!/usr/bin/env python3
"""
Jacobi paralelo (MPI cartesiano) com logging detalhado em CSV

Exemplo para rodar:
mpirun -np 4 python3 jacobi_cart_create.py --N 50 --nx 2 --ny 2 
"""

from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def exchange_halos(cart, U, local_nx, local_ny, left, right, down, up):
    reqs = []
    recv_temps = []  

    # Helper to schedule Isend/Irecv using contiguous buffers
    def post_send(dest, send_slice):
        # ensure contiguous send buffer (zero-copy when already contiguous)
        sendbuf = np.ascontiguousarray(send_slice)
        reqs.append(cart.Isend(sendbuf, dest=dest))

    def post_recv(source, recv_shape, target_setter):
        # recv into a contiguous temporary buffer, then copy into target with target_setter(buf)
        recvbuf = np.empty(recv_shape, dtype=U.dtype)
        reqs.append(cart.Irecv(recvbuf, source=source))
        recv_temps.append((recvbuf, target_setter))

    # RIGHT neighbor: send rightmost interior column, recv into right halo column
    if right != MPI.PROC_NULL:
        send_slice = U[local_nx, 1:local_ny+1]           # this is a row -> contiguous
        # row is contiguous; can send directly (ascontiguous ensures contiguity)
        post_send(right, send_slice)
        # receive into temp then copy into U[local_nx+1, 1:local_ny+1]
        post_recv(right, (local_ny,), lambda buf: U.__setitem__((local_nx+1, slice(1, local_ny+1)), buf))
    else:
        U[local_nx+1, 1:local_ny+1] = 0.0

    # LEFT neighbor: send leftmost interior column, recv into left halo
    if left != MPI.PROC_NULL:
        send_slice = U[1, 1:local_ny+1]                  # row -> contiguous
        post_send(left, send_slice)
        post_recv(left, (local_ny,), lambda buf: U.__setitem__((0, slice(1, local_ny+1)), buf))
    else:
        U[0, 1:local_ny+1] = 0.0

    # UP neighbor: send top interior row (vertical slice) -> this is a column slice (strided)
    if up != MPI.PROC_NULL:
        # send U[1:local_nx+1, local_ny] is strided -> make contiguous sendbuf
        sendbuf_up = np.ascontiguousarray(U[1:local_nx+1, local_ny])
        reqs.append(cart.Isend(sendbuf_up, dest=up))
        # receive into temp then copy into U[1:local_nx+1, local_ny+1]
        recvbuf_up = np.empty((local_nx,), dtype=U.dtype)
        reqs.append(cart.Irecv(recvbuf_up, source=up))
        recv_temps.append((recvbuf_up, lambda buf: U.__setitem__((slice(1, local_nx+1), local_ny+1), buf)))
    else:
        U[1:local_nx+1, local_ny+1] = 0.0

    # DOWN neighbor: send bottom interior row (vertical slice) -> strided
    if down != MPI.PROC_NULL:
        sendbuf_down = np.ascontiguousarray(U[1:local_nx+1, 1])
        reqs.append(cart.Isend(sendbuf_down, dest=down))
        recvbuf_down = np.empty((local_nx,), dtype=U.dtype)
        reqs.append(cart.Irecv(recvbuf_down, source=down))
        recv_temps.append((recvbuf_down, lambda buf: U.__setitem__((slice(1, local_nx+1), 0), buf)))
    else:
        U[1:local_nx+1, 0] = 0.0

    # Wait for all non-blocking ops to finish
    if reqs:
        MPI.Request.Waitall(reqs)

    # Copy received temp buffers into the U halo locations
    for buf, setter in recv_temps:
        setter(buf)

def grid_dims(nx, ny, size):
    if nx * ny != size:
        raise ValueError(f"Erro: nx * ny = {nx} * {ny} = {nx * ny} != {size} (total de processos)")
    return [nx, ny]

def errors(new, old):
    ea = np.max(np.abs(new - old))
    er = np.max(np.abs(new - old) / (np.abs(new) + 1e-10))
    return ea, er



def _compute_grid_dims(size):
    for px in range(int(math.sqrt(size)), 0, -1): #
        if size % px == 0:
            return [px, size // px]
    return [1, size]

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

def jacobi_mpi_cart(omega, N, nx, ny, max_iter=10000, tol=1e-8, L=1.0, block_size=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        try:
            dims = grid_dims(nx, ny, size)
        except ValueError as e:
            print(f"ERRO: {e}")
            error_flag = True
        else:
            error_flag = False
    else:
        error_flag = None
        dims = None
        
    # Broadcast do status de erro
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
        return meta, []

    U = np.zeros((local_nx + 2, local_ny + 2))
    Uold = np.zeros_like(U)
    f_local = np.zeros_like(U)

    def f_global(i, j):
        x = i * dx
        y = j * dx
        return 5 * np.pi**2 * np.sin(np.pi * x) * np.sin(2*np.pi * y)
        #return 1.0
    # Inicializar f_local
    for i in range(1, local_nx + 1):
        i_global = start_x + (i - 1)
        for j in range(1, local_ny + 1):
            j_global = start_y + (j - 1)
            f_local[i, j] = f_global(i_global, j_global)

    comm.Barrier()
    t0 = time.perf_counter()
    final_error = None
    comm_log = []

    global_iteration = 0
    parity_offset = (start_x + start_y) % 2
    # build red/black masks once
    i_idx, j_idx = np.indices((local_nx + 2, local_ny + 2))
    color_mask = (i_idx + j_idx + parity_offset) % 2

    mask_red = (color_mask == 0)
    mask_black = (color_mask == 1)
    
    while global_iteration < max_iter:
        global_iteration += 1
        max_err_loc = 10.0
        max_eabs_loc = 10.0
        max_erel_loc = 10.0
        Uold[:,:] = U[:,:]
        
        # === FASE 1: COMUNICAÇÃO (ANTES do cálculo) ===
        exchange_halos(cart, U, local_nx, local_ny, left, right, down, up)

        # === FASE 2: BLOCO DE ITERAÇÕES LOCAIS (VETORIZADO) ===
        # Build color masks once per process
        i_idx, j_idx = np.indices(U.shape)
        color_mask = (i_idx + j_idx + parity_offset) % 2
        mask_red = (color_mask == 0)
        mask_black = (color_mask == 1)

        for local_iter in range(block_size):
            # --- RED phase ---
            exchange_halos(cart, U, local_nx, local_ny, left, right, down, up)

            val = np.zeros_like(U)
            val[1:-1, 1:-1] = 0.25 * (
                U[:-2, 1:-1] + U[2:, 1:-1] +
                U[1:-1, :-2] + U[1:-1, 2:] +
                (dx * dx) * f_local[1:-1, 1:-1]
            )

            U[mask_red] = omega * val[mask_red] + (1 - omega) * U[mask_red]

            # --- exchange between color phases ---
            exchange_halos(cart, U, local_nx, local_ny, left, right, down, up)

            # --- BLACK phase ---
            val[1:-1, 1:-1] = 0.25 * (
                U[:-2, 1:-1] + U[2:, 1:-1] +
                U[1:-1, :-2] + U[1:-1, 2:] +
                (dx * dx) * f_local[1:-1, 1:-1]
            )

            U[mask_black] = omega * val[mask_black] + (1 - omega) * U[mask_black]

            diff = np.abs(U - Uold)
            max_eabs_loc = np.max(diff)
            max_erel_loc = np.max(diff / (np.abs(U) + 1e-10))
        # === FASE 3: VERIFICAÇÃO DE CONVERGÊNCIA ===
        max_eabs_glob = comm.allreduce(max_eabs_loc, op=MPI.MAX)
        max_erel_glob = comm.allreduce(max_erel_loc, op=MPI.MAX)
        
        print(f"{max_eabs_glob}, {max_erel_glob}")
        #print(U)
        if max_eabs_glob < tol:
            final_error = max_eabs_glob
            break
        if max_erel_glob < 1e-6:
            final_error = max_erel_glob
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    exec_time = time.perf_counter() - t0
    comm_time_total = sum([c[-1] for c in comm_log])
    overhead = comm_time_total / exec_time if exec_time > 0 else 0.0
    

    U_local_data = U[1:-1, 1:-1].flatten().tolist()

    meta = dict(
        rank=rank,
        start_x=start_x, end_x=end_x,
        start_y=start_y, end_y=end_y,
        local_nx=local_nx, local_ny=local_ny,
        iterations=global_iteration,  # CORRIGIDO: era 'iteration'
        exec_time=exec_time,
        comm_time=comm_time_total,
        overhead=overhead,
        final_error=final_error,
        U_data = U_local_data,
        data_length=len(U_local_data)
    )
    
    U_local = U[1:-1, 1:-1].copy()  # Dados sem guard cells
    domain_info = (start_x, end_x, start_y, end_y, U_local)
    return meta, comm_log, domain_info

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--nx", type=int, required=True, help="Número de processos na dimensão X")
    parser.add_argument("--ny", type=int, required=True, help="Número de processos na dimensão Y")
    parser.add_argument("--max_iter", type=int, default=100000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--local_iters", type=int, default=1, 
                       help="Número de iterações locais entre comunicações")
    parser.add_argument("--omega", type=float, default=1.0,
                       help="Fator de relaxação (1.0 = Jacobi padrão)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    nx = args.nx
    ny = args.ny
    N = args.N

    # CORRIGIDO: local_iters -> block_size
    meta, comm_log, sol_local = jacobi_mpi_cart(args.omega,
        args.N, args.nx, args.ny, 
        max_iter=args.max_iter, 
        tol=args.tol, 
        block_size=args.local_iters,
    )

    # junta metadados e logs em rank 0
    all_meta = comm.gather(meta, root=0)
    all_logs = comm.gather(comm_log, root=0)
    all_solutions = comm.gather(sol_local, root=0)
    if rank == 0:
        rows = []
        for m, logs in zip(all_meta, all_logs):
            rows.append(m)
            for entry in logs:
                rows.append(dict(
                    rank=entry[1],
                    iteration=entry[0],
                    neighbor=entry[2],
                    direction=entry[3],
                    n_points=entry[4],
                    comm_time=entry[5]
                ))
        df = pd.DataFrame(rows)
        caminho_csv = os.path.join(f'output/results_{args.nx}x{args.ny}_{args.N}.csv')
        df.to_csv(caminho_csv, index=False)
        print(f"[rank 0] Arquivo results_{args.nx}x{args.ny}.csv salvo.")
        
        # solução global
        U_global = np.zeros((args.N, args.N))
        for (start_x, end_x, start_y, end_y, U_local) in all_solutions:
            U_global[start_x:end_x, start_y:end_y] = U_local

        # Salva em formato NumPy
        np.save(f"solution_{nx}x{ny}_{N}.npy", U_global)
if __name__ == "__main__":
    main()
