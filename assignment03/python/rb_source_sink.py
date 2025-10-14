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
    # send/recv left-right
    if right != MPI.PROC_NULL:
        sendbuf = U[local_nx, 1:-1].copy()
        recvbuf = np.empty(local_ny, dtype=np.float64)
        cart.Sendrecv(sendbuf, dest=right, recvbuf=recvbuf, source=right)
        U[local_nx+1, 1:-1] = recvbuf
    else:
        U[local_nx+1, 1:-1] = 0.0
    if left != MPI.PROC_NULL:
        sendbuf = U[1, 1:-1].copy()
        recvbuf = np.empty(local_ny, dtype=np.float64)
        cart.Sendrecv(sendbuf, dest=left, recvbuf=recvbuf, source=left)
        U[0, 1:-1] = recvbuf
    else:
        U[0, 1:-1] = 0.0
    # send/recv up-down
    if up != MPI.PROC_NULL:
        sendbuf = U[1:-1, local_ny].copy()
        recvbuf = np.empty(local_nx, dtype=np.float64)
        cart.Sendrecv(sendbuf, dest=up, recvbuf=recvbuf, source=up)
        U[1:-1, local_ny+1] = recvbuf
    else:
        U[1:-1, local_ny+1] = 0.0
    if down != MPI.PROC_NULL:
        sendbuf = U[1:-1, 1].copy()
        recvbuf = np.empty(local_nx, dtype=np.float64)
        cart.Sendrecv(sendbuf, dest=down, recvbuf=recvbuf, source=down)
        U[1:-1, 0] = recvbuf
    else:
        U[1:-1, 0] = 0.0

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
        return 2*(1-6*x*x)*y*(1-y) + 2*(1-6*y*y)*x*(1-x)

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
    
    while global_iteration < max_iter:
        global_iteration += 1
        max_err_loc = 0.0
        max_eabs_loc = 0.0
        max_erel_loc = 0.0
        Uold[:,:] = U[:,:]
        
        # === FASE 1: COMUNICAÇÃO (ANTES do cálculo) ===
        exchange_halos(cart, U, local_nx, local_ny, left, right, down, up)

        # === FASE 2: BLOCO DE ITERAÇÕES LOCAIS ===
        for local_iter in range(block_size):
            # Vermelho
            for i in range(1, local_nx + 1):
                i_global = start_x + (i - 1)
                for j in range(1, local_ny + 1):
                    j_global = start_y + (j - 1)
                    if (i_global + j_global) % 2 == 0 and not (i_global in (0, N - 1) or j_global in (0, N - 1)):
                        val = 0.25 * (
                            U[i-1, j] + U[i+1, j] + U[i, j-1] + U[i, j+1]
                            - (dx*dx) * f_local[i, j]
                        )
                        U[i,j] = (omega * val) + (1 - omega) * U[i, j]

                        err1, err2 = errors(U[i,j], Uold[i,j])
                        max_eabs_loc = max(err1,max_eabs_loc)
                        max_erel_loc = max(err2,max_erel_loc)
                    
            exchange_halos(cart, U, local_nx, local_ny, left, right, down, up)
            
            # Preto
            for i in range(1, local_nx + 1):
                i_global = start_x + (i - 1)
                for j in range(1, local_ny + 1):
                    j_global = start_y + (j - 1)
                    if (i_global + j_global) % 2 == 1 and not (i_global in (0, N - 1) or j_global in (0, N - 1)):
                        val = 0.25 * (
                            U[i-1, j] + U[i+1, j] + U[i, j-1] + U[i, j+1]
                            - (dx*dx) * f_local[i, j]
                        )
                        U[i,j] = (omega * val) + (1 - omega) * U[i, j]

                        err1, err2 = errors(U[i,j], Uold[i,j])
                        max_eabs_loc = max(err1,max_eabs_loc)
                        max_erel_loc = max(err2,max_erel_loc)

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
        U = U,
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
        #
        # solução global
        U_global = np.zeros((args.N, args.N))
        for (start_x, end_x, start_y, end_y, U_local) in all_solutions:
            U_global[start_x:end_x, start_y:end_y] = U_local

        # Salva em formato NumPy
        np.save("solution.npy", U_global)
if __name__ == "__main__":
    main()
