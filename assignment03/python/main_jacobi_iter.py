#!/usr/bin/env python3
"""
Jacobi paralelo (MPI cartesiano) com logging detalhado em CSV

Exemplo para rodar:
mpirun -np 4 python3 main_jacobi_iter.py --N 50 --nx 2 --ny 2 
"""


from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import pandas as pd
import os
import matplotlib.pyplot as plt





def grid_dims(nx, ny, size):
    if nx * ny != size:
        raise ValueError(f"Erro: nx * ny = {nx} * {ny} = {nx * ny} != {size} (total de processos)")
    return [nx, ny]

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

def jacobi_mpi_cart(N, nx, ny, max_iter=10000, tol=1e-8, L=1.0, block_size=1):
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
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    def f_global(i, j):
        x = i * dx
        y = j * dx
        return 8.0 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

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
        
        # === FASE 1: COMUNICAÇÃO (ANTES do cálculo) ===
        # comunicação topo
        if up != MPI.PROC_NULL:
            buf = U[1:-1, local_ny].copy()
            recv_buf = np.empty(local_nx, dtype=np.float64)
            t1 = time.perf_counter()
            cart.Sendrecv(buf, dest=up, recvbuf=recv_buf, source=up)
            dt = time.perf_counter() - t1
            U[1:-1, local_ny + 1] = recv_buf
            comm_log.append([global_iteration, rank, up, "up", buf.size, dt])
        else:
            U[1:-1, local_ny + 1] = 0.0

        # comunicação baixo
        if down != MPI.PROC_NULL:
            buf = U[1:-1, 1].copy()
            recv_buf = np.empty(local_nx, dtype=np.float64)
            t1 = time.perf_counter()
            cart.Sendrecv(buf, dest=down, recvbuf=recv_buf, source=down)
            dt = time.perf_counter() - t1
            U[1:-1, 0] = recv_buf
            comm_log.append([global_iteration, rank, down, "down", buf.size, dt])
        else:
            U[1:-1, 0] = 0.0

        # comunicação direita
        if right != MPI.PROC_NULL:
            buf = U[local_nx, 1:-1].copy()
            recv_buf = np.empty(local_ny, dtype=np.float64)
            t1 = time.perf_counter()
            cart.Sendrecv(buf, dest=right, recvbuf=recv_buf, source=right)
            dt = time.perf_counter() - t1
            U[local_nx + 1, 1:-1] = recv_buf
            comm_log.append([global_iteration, rank, right, "right", buf.size, dt])
        else:
            U[local_nx + 1, 1:-1] = 0.0

        # comunicação esquerda
        if left != MPI.PROC_NULL:
            buf = U[1, 1:-1].copy()
            recv_buf = np.empty(local_ny, dtype=np.float64)
            t1 = time.perf_counter()
            cart.Sendrecv(buf, dest=left, recvbuf=recv_buf, source=left)
            dt = time.perf_counter() - t1
            U[0, 1:-1] = recv_buf
            comm_log.append([global_iteration, rank, left, "left", buf.size, dt])
        else:
            U[0, 1:-1] = 0.0

        # === FASE 2: BLOCO DE ITERAÇÕES LOCAIS ===
        for local_iter in range(block_size):
            # === ATUALIZAÇÃO JACOBI VETORIZADA ===
            # Aplicar a fórmula de Jacobi apenas nos pontos internos (excluindo bordas/halos)
            Unew[1:local_nx+1, 1:local_ny+1] = 0.25 * (
                U[0:local_nx, 1:local_ny+1] +     # esquerda
                U[2:local_nx+2, 1:local_ny+1] +   # direita  
                U[1:local_nx+1, 0:local_ny] +     # baixo
                U[1:local_nx+1, 2:local_ny+2] +   # cima
                (dx*dx) * f_local[1:local_nx+1, 1:local_ny+1]
            )
            
            # === CÁLCULO DO ERRO VETORIZADO ===
            # Calcular erro apenas nos pontos internos
            error_matrix = np.abs(Unew[1:local_nx+1, 1:local_ny+1] - U[1:local_nx+1, 1:local_ny+1])
            max_err_loc = np.max(error_matrix)
            
            # Trocar arrays para próxima iteração
            U, Unew = Unew, U

        # === FASE 3: VERIFICAÇÃO DE CONVERGÊNCIA ===
        max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)
        if rank == 0:  # Opcional: print apenas no rank 0 para evitar poluição
            print(f"Iteração {global_iteration}: Erro máximo = {max_err_glob}")

        if max_err_glob < tol:
            final_error = max_err_glob
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    exec_time = time.perf_counter() - t0
    comm_time_total = sum([c[-1] for c in comm_log])
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
        U_data = u_local_data,
        data_length=len(u_local_data)
    )
    
    return meta, comm_log, U



def analytical(X, Y):
    return np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--nx", type=int, required=True, help="Número de processos na dimensão X")
    parser.add_argument("--ny", type=int, required=True, help="Número de processos na dimensão Y")
    parser.add_argument("--max_iter", type=int, default=200000)
    parser.add_argument("--tol", type=float, default=1e-8)
    parser.add_argument("--local_iters", type=int, default=1, 
                       help="Número de iterações locais entre comunicações")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    
    meta, comm_log, U = jacobi_mpi_cart(
        args.N, args.nx, args.ny, 
        max_iter=args.max_iter, 
        tol=args.tol, 
        block_size=args.local_iters
    )
    #plot = plot_3d_comparison(U, args.N, args.nx, args.ny, analytical)

    # junta metadados e logs em rank 0
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
                    neighbor=entry[2],
                    direction=entry[3],
                    n_points=entry[4],
                    comm_time=entry[5]
                ))
        df = pd.DataFrame(rows)
        caminho_csv = os.path.join(f'output/results_{args.nx}x{args.ny}.csv')
        df.to_csv(caminho_csv, index=False)
        print(f"[rank 0] Arquivo results_{args.nx}x{args.ny}.csv salvo.")

if __name__ == "__main__":
    main()
