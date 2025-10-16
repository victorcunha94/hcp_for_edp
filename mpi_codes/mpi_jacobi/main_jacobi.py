#!/usr/bin/env python3
"""
Jacobi paralelo (MPI cartesiano) com logging detalhado em CSV

Exemplo para rodar:
mpirun -np 4 python3 main_jacobi_iter.py --N 512 --nx 2 --ny 2 --local_iters 100
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
    for px in range(int(math.sqrt(size)), 0, -1):
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
    
    # Buffers de recebimento
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
        
        # === FASE 1: COMUNICAÇÃO (Isend/Irecv) ===
        requests = []
        t1_comm = time.perf_counter()

        # Buffers de Envio (necessários para slices não contíguos)
        send_buf_up = U[1:-1, local_ny].copy()
        send_buf_down = U[1:-1, 1].copy()
        # Buffers Direita/Esquerda são contíguos no default C-Order (em numpy), mas o slice vertical (y) é não-contíguo.
        # Por segurança, vamos manter a cópia em todos os buffers que não são linhas completas.
        # A otimização real aqui seria usar tipos derivados MPI ou reordenar o array para F-Order.

        # 1. Comunicação TOPO (Y - Cima) - Não contígua
        if up != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_up, dest=up, tag=0))
            requests.append(cart.Irecv(recv_buf_up, source=up, tag=1))

        # 2. Comunicação BAIXO (Y - Baixo) - Não contígua
        if down != MPI.PROC_NULL:
            requests.append(cart.Isend(send_buf_down, dest=down, tag=1))
            requests.append(cart.Irecv(recv_buf_down, source=down, tag=0))

        # 3. Comunicação DIREITA (X - Direita) - Contígua
        if right != MPI.PROC_NULL:
            requests.append(cart.Isend(U[local_nx, 1:-1], dest=right, tag=2))
            requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))

        # 4. Comunicação ESQUERDA (X - Esquerda) - Contígua
        if left != MPI.PROC_NULL:
            requests.append(cart.Isend(U[1, 1:-1], dest=left, tag=3))
            requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))


        # === FASE 2: CÁLCULO INTERNO (Sobreposição) ===
        # Não fazemos cálculo aqui para Jacobi com block_size > 1 para garantir correção numérica.


        # === FASE 3: SINCRONIZAÇÃO E APLICAÇÃO DOS DADOS RECEBIDOS ===
        MPI.Request.Waitall(requests)
        
        dt = time.perf_counter() - t1_comm
        
        # Aplica dados
        if up != MPI.PROC_NULL:
            U[1:-1, local_ny + 1] = recv_buf_up
            comm_log.append([global_iteration, rank, up, "up", recv_buf_up.size, dt])

        if down != MPI.PROC_NULL:
            U[1:-1, 0] = recv_buf_down
            comm_log.append([global_iteration, rank, down, "down", recv_buf_down.size, dt])

        if right != MPI.PROC_NULL:
            U[local_nx + 1, 1:-1] = recv_buf_right
            comm_log.append([global_iteration, rank, right, "right", recv_buf_right.size, dt])

        if left != MPI.PROC_NULL:
            U[0, 1:-1] = recv_buf_left
            comm_log.append([global_iteration, rank, left, "left", recv_buf_left.size, dt])

        # === FASE 4: BLOCO DE ITERAÇÕES LOCAIS (APÓS ATUALIZAÇÃO DOS HALOS) ===
        t0 = time.perf_counter()
        max_err_loc = 0.0

        for local_iter in range(block_size):
            global_iteration += 1

            # ATUALIZAÇÃO JACOBI VETORIZADA
            Unew[1:local_nx+1, 1:local_ny+1] = 0.25 * (
                U[0:local_nx, 1:local_ny+1] + 
                U[2:local_nx+2, 1:local_ny+1] + 
                U[1:local_nx+1, 0:local_ny] + 
                U[1:local_nx+1, 2:local_ny+2] + 
                (dx*dx) * f_local[1:local_nx+1, 1:local_ny+1]
            )
            
            # CÁLCULO DO ERRO VETORIZADO
            error_matrix = np.abs(Unew[1:local_nx+1, 1:local_ny+1] - U[1:local_nx+1, 1:local_ny+1])
            max_err_loc = np.max(error_matrix)
            
            # Trocar arrays para próxima iteração
            U, Unew = Unew, U
        
        exec_time += time.perf_counter() - t0

        # === FASE 5: VERIFICAÇÃO DE CONVERGÊNCIA (APÓS BLOCO) ===
        max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX)
        
        if max_err_glob < tol:
            final_error = max_err_glob
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    # Note: O loop de 'block_size' incrementa global_iteration.
    # O número total de iterações pode exceder max_iter.
    
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
    if rank == 0:
        print(f"Tempo de execução = {exec_time}|Overhead = {overhead}|")
    return meta, comm_log, U


def analytical(X, Y):
    return np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=50)
    parser.add_argument("--nx", type=int, required=True, help="Número de processos na dimensão X")
    parser.add_argument("--ny", type=int, required=True, help="Número de processos na dimensão Y")
    parser.add_argument("--max_iter", type=int, default=1000000)
    parser.add_argument("--tol", type=float, default=1e-10)
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
        if not os.path.exists('output'):
            os.makedirs('output')
        caminho_csv = os.path.join(f'output/results_{args.nx}x{args.ny}.csv')
        df.to_csv(caminho_csv, index=False)
        print(f"[rank 0] Arquivo {caminho_csv} salvo.")

        

if __name__ == "__main__":
    main()
