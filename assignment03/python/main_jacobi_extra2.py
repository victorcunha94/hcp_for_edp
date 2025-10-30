#!/usr/bin/env python3

"""
mpirun -np 4 python3 main_jacobi_extra.py --N 514 --nx 2 --ny 2 
"""

from mpi4py import MPI
import numpy as np
import time
import argparse
import math
import pandas as pd
import os

def jacobi_mpi_cart_optimized_v1(N, nx, ny, max_iter=1000000, tol=1e-9, L=1.0, block_size=1):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. Configuração da Grade usando MPI_Dims_create ---
    if rank == 0:
        # Se nx e ny forem fornecidos como 0, usar MPI_Dims_create para determinar automaticamente
        if nx == 0 and ny == 0:
            # Criar array de dimensões inicializado com zeros
            dims = [0, 0]
            # MPI.Dims_create preenche o array com uma divisão balanceada
            MPI.Compute_dims(size, 2, dims)
            nx, ny = dims
            print(f"MPI_Dims_create: {size} processos -> {nx}x{ny}")
        else:
            # Verificar se nx * ny corresponde ao número de processos
            if nx * ny != size:
                print(f"ERRO: nx * ny = {nx} * {ny} = {nx * ny} != {size}")
                error_flag = True
            else:
                dims = [nx, ny]
                error_flag = False
    else:
        dims = None
        error_flag = None

    # Transmitir flag de erro e abortar se a configuração da grade estiver errada
    error_flag = comm.bcast(error_flag, root=0)
    if error_flag:
        if rank == 0:
            print("A configuração nx, ny não é válida para o número de processos.")
        comm.Abort(1)

    # Transmitir as dimensões da grade para todos os processos
    dims = comm.bcast(dims, root=0)
    nx, ny = dims

    # --- 2. Criar comunicador cartesiano ---
    if size == 1:
        # Modo sequencial
        cart = None
        coords = [0, 0]
        left, right, down, up = MPI.PROC_NULL, MPI.PROC_NULL, MPI.PROC_NULL, MPI.PROC_NULL
    else:
        # Criar comunicador cartesiano
        cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        coords = cart.Get_coords(rank)
        left, right = cart.Shift(0, 1)
        down, up = cart.Shift(1, 1)

    # --- 3. Partição usando MPI ---
    dx = L / (N - 1)
    dx2 = dx * dx
    
    # Dimensões globais do domínio (incluindo bordas)
    global_dims = [N, N]
    
    # Dimensões do domínio interno (sem bordas)
    interior_dims = [N - 2, N - 2]
    
    # Calcular partição
    if size == 1:
        # Caso sequencial - domínio completo
        start_x, end_x = 1, N - 1
        start_y, end_y = 1, N - 1
        local_nx = N - 2
        local_ny = N - 2
    else:
        # Para direção X
        sizes_x = [interior_dims[0]] * dims[0]
        subsizes_x = [0] * dims[0]
        starts_x = [0] * dims[0]
        
        base_x = interior_dims[0] // dims[0]
        rem_x = interior_dims[0] % dims[0]
        
        for i in range(dims[0]):
            if i < rem_x:
                subsizes_x[i] = base_x + 1
            else:
                subsizes_x[i] = base_x
            if i == 0:
                starts_x[i] = 0
            else:
                starts_x[i] = starts_x[i-1] + subsizes_x[i-1]
        
        # Para direção Y
        sizes_y = [interior_dims[1]] * dims[1]
        subsizes_y = [0] * dims[1]
        starts_y = [0] * dims[1]
        
        base_y = interior_dims[1] // dims[1]
        rem_y = interior_dims[1] % dims[1]
        
        for i in range(dims[1]):
            if i < rem_y:
                subsizes_y[i] = base_y + 1
            else:
                subsizes_y[i] = base_y
            if i == 0:
                starts_y[i] = 0
            else:
                starts_y[i] = starts_y[i-1] + subsizes_y[i-1]
        
        # Coordenadas atuais deste processo
        my_coord_x = coords[0]
        my_coord_y = coords[1]
        
        # Calcular índices locais
        start_x = starts_x[my_coord_x] + 1  # +1 para pular borda esquerda
        end_x = start_x + subsizes_x[my_coord_x]
        start_y = starts_y[my_coord_y] + 1  # +1 para pular borda inferior
        end_y = start_y + subsizes_y[my_coord_y]
        
        local_nx = subsizes_x[my_coord_x]
        local_ny = subsizes_y[my_coord_y]

    if local_nx == 0 or local_ny == 0:
        meta = dict(rank=rank, error="Empty subdomain")
        return meta, [], None

    # --- 4. Alocação e Inicialização ---
    # Alocar U e Unew de uma vez, incluindo as bordas fantasma (+2)
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    # Pré-calcular f_local
    i_global = np.arange(start_x, end_x).reshape(-1, 1)  # Colunas (x)
    j_global = np.arange(start_y, end_y).reshape(1, -1)  # Linhas (y)
    x = i_global * dx
    y = j_global * dx
    f_local[1:local_nx+1, 1:local_ny+1] = np.sin(4*np.pi * x) * np.sin(4*np.pi * y)

    # Buffers de comunicação (apenas se size > 1)
    if size > 1:
        send_buf_up = np.empty(local_nx, dtype=np.float64)
        send_buf_down = np.empty(local_nx, dtype=np.float64)
        send_buf_right = np.empty(local_ny, dtype=np.float64)
        send_buf_left = np.empty(local_ny, dtype=np.float64)

        recv_buf_up = np.empty(local_nx, dtype=np.float64)
        recv_buf_down = np.empty(local_nx, dtype=np.float64)
        recv_buf_right = np.empty(local_ny, dtype=np.float64)
        recv_buf_left = np.empty(local_ny, dtype=np.float64)

    # --- 5. Loop Principal de Jacobi (VERSÃO SIMPLIFICADA) ---
    exec_time = 0.0
    comm_time_total = 0.0
    final_error = None
    global_iteration = 0
    check_interval = 1  # Verificar convergência a cada 100 iterações

    comm.Barrier()
    t_total_start = time.perf_counter()

    while global_iteration < max_iter:
        
        # --- Comunicação a cada iteração (para consistência) ---
        if size > 1:
            t_comm_start = time.perf_counter()
            requests = []

            # Preparar buffers de envio
            np.copyto(send_buf_up, U[1:-1, -2])    # Linha abaixo da borda superior
            np.copyto(send_buf_down, U[1:-1, 1])   # Linha acima da borda inferior
            np.copyto(send_buf_right, U[-2, 1:-1]) # Coluna à esquerda da borda direita
            np.copyto(send_buf_left, U[1, 1:-1])   # Coluna à direita da borda esquerda

            # Postar Recebimentos
            if up != MPI.PROC_NULL:
                requests.append(cart.Irecv(recv_buf_up, source=up, tag=1)) 
            if down != MPI.PROC_NULL:
                requests.append(cart.Irecv(recv_buf_down, source=down, tag=0)) 
            if right != MPI.PROC_NULL:
                requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))
            if left != MPI.PROC_NULL:
                requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))

            # Postar Envios
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
        
        # --- Uma iteração por vez ---
        t_comp_start = time.perf_counter()
        
        # Kernel de Jacobi (vectorizado)
        Unew[1:-1, 1:-1] = 0.25 * (
            U[:-2, 1:-1] + U[2:, 1:-1] +
            U[1:-1, :-2] + U[1:-1, 2:] +
            dx2 * f_local[1:-1, 1:-1]
        )
        
        # Calcular erro
        current_err = np.max(np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1])) / np.max(np.abs(Unew[1:-1, 1:-1]))
        
        # Swap arrays
        U, Unew = Unew, U
        
        exec_time += time.perf_counter() - t_comp_start
        global_iteration += 1
        
        # --- Verificação de Convergência ---
        if global_iteration % check_interval == 0:
            max_err_glob = comm.allreduce(current_err, op=MPI.MAX) if size > 1 else current_err
            if max_err_glob < tol:
                final_error = max_err_glob
                break

    # --- Cálculo final do erro ---
    if final_error is None:
        # Recalcular erro da última iteração
        current_err = np.max(np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1])) / np.max(np.abs(Unew[1:-1, 1:-1]))
        final_error = comm.allreduce(current_err, op=MPI.MAX) if size > 1 else current_err

    # --- 6. Finalização ---
    t_total_end = time.perf_counter()
    total_time_local = t_total_end - t_total_start
    
    overhead = comm_time_total / total_time_local if total_time_local > 0 else 0.0

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
        U_data=u_local_data
    )

    # Reduções para estatísticas globais
    if size == 1:
        max_total_time = total_time_local
        max_exec_time = exec_time
        total_comm_time = comm_time_total
    else:
        max_total_time = comm.reduce(total_time_local, op=MPI.MAX, root=0)
        max_exec_time = comm.reduce(exec_time, op=MPI.MAX, root=0) 
        total_comm_time = comm.reduce(comm_time_total, op=MPI.SUM, root=0)

    if rank == 0:
        print("DEBUG: Depois das reduções")
        print(f"--------------------------------------------------------")
        print(f"Método de Jacobi Paralelo Concluído (MPI {nx}x{ny})")
        print(f"Iterações = {global_iteration}")
        print(f"Erro Final = {final_error:.8e}")
        print(f"Tempo Wall-Clock = {max_total_time:.4f}s")
        print(f"Tempo Máximo Computação = {max_exec_time:.4f}s") 
        print(f"Tempo Total Comunicação = {total_comm_time:.4f}s")
        if max_exec_time > 0:
            print(f"Overhead = {total_comm_time/max_exec_time:.4f}")
        print(f"--------------------------------------------------------")

    return meta, [], U


def jacobi_mpi_cart_optimized_v2(N, nx, ny, max_iter=1000000, tol=1e-9, L=1.0, block_size=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- dims: se nx,ny forem 0 use fatoração simples equilibrada ---
    if rank == 0:
        if nx == 0 or ny == 0:
            # escolhe um divisor próximo de sqrt(size)
            nx_tmp = int(np.floor(np.sqrt(size)))
            while nx_tmp > 0 and size % nx_tmp != 0:
                nx_tmp -= 1
            if nx_tmp == 0:
                nx_tmp = 1
            nx_root = nx_tmp
            ny_root = size // nx_root
            dims = [nx_root, ny_root]
        else:
            dims = [nx, ny]
            if dims[0] * dims[1] != size:
                print(f"ERRO: nx*ny ({dims[0]}x{dims[1]}) != nprocs ({size})")
                error_flag = True
            else:
                error_flag = False
    else:
        dims = None
        error_flag = None

    error_flag = comm.bcast(error_flag if 'error_flag' in locals() else False, root=0)
    if error_flag:
        comm.Abort(1)
    dims = comm.bcast(dims, root=0)
    nx, ny = dims

    # --- communicator/cartesian coords ---
    if size == 1:
        cart = None
        coords = (0, 0)
        left = right = down = up = MPI.PROC_NULL
    else:
        cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        coords = cart.Get_coords(rank)
        left, right = cart.Shift(0, 1)
        down, up = cart.Shift(1, 1)

    # --- partition (balanced block with remainder) ---
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

    # --- alloc & init ---
    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    i_global = np.arange(start_x, end_x).reshape(-1, 1)
    j_global = np.arange(start_y, end_y).reshape(1, -1)
    x = i_global * dx
    y = j_global * dx
    f_local[1:local_nx+1, 1:local_ny+1] = np.sin(4*np.pi * x) * np.sin(4*np.pi * y)

    if size > 1:
        send_up   = np.empty(local_nx, dtype=np.float64)
        send_down = np.empty(local_nx, dtype=np.float64)
        send_right= np.empty(local_ny, dtype=np.float64)
        send_left = np.empty(local_ny, dtype=np.float64)

        recv_up   = np.empty(local_nx, dtype=np.float64)
        recv_down = np.empty(local_nx, dtype=np.float64)
        recv_right= np.empty(local_ny, dtype=np.float64)
        recv_left = np.empty(local_ny, dtype=np.float64)

    exec_time = 0.0
    comm_time_total = 0.0
    global_iteration = 0
    check_interval = min(100, max(1, block_size * 5))
    final_error = None
    global_converged = False

    comm.Barrier()
    t_total_start = time.perf_counter()

    while global_iteration < max_iter and not global_converged:
        # --- halo exchange (only if parallel) ---
        if size > 1:
            t_comm_start = time.perf_counter()

            # fill send buffers
            np.copyto(send_up,   U[1:-1, -2])
            np.copyto(send_down, U[1:-1, 1])
            np.copyto(send_right,U[-2, 1:-1])
            np.copyto(send_left, U[1, 1:-1])

            # post all Irecv/Isend in a short list comprehension (None for PROC_NULL)
            reqs = []
            pairs = [
                (up,    cart.Irecv(recv_up,    source=up,    tag=1) if up != MPI.PROC_NULL else None,
                        cart.Isend(send_up,    dest=up,    tag=0) if up != MPI.PROC_NULL else None),
                (down,  cart.Irecv(recv_down,  source=down,  tag=0) if down != MPI.PROC_NULL else None,
                        cart.Isend(send_down,  dest=down,  tag=1) if down != MPI.PROC_NULL else None),
                (right, cart.Irecv(recv_right, source=right, tag=3) if right != MPI.PROC_NULL else None,
                        cart.Isend(send_right, dest=right, tag=2) if right != MPI.PROC_NULL else None),
                (left,  cart.Irecv(recv_left,  source=left,  tag=2) if left != MPI.PROC_NULL else None,
                        cart.Isend(send_left,   dest=left,  tag=3) if left != MPI.PROC_NULL else None),
            ]
            for _, recv_req, send_req in pairs:
                if recv_req is not None: reqs.append(recv_req)
            for _, recv_req, send_req in pairs:
                if send_req is not None: reqs.append(send_req)

            if reqs:
                MPI.Request.Waitall(reqs)

            # apply received halos
            if up != MPI.PROC_NULL:    U[1:-1, -1] = recv_up
            if down != MPI.PROC_NULL:  U[1:-1, 0]  = recv_down
            if right != MPI.PROC_NULL: U[-1, 1:-1] = recv_right
            if left != MPI.PROC_NULL:  U[0, 1:-1]  = recv_left

            comm_time_total += time.perf_counter() - t_comm_start

        # --- local block of Jacobi iterations ---
        t_comp_start = time.perf_counter()
        max_err_loc = 0.0

        for _ in range(block_size):
            if global_iteration >= max_iter:
                break
            Unew[1:-1, 1:-1] = 0.25 * (
                U[:-2, 1:-1] + U[2:, 1:-1] +
                U[1:-1, :-2] + U[1:-1, 2:] +
                dx2 * f_local[1:-1, 1:-1]
            )

            current_err = np.max(np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1]))
            denom = np.max(np.abs(Unew[1:-1, 1:-1]))
            current_err = (current_err / denom) if denom != 0 else current_err
            max_err_loc = max(max_err_loc, current_err)

            U, Unew = Unew, U
            global_iteration += 1

        exec_time += time.perf_counter() - t_comp_start

        # --- convergence check periodically ---
        if global_iteration % check_interval == 0:
            max_err_glob = comm.allreduce(max_err_loc, op=MPI.MAX) if size > 1 else max_err_loc
            if max_err_glob < tol:
                final_error = max_err_glob
                global_converged = True

    # final error if not converged earlier
    if final_error is None:
        last_err = np.max(np.abs(U[1:-1,1:-1] - Unew[1:-1,1:-1]))
        denom = np.max(np.abs(U[1:-1,1:-1]))
        last_err = (last_err / denom) if denom != 0 else last_err
        final_error = comm.allreduce(last_err, op=MPI.MAX) if size > 1 else last_err

    # timings & reductions
    t_total_end = time.perf_counter()
    total_time_local = t_total_end - t_total_start
    overhead = comm_time_total / total_time_local if total_time_local > 0 else 0.0

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
        U_data = U[1:-1,1:-1].flatten().tolist()
    )

    # global stats (root)
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
        print(f"Método de Jacobi Paralelo Concluído (MPI {nx}x{ny})")
        print(f"Iterações = {global_iteration}")
        print(f"Erro Final = {final_error:.8e}")
        print(f"Tempo Wall-Clock = {max_total_time:.4f}s")
        print(f"Tempo Máximo Computação = {max_exec_time:.4f}s")
        print(f"Tempo Total Comunicação = {total_comm_time:.4f}s")
        if max_exec_time > 0:
            print(f"Overhead = {total_comm_time/max_exec_time:.4f}")
        print("--------------------------------------------------------")

    return meta, [], U




def jacobi_mpi_cart_optimized(N, nx, ny, max_iter=1000000, tol=1e-9, L=1.0):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        if nx == 0 and ny == 0:
            dims = [0, 0]
            MPI.Compute_dims(size, 2, dims)
            nx, ny = dims
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
        comm.Abort(1)

    dims = comm.bcast(dims, root=0)
    nx, ny = dims

    if size == 1:
        cart = None
        coords = [0, 0]
        left = right = down = up = MPI.PROC_NULL
    else:
        cart = comm.Create_cart(dims, periods=[False, False], reorder=True)
        coords = cart.Get_coords(rank)
        left, right = cart.Shift(0, 1)
        down, up = cart.Shift(1, 1)

    dx = L / (N - 1)
    dx2 = dx * dx

    interior_x = N - 2
    interior_y = N - 2

    if size == 1:
        start_x, end_x = 1, N - 1
        start_y, end_y = 1, N - 1
        local_nx = N - 2
        local_ny = N - 2
    else:
        base_x, rem_x = divmod(interior_x, nx)
        base_y, rem_y = divmod(interior_y, ny)

        sizes_x = [base_x + 1 if i < rem_x else base_x for i in range(nx)]
        sizes_y = [base_y + 1 if i < rem_y else base_y for i in range(ny)]

        starts_x = np.cumsum([0] + sizes_x[:-1])
        starts_y = np.cumsum([0] + sizes_y[:-1])

        cx, cy = coords
        start_x = starts_x[cx] + 1
        end_x = start_x + sizes_x[cx]
        start_y = starts_y[cy] + 1
        end_y = start_y + sizes_y[cy]

        local_nx = sizes_x[cx]
        local_ny = sizes_y[cy]

    if local_nx == 0 or local_ny == 0:
        return dict(rank=rank, error="Empty subdomain"), [], None

    U = np.zeros((local_nx + 2, local_ny + 2), dtype=np.float64)
    Unew = np.zeros_like(U)
    f_local = np.zeros_like(U)

    i_global = np.arange(start_x, end_x).reshape(-1, 1)
    j_global = np.arange(start_y, end_y).reshape(1, -1)
    x = i_global * dx
    y = j_global * dx
    f_local[1:-1, 1:-1] = np.sin(4*np.pi * x) * np.sin(4*np.pi * y)

    if size > 1:
        send_up = np.empty(local_nx); recv_up = np.empty(local_nx)
        send_down = np.empty(local_nx); recv_down = np.empty(local_nx)
        send_right = np.empty(local_ny); recv_right = np.empty(local_ny)
        send_left = np.empty(local_ny); recv_left = np.empty(local_ny)

    comm_time_total = 0.0
    exec_time = 0.0
    global_iteration = 0

    comm.Barrier()
    t_total_start = time.perf_counter()

    while global_iteration < max_iter:

        if size > 1:
            t_comm = time.perf_counter()

            np.copyto(send_up, U[1:-1, -2])
            np.copyto(send_down, U[1:-1, 1])
            np.copyto(send_right, U[-2, 1:-1])
            np.copyto(send_left, U[1, 1:-1])

            req = []
            if up != MPI.PROC_NULL: req.append(cart.Irecv(recv_up, up, 1))
            if down != MPI.PROC_NULL: req.append(cart.Irecv(recv_down, down, 0))
            if right != MPI.PROC_NULL: req.append(cart.Irecv(recv_right, right, 3))
            if left != MPI.PROC_NULL: req.append(cart.Irecv(recv_left, left, 2))

            if up != MPI.PROC_NULL: req.append(cart.Isend(send_up, up, 0))
            if down != MPI.PROC_NULL: req.append(cart.Isend(send_down, down, 1))
            if right != MPI.PROC_NULL: req.append(cart.Isend(send_right, right, 2))
            if left != MPI.PROC_NULL: req.append(cart.Isend(send_left, left, 3))

            MPI.Request.Waitall(req)

            if up != MPI.PROC_NULL: U[1:-1, -1] = recv_up
            if down != MPI.PROC_NULL: U[1:-1, 0] = recv_down
            if right != MPI.PROC_NULL: U[-1, 1:-1] = recv_right
            if left != MPI.PROC_NULL: U[0, 1:-1] = recv_left

            comm_time_total += time.perf_counter() - t_comm

        t_comp = time.perf_counter()

        Unew[1:-1, 1:-1] = 0.25 * (
            U[:-2,1:-1] + U[2:,1:-1] +
            U[1:-1,:-2] + U[1:-1,2:] +
            dx2 * f_local[1:-1,1:-1]
        )

        diff_loc = np.max(np.abs(Unew[1:-1,1:-1] - U[1:-1,1:-1]))
        norm_loc = np.max(np.abs(Unew[1:-1,1:-1]))

        U, Unew = Unew, U

        exec_time += time.perf_counter() - t_comp
        global_iteration += 1

        diff = comm.allreduce(diff_loc, MPI.MAX) if size > 1 else diff_loc
        norm = comm.allreduce(norm_loc, MPI.MAX) if size > 1 else norm_loc
        err = diff / norm if norm != 0 else diff

        if err < tol:
            final_error = err
            break
    else:
        final_error = err

    comm.Barrier()
    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    # Overhead = tempo de comunicação / tempo de computação
    overhead = comm_time_total / exec_time if exec_time > 0 else 0.0

    U_data = U[1:-1, 1:-1].flatten().tolist()

    meta = dict(
        rank=rank,
        start_x=start_x, end_x=end_x,
        start_y=start_y, end_y=end_y,
        local_nx=local_nx, local_ny=local_ny,
        iterations=global_iteration,
        exec_time=exec_time,
        comm_time=comm_time_total,
        total_time=total_time,
        overhead=overhead,
        final_error=final_error,
        U_data=U_data
    )

    comm_log = dict(
        rank=rank,
        comm_time=comm_time_total,
        exec_time=exec_time,
        total_time=total_time,
        iterations=global_iteration
    )

    # Para imprimir a decomposição use nx, ny (dims) que você já tem
    if rank == 0:
        try:
            proc_x, proc_y = nx, ny
        except NameError:
            proc_x = proc_y = None

        print("\n--------------------------------------------------------")
        print(f"   Método de Jacobi Paralelo Concluído (MPI {proc_x}x{proc_y})")
        print(f"   Iterações: {global_iteration}")
        print(f"   Erro final: {final_error:.6e}")
        print("--------------------------------------------------------")
        print(f"   Tempo total:              {total_time:.10f} s")
        print(f"   Tempo de cálculo (comp):  {exec_time:.10f} s")
        print(f"   Tempo de comunicação:     {comm_time_total:.6f} s")
        print(f"   Overhead comm/comp:       {overhead:.6f}")
        print("--------------------------------------------------------\n")

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
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # === CHAMADA ATUALIZADA ===
    meta, comm_log, U = jacobi_mpi_cart_optimized(
        args.N, args.nx, args.ny,
        max_iter=args.max_iter,
        tol=args.tol
    )

    # Coleta metadados de todos os processos
    all_meta = comm.gather(meta, root=0)

    if rank == 0:

        if not os.path.exists('output'):
            os.makedirs('output')

        caminho_csv = f'output/results_{args.nx}x{args.ny}.csv'

        # Filtra processos com dados válidos
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
