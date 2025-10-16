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
    exec_time=0
    comm.Barrier()
    final_error = None
    comm_log = []

    global_iteration = 0
    
   


# Assumindo que estas variáveis (local_nx, local_ny, up, down, left, right, cart)
# foram definidas anteriormente na função 'jacobi_mpi_cart'.
# local_nx = tamanho do subdomínio em X (dimensão 0)
# local_ny = tamanho do subdomínio em Y (dimensão 1)
# U = array numpy de tamanho (local_nx+2, local_ny+2)
# U_new = array numpy de tamanho (local_nx+2, local_ny+2)

# Definição dos buffers de recebimento fora do loop (melhor performance)
# Buffers para comunicação vertical (dimensão X)
    recv_buf_up   = np.empty(local_nx, dtype=np.float64) # Recebe de 'up'
    recv_buf_down = np.empty(local_nx, dtype=np.float64) # Recebe de 'down'

# Buffers para comunicação horizontal (dimensão Y)
# O mpi4py não se dá bem com slices não contíguos (U[1:-1, local_nx]), 
# então o buffer precisa ter o tamanho da borda local_ny
    recv_buf_right = np.empty(local_ny, dtype=np.float64) # Recebe de 'right'
    recv_buf_left  = np.empty(local_ny, dtype=np.float64)  # Recebe de 'left'


    while global_iteration < max_iter:
        global_iteration += 1
        max_err_loc = 0.0
        requests = []

        # === FASE 1: INÍCIO DA COMUNICAÇÃO NÃO BLOQUEANTE (Isend/Irecv) ===
        t1_comm = time.perf_counter()

        # --- 1. Comunicação TOPO (Y - Cima) ---
        if up != MPI.PROC_NULL:
            # Envia a borda superior (U[1:-1, local_ny]) para o vizinho 'up'
            send_buf_up = U[1:-1, local_ny].copy()
            # Tag 0: Convenção para "Borda Superior"
            requests.append(cart.Isend(send_buf_up, dest=up, tag=0))
            
            # Recebe do vizinho 'up' no ghost cell superior (U[1:-1, local_ny + 1])
            # O vizinho 'up' envia sua borda inferior, que usamos tag 1
            requests.append(cart.Irecv(recv_buf_up, source=up, tag=1))
            # Note: O log de comunicação será adicionado APÓS o Waitall para medir o tempo real.

        # --- 2. Comunicação BAIXO (Y - Baixo) ---
        if down != MPI.PROC_NULL:
            # Envia a borda inferior (U[1:-1, 1]) para o vizinho 'down'
            send_buf_down = U[1:-1, 1].copy()
            # Tag 1: Convenção para "Borda Inferior"
            # CORREÇÃO: dest=down
            requests.append(cart.Isend(send_buf_down, dest=down, tag=1))
            
            # Recebe do vizinho 'down' no ghost cell inferior (U[1:-1, 0])
            # O vizinho 'down' envia sua borda superior, que usamos tag 0
            # CORREÇÃO: source=down
            requests.append(cart.Irecv(recv_buf_down, source=down, tag=0))

        # --- 3. Comunicação DIREITA (X - Direita) ---
        if right != MPI.PROC_NULL:
            # Envia a borda direita (U[local_nx, 1:-1]) para o vizinho 'right'
            send_buf_right = U[local_nx, 1:-1].copy()
            # Tag 2: Convenção para "Borda Direita"
            # CORREÇÃO: dest=right
            requests.append(cart.Isend(send_buf_right, dest=right, tag=2))
            
            # Recebe do vizinho 'right' no ghost cell direito (U[local_nx + 1, 1:-1])
            # O vizinho 'right' envia sua borda esquerda, que usamos tag 3
            # CORREÇÃO: source=right
            requests.append(cart.Irecv(recv_buf_right, source=right, tag=3))

        # --- 4. Comunicação ESQUERDA (X - Esquerda) ---
        if left != MPI.PROC_NULL:
            # Envia a borda esquerda (U[1, 1:-1]) para o vizinho 'left'
            send_buf_left = U[1, 1:-1].copy()
            # Tag 3: Convenção para "Borda Esquerda"
            # CORREÇÃO: dest=left está correto
            requests.append(cart.Isend(send_buf_left, dest=left, tag=3))
            
            # Recebe do vizinho 'left' no ghost cell esquerdo (U[0, 1:-1])
            # O vizinho 'left' envia sua borda direita, que usamos tag 2
            # CORREÇÃO: source=left está correto
            requests.append(cart.Irecv(recv_buf_left, source=left, tag=2))


        # === FASE 2: CÁLCULO INTERNO (Onde a Comunicação se Sobrepõe) ===
        # Você deve colocar a maior parte do seu cálculo (apenas células internas, sem ghosts)
        # AQUI, entre o Isend/Irecv e o Waitall, para mascarar a latência da comunicação.
        
        # Exemplo: cálculo apenas nas células centrais que não dependem das ghosts
        # U_new[2:local_nx, 2:local_ny] = (U[1:local_nx-1, 2:local_ny] + ... ) / 4.0
        # O código exato do cálculo do Jacobi foi omitido, mas deve ser feito aqui.
        
        
        # === FASE 3: SINCRONIZAÇÃO E APLICAÇÃO DOS DADOS RECEBIDOS ===
        
        # Espera que todas as comunicações iniciadas (Isend e Irecv) terminem.
        MPI.Request.Waitall(requests)
        
        # Mede o tempo total de comunicação (incluindo sobreposição)
        dt = time.perf_counter() - t1_comm
        
        # --- 1. Aplica dados TOPO ---
        if up != MPI.PROC_NULL:
            # CORREÇÃO CRÍTICA: Atribui o buffer recebido (recv_buf_up) à linha ghost superior (local_ny + 1)
            U[1:-1, local_ny + 1] = recv_buf_up 
            # Log: (local_nx é a dimensão do buffer vertical)
            comm_log.append([global_iteration, rank, up, "up", recv_buf_up.size, dt])

        # --- 2. Aplica dados BAIXO ---
        if down != MPI.PROC_NULL:
            # CORREÇÃO CRÍTICA: Atribui o buffer recebido (recv_buf_down) à linha ghost inferior (0)
            U[1:-1, 0] = recv_buf_down
            # Log:
            comm_log.append([global_iteration, rank, down, "down", recv_buf_down.size, dt])

        # --- 3. Aplica dados DIREITA ---
        if right != MPI.PROC_NULL:
            # CORREÇÃO CRÍTICA: Atribui o buffer recebido (recv_buf_right) à coluna ghost direita (local_nx + 1)
            U[local_nx + 1, 1:-1] = recv_buf_right
            # Log: (local_ny é a dimensão do buffer horizontal)
            comm_log.append([global_iteration, rank, right, "right", recv_buf_right.size, dt])

        # --- 4. Aplica dados ESQUERDA ---
        if left != MPI.PROC_NULL:
            # CORREÇÃO CRÍTICA: Atribui o buffer recebido (recv_buf_left) à coluna ghost esquerda (0)
            U[0, 1:-1] = recv_buf_left
            # Log:
            comm_log.append([global_iteration, rank, left, "left", recv_buf_left.size, dt])

        # === FASE 4: CÁLCULO NAS BORDAS (AGORA com as células ghost atualizadas) ===
        # Agora, calcule as células do subdomínio que dependem das fronteiras.
        # U_new[1, 1:-1], U_new[local_nx, 1:-1], U_new[1:-1, 1], U_new[1:-1, local_ny]

        # ... (O restante do seu código do Jacobi, incluindo a troca U <-> U_new e o cálculo do erro) ...

        # === FASE 2: BLOCO DE ITERAÇÕES LOCAIS ===
        t0 = time.perf_counter()
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
        #if rank == 0:  # Opcional: print apenas no rank 0 para evitar poluição
            #print(f"Iteração {global_iteration}: Erro máximo = {max_err_glob}")
        exec_time += time.perf_counter() - t0
        if max_err_glob < tol:
            final_error = max_err_glob
            break

    if final_error is None:
        final_error = comm.allreduce(max_err_loc, op=MPI.MAX)

    #exec_time = time.perf_counter() - t0
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
#        print(f"[rank 0] Arquivo results_{args.nx}x{args.ny}.csv salvo.")

        

if __name__ == "__main__":
    main()
