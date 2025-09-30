# poisson_mpi_analytics_fixed.py
import numpy as np
import time
import csv
import os
from mpi4py import MPI

class MPIAnalytics:
    def __init__(self, comm, rank, coords):
        self.comm = comm
        self.rank = rank
        self.coords = coords
        self.communication_time = 0.0
        self.computation_time = 0.0
        self.communication_count = 0
        self.shared_points = []
        
    def log_communication(self, direction, points_sent, points_received, comm_time):
        self.communication_time += comm_time
        self.communication_count += 1
        self.shared_points.append({
            'iteration': self.communication_count,
            'direction': direction,
            'points_sent': points_sent,
            'points_received': points_received,
            'time': comm_time
        })

def jacobi_mpi_with_analytics(N, max_iter=10000, tol=1e-8, output_file="mpi_performance.csv"):
    """
    Jacobi MPI com métricas detalhadas - VERSÃO COMPLETAMENTE CORRIGIDA
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Setup do grid cartesiano
    dims = MPI.Compute_dims(size, 2)
    cart_comm = comm.Create_cart(dims, periods=[False, False], reorder=True)
    coords = cart_comm.Get_coords(rank)
    
    # Sistema de analytics
    analytics = MPIAnalytics(comm, rank, coords)
    
    # Obter vizinhos
    left, right = cart_comm.Shift(0, 1)
    down, up = cart_comm.Shift(1, 1)
    
    if rank == 0:
        print(f"=== MPI JACOBI ANALYTICS ===")
        print(f"Grid: {dims[0]}x{dims[1]}, Malha: {N}x{N}")
    
    # Parâmetros do problema
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    
    # Decomposição do domínio
    points_x = N - 2  # Pontos internos na direção x
    points_y = N - 2  # Pontos internos na direção y
    
    points_per_proc_x = points_x // dims[0]
    points_per_proc_y = points_y // dims[1]
    
    # Calcular subdomínio local
    start_x = 1 + coords[0] * points_per_proc_x
    end_x = start_x + points_per_proc_x
    start_y = 1 + coords[1] * points_per_proc_y
    end_y = start_y + points_per_proc_y
    
    if coords[0] == dims[0] - 1:
        end_x = N - 1
    if coords[1] == dims[1] - 1:
        end_y = N - 1
    
    local_nx = end_x - start_x
    local_ny = end_y - start_y
    
    if rank == 0:
        print(f"Processo {rank}: domínio ({start_x},{start_y})->({end_x},{end_y}), tamanho: {local_nx}x{local_ny}")
    
    # Arrays locais com halos
    U_local = np.zeros((local_nx + 2, local_ny + 2))
    U_new_local = np.zeros((local_nx + 2, local_ny + 2))
    
    # Condições de contorno - APENAS para processos nas bordas globais
    if coords[0] == 0:  # Borda esquerda global
        U_local[1, 1:-1] = 0.0
    if coords[0] == dims[0] - 1:  # Borda direita global
        U_local[-2, 1:-1] = 0.0
    if coords[1] == 0:  # Borda inferior global
        U_local[1:-1, 1] = 0.0
    if coords[1] == dims[1] - 1:  # Borda superior global
        U_local[1:-1, -2] = 0.0
    
    # Função fonte
    def f_global(i_global, j_global):
        x = i_global * dx
        y = j_global * dy
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Pré-calcular f nos pontos internos
    for i_local in range(1, local_nx + 1):
        for j_local in range(1, local_ny + 1):
            i_global = start_x + (i_local - 1)
            j_global = start_y + (j_local - 1)
            U_local[i_local, j_local] = f_global(i_global, j_global)
    
    # Iniciar loop principal
    comm.Barrier()
    total_start_time = time.time()
    
    for iteration in range(max_iter):
        max_error_local = 0.0
        
        # COMUNICAÇÃO - SINTAXE CORRIGIDA COM ARGUMENTOS NOMEADOS
        comm_start = time.time()
        
        # Comunicação vertical: cima/baixo
        if up != MPI.PROC_NULL:
            send_up = U_local[1:-1, -2].copy()  # Linha para enviar para cima
            recv_up = np.empty(local_nx, dtype=np.float64)
            # CORREÇÃO: usar sendbuf= e recvbuf= explicitamente
            cart_comm.Sendrecv(sendbuf=send_up, dest=up, recvbuf=recv_up, source=down)
            U_local[1:-1, -1] = recv_up  # Preencher halo superior
            analytics.log_communication('up', len(send_up), len(recv_up), time.time() - comm_start)
        
        if down != MPI.PROC_NULL:
            comm_start = time.time()
            send_down = U_local[1:-1, 1].copy()  # Linha para enviar para baixo
            recv_down = np.empty(local_nx, dtype=np.float64)
            cart_comm.Sendrecv(sendbuf=send_down, dest=down, recvbuf=recv_down, source=up)
            U_local[1:-1, 0] = recv_down  # Preencher halo inferior
            analytics.log_communication('down', len(send_down), len(recv_down), time.time() - comm_start)
        
        # Comunicação horizontal: esquerda/direita
        if right != MPI.PROC_NULL:
            comm_start = time.time()
            send_right = U_local[-2, 1:-1].copy()  # Coluna para enviar para direita
            recv_right = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(sendbuf=send_right, dest=right, recvbuf=recv_right, source=left)
            U_local[-1, 1:-1] = recv_right  # Preencher halo direito
            analytics.log_communication('right', len(send_right), len(recv_right), time.time() - comm_start)
        
        if left != MPI.PROC_NULL:
            comm_start = time.time()
            send_left = U_local[1, 1:-1].copy()  # Coluna para enviar para esquerda
            recv_left = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(sendbuf=send_left, dest=left, recvbuf=recv_left, source=right)
            U_local[0, 1:-1] = recv_left  # Preencher halo esquerdo
            analytics.log_communication('left', len(send_left), len(recv_left), time.time() - comm_start)
        
        # COMPUTAÇÃO
        comp_start = time.time()
        for i in range(1, local_nx + 1):
            for j in range(1, local_ny + 1):
                # Verificar se é ponto de contorno global
                is_global_boundary = (
                    (coords[0] == 0 and i == 1) or
                    (coords[0] == dims[0] - 1 and i == local_nx) or
                    (coords[1] == 0 and j == 1) or
                    (coords[1] == dims[1] - 1 and j == local_ny)
                )
                
                if not is_global_boundary:
                    U_new_local[i, j] = 0.25 * (
                        U_local[i-1, j] + U_local[i+1, j] + 
                        U_local[i, j-1] + U_local[i, j+1] - 
                        dx*dy * U_local[i, j]
                    )
                    error = abs(U_new_local[i, j] - U_local[i, j])
                    max_error_local = max(max_error_local, error)
                else:
                    # Manter condições de contorno
                    U_new_local[i, j] = U_local[i, j]
        
        analytics.computation_time += time.time() - comp_start
        
        # Trocar arrays para próxima iteração
        U_local, U_new_local = U_new_local, U_local
        
        # Verificar convergência
        max_error_global = comm.allreduce(max_error_local, op=MPI.MAX)
        
        if iteration % 100 == 0 and rank == 0:
            print(f"Iteração {iteration}, erro: {max_error_global:.2e}")
        
        if max_error_global < tol:
            if rank == 0:
                print(f"✓ Convergência alcançada na iteração {iteration}")
            break
    
    total_time = time.time() - total_start_time
    
    # COLETAR MÉTRICAS
    metrics = {
        'process_id': rank,
        'coords_x': coords[0],
        'coords_y': coords[1],
        'total_processes': size,
        'grid_size': N,
        'iterations': iteration + 1,
        'total_time': total_time,
        'computation_time': analytics.computation_time,
        'communication_time': analytics.communication_time,
        'communication_count': analytics.communication_count,
        'local_domain_size': f"{local_nx}x{local_ny}",
        'domain_start': f"({start_x},{start_y})",
        'domain_end': f"({end_x},{end_y})",
        'points_calculated': local_nx * local_ny,
        'points_shared': sum([comm['points_sent'] for comm in analytics.shared_points]),
        'final_error': max_error_global
    }
    
    # Salvar métricas em CSV - EVITAR ESCRITA CONCORRENTE
    if rank == 0:
        # Criar arquivo com header
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            writer.writeheader()
    
    comm.Barrier()  # Esperar processo 0 criar o arquivo
    
    # Coletar todas as métricas no processo 0 usando MPI
    all_metrics = comm.gather(metrics, root=0)
    
    if rank == 0:
        # Escrever todas as métricas
        with open(output_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            for metric in all_metrics:
                writer.writerow(metric)
        print(f"✓ Métricas salvas em {output_file}")
    
    # Salvar detalhes de comunicação por processo
    comm_details_file = f"comm_details_{rank}.csv"
    with open(comm_details_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['iteration', 'direction', 'points_sent', 'points_received', 'time'])
        writer.writeheader()
        for comm in analytics.shared_points:
            writer.writerow(comm)
    
    if rank == 0:
        print(f"✓ Detalhes de comunicação salvos em comm_details_*.csv")
        print(f"✓ Tempo total: {total_time:.4f}s, Iterações: {iteration + 1}")
    
    return metrics

def run_detailed_test():
    """Executa teste detalhado com diferentes configurações"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"\nExecutando com {size} processos...")
    
    # Teste com malha média para bom balanceamento
    if size == 1:
        N = 100
    elif size == 2:
        N = 100  
    elif size == 4:
        N = 100
    elif size == 8:
        N = 200
    else:
        N = 200
    
    metrics = jacobi_mpi_with_analytics(N, max_iter=1000, tol=1e-6)
    
    return metrics

if __name__ == "__main__":
    run_detailed_test()
