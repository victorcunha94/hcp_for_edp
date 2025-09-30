import numpy as np
import time

# Para uso com MPI real, descomente:
from mpi4py import MPI

def jacobi_mpi_cart(N, max_iter=10000, tol=1e-8):
    """
    Jacobi com decomposição CORRETA do domínio usando MPI Cartesiano
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # ========== CRIAÇÃO DO GRID CARTESIANO ==========
    dims = MPI.Compute_dims(size, 2)
    periods = [False, False]  # Não periódico para condições de Dirichlet
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=True)
    
    coords = cart_comm.Get_coords(rank)
    print(coords)
    
    # Obter ranks dos vizinhos
    left, right = cart_comm.Shift(0, 1)
    down, up = cart_comm.Shift(1, 1)
    
    # ========== DECOMPOSIÇÃO CORRETA DO DOMÍNIO ==========
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    
    # Dividir pontos INTERNOS entre processos (excluindo bordas globais)
    # Cada processo fica com um bloco de pontos internos
    points_x = N - 2  # Pontos internos na direção x
    points_y = N - 2  # Pontos internos na direção y
    
    # Número de pontos por processo em cada direção
    points_per_proc_x = points_x // dims[0]
    points_per_proc_y = points_y // dims[1]
    
    # Calcular índices GLOBAIS do subdomínio local (pontos internos)
    start_x = 1 + coords[0] * points_per_proc_x      # Primeiro ponto interno em x
    end_x = start_x + points_per_proc_x              # Último ponto interno em x
    
    start_y = 1 + coords[1] * points_per_proc_y      # Primeiro ponto interno em y  
    end_y = start_y + points_per_proc_y              # Último ponto interno em y
    
    # Ajustar último processo em cada direção para pegar pontos extras
    if coords[0] == dims[0] - 1:
        end_x = N - 1
    if coords[1] == dims[1] - 1:
        end_y = N - 1
    
    # Tamanho do subdomínio local (apenas pontos internos)
    local_nx = end_x - start_x
    local_ny = end_y - start_y
    
    if rank == 0:
        print(f"Decomposição: grid {dims[0]}x{dims[1]}")
        print(f"Pontos internos totais: {points_x}x{points_y}")
        print(f"Pontos por processo: ~{points_per_proc_x}x{points_per_proc_y}")
    
    # ========== ARRAYS LOCAIS COM HALOS ==========
    # Adicionar 1 camada de halo em cada lado
    U_local     = np.zeros((local_nx + 2, local_ny + 2))
    U_new_local = np.zeros((local_nx + 2, local_ny + 2))
    
    # Mapeamento: índice local -> índice global
    # U_local[0, :] = halo esquerdo (recebe de left)
    # U_local[-1, :] = halo direito (recebe de right)  
    # U_local[:, 0] = halo inferior (recebe de down)
    # U_local[:, -1] = halo superior (recebe de up)
    # U_local[1:-1, 1:-1] = pontos internos locais
    
    # ========== CONDIÇÕES DE CONTORNO GLOBAIS ==========
    # Apenas processos nas bordas aplicam condições aos pontos internos locais
    if coords[0] == 0:  # Processo na borda esquerda global
        # O primeiro ponto interno local está na borda global
        U_local[1, :] = 0.0  # Condição de Dirichlet
    
    if coords[0] == dims[0] - 1:  # Processo na borda direita global  
        # O último ponto interno local está na borda global
        U_local[-2, :] = 0.0
    
    if coords[1] == 0:  # Processo na borda inferior global
        U_local[:, 1] = 0.0
    
    if coords[1] == dims[1] - 1:  # Processo na borda superior global
        U_local[:, -2] = 0.0
    
    # ========== FUNÇÃO FONTE ==========
    def f_global(i_global, j_global):
        x = i_global * dx
        y = j_global * dy
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Pré-calcular f nos pontos internos locais
    for i_local in range(1, local_nx + 1):
        for j_local in range(1, local_ny + 1):
            i_global = start_x + (i_local - 1)  # -1 porque i_local=1 é o primeiro ponto interno
            j_global = start_y + (j_local - 1)
            U_local[i_local, j_local] = f_global(i_global, j_global)
    
    # ========== LOOP PRINCIPAL ==========
    comm.Barrier() # Barreira para esperar todos os processos concluírem os cálculos
    start_time = time.time()
    
    for iteration in range(max_iter):
        max_error_local = 0.0
        
        # ========== COMUNICAÇÃO DE HALOS ==========
        # Trocar apenas as fronteiras internas entre subdomínios
        
        # 1. COMUNICAÇÃO VERTICAL (y): cima/baixo
        if up != MPI.PROC_NULL:
            # Enviar linha superior interna para processo de cima
            send_up = U_local[1:-1, -2].copy()    # Penúltima coluna (interna)
            recv_up = np.empty(local_nx, dtype=np.float64)
            cart_comm.Sendrecv(send_up, up, recv_buffer=recv_up, source=down) # Entender o source=down
            U_local[1:-1, -1] = recv_up           # Preencher halo superior
        
        if down != MPI.PROC_NULL:
            # Enviar linha inferior interna para processo de baixo  
            send_down = U_local[1:-1, 1].copy()   # Segunda coluna (interna)
            recv_down = np.empty(local_nx, dtype=np.float64)
            cart_comm.Sendrecv(send_down, down, recv_buffer=recv_down, source=up)
            U_local[1:-1, 0] = recv_down          # Preencher halo inferior
        
        # 2. COMUNICAÇÃO HORIZONTAL (x): esquerda/direita
        if right != MPI.PROC_NULL:
            # Enviar coluna direita interna para processo da direita
            send_right = U_local[-2, 1:-1].copy() # Penúltima linha (interna)
            recv_right = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(send_right, right, recv_buffer=recv_right, source=left)
            U_local[-1, 1:-1] = recv_right        # Preencher halo direito
        
        if left != MPI.PROC_NULL:
            # Enviar coluna esquerda interna para processo da esquerda
            send_left = U_local[1, 1:-1].copy()   # Segunda linha (interna)
            recv_left = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(send_left, left, recv_buffer=recv_left, source=right)
            U_local[0, 1:-1] = recv_left          # Preencher halo esquerdo
        
        # ========== ATUALIZAÇÃO JACOBI ==========
        for i in range(1, local_nx + 1):
            for j in range(1, local_ny + 1):
                # Apenas atualizar pontos que não estão na borda global
                is_global_boundary = (
                    (coords[0] == 0 and i == 1) or          # Borda esquerda
                    (coords[0] == dims[0] - 1 and i == local_nx) or  # Borda direita
                    (coords[1] == 0 and j == 1) or          # Borda inferior
                    (coords[1] == dims[1] - 1 and j == local_ny)     # Borda superior
                )
                
                if not is_global_boundary:
                    U_new_local[i, j] = 0.25 * (
                        U_local[i-1, j] + U_local[i+1, j] + 
                        U_local[i, j-1] + U_local[i, j+1] - 
                        dx*dy * U_local[i, j]  # f(x,y) já está em U_local
                    )
                    
                    error = abs(U_new_local[i, j] - U_local[i, j])
                    max_error_local = max(max_error_local, error)
                else:
                    # Manter condições de contorno globais
                    U_new_local[i, j] = U_local[i, j]
        
        # Trocar arrays
        U_local, U_new_local = U_new_local, U_local
        
        # Verificar convergência
        max_error_global = comm.allreduce(max_error_local, op=MPI.MAX)
        
        if max_error_global < tol:
            if rank == 0:
                print(f"Convergido em {iteration} iterações")
            break
    
    execution_time = time.time() - start_time
    return U_local[1:-1, 1:-1], iteration + 1, execution_time, max_error_global


def jacobi_mpi_cart_sequential(N, max_iter=10000, tol=1e-8):
    """
    Versão sequencial do algoritmo MPI Cartesiano para estudo e debug
    Simula o comportamento de 1 processo MPI
    
    Args:
        N: Tamanho total da malha (N x N)
        max_iter: Número máximo de iterações
        tol: Tolerância para convergência
    
    Returns:
        U: Solução completa
        iterations: Número de iterações executadas
        execution_time: Tempo de execução
        max_error: Erro máximo final
    """
    print("=== MODO SEQUENCIAL (1 processo) ===")
    
    # ========== SIMULAÇÃO DE 1 PROCESSO MPI ==========
    dims = [1, 1]
    coords = [0, 0]
    rank = 0
    
    # ========== PARÂMETROS DO PROBLEMA ==========
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    
    # ========== INICIALIZAÇÃO ==========
    U = np.zeros((N, N))
    U_new = np.zeros((N, N))
    
    # Aplicar condições de contorno globais
    U[0, :] = 0.0    # Borda esquerda
    U[-1, :] = 0.0   # Borda direita
    U[:, 0] = 0.0    # Borda inferior
    U[:, -1] = 0.0   # Borda superior
    
    # ========== FUNÇÃO FONTE ==========
    def f_global(i, j):
        x = i * dx
        y = j * dy
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Pré-calcular f
    f_vals = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f_vals[i, j] = f_global(i, j)
    
    # ========== LOOP DE JACOBI ==========
    print(f"Iniciando Jacobi sequencial para malha {N}x{N}")
    start_time = time.time()
    
    for iteration in range(max_iter):
        max_error = 0.0
        
        # Atualizar pontos internos
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                U_new[i, j] = 0.25 * (
                    U[i-1, j] + U[i+1, j] + 
                    U[i, j-1] + U[i, j+1] - 
                    dx*dy * f_vals[i, j]
                )
                
                error = abs(U_new[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        # Trocar arrays para próxima iteração
        U, U_new = U_new, U
        
        # Verificar convergência
        if iteration % 100 == 0:
            print(f"Iteração {iteration}, erro: {max_error:.2e}")
        
        if max_error < tol:
            print(f"✓ Convergência alcançada na iteração {iteration}")
            break
    
    execution_time = time.time() - start_time
    
    print(f"Tempo total: {execution_time:.4f} segundos")
    print(f"Erro final: {max_error:.2e}")
    
    return U, iteration + 1, execution_time, max_error


def compare_sequential_vs_mpi_logic(N=50):
    """
    Compara a versão sequencial com uma simulação de lógica MPI
    Útil para entender o algoritmo antes de executar com MPI real
    """
    print("=" * 60)
    print("ESTUDO DO ALGORITMO MPI CARTESIANO")
    print("=" * 60)
    
    # Teste com versão sequencial
    print("\n1. EXECUTANDO VERSÃO SEQUENCIAL:")
    U_seq, iter_seq, time_seq, error_seq = jacobi_mpi_cart_sequential(N)
    
    print(f"\n2. RESUMO DO ALGORITMO:")
    print(f"   - Malha: {N} x {N}")
    print(f"   - Iterações: {iter_seq}")
    print(f"   - Tempo: {time_seq:.4f}s")
    print(f"   - Erro final: {error_seq:.2e}")
    
    print(f"\n3. SOLUÇÃO:")
    print(f"   - Forma da solução: {U_seq.shape}")
    print(f"   - Valor máximo: {np.max(U_seq):.6f}")
    print(f"   - Valor mínimo: {np.min(U_seq):.6f}")
    
    print(f"\n4. PARA EXECUTAR COM MPI REAL:")
    print(f"   mpirun -n 4 python script.py")
    print(f"   (onde 4 = 2x2 processos)")


if __name__ == "__main__":
    # Para testar sem MPI, use a versão sequencial
    compare_sequential_vs_mpi_logic(N=50)
    
    # Para executar com MPI real, descomente:
    # if MPI.COMM_WORLD.Get_size() > 1:
    #     U, iterations, time, error = jacobi_mpi_cart(100)
    # else:
    #     print("Execute com: mpirun -n 4 python script.py")
