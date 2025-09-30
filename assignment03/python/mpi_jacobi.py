import numpy as np
from mpi4py import MPI
import time
import numpy as np
from scipy.sparse import diags, kron, eye
from scipy.sparse.linalg import spsolve
import time
from numba import jit, prange

################# Geração da malha ####################
xl, xr, yb, yt = 0, 1, 0, 1
N = 50  # Aumentado para ver melhor as diferenças
X = np.linspace(xl, xr, N, endpoint=True)
Y = np.linspace(yb, yt, N, endpoint=True)
x, y = np.meshgrid(X, Y, indexing='ij')
dx = (xr - xl) / (N - 1)
dy = (yt - yb) / (N - 1)
#######################################################

# Função fonte
def f(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Solução analítica para comparação
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

######### MÉTODO 1: Jacobi Básico ##########
def jacobi_basic(N, dx, dy, max_iter=10000, tol=1e-8):
    """Método de Jacobi básico"""
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Aplicar condições de contorno de Dirichlet
    U[0, :] = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0] = exact_solution(X, Y[0])
    U[:, -1] = exact_solution(X, Y[-1])
    
    Unew = U.copy()
    
    start_time = time.time()
    for k in range(max_iter):
        max_error = 0
        
        # Atualizar pontos internos
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                Unew[i, j] = 0.25 * (U[i-1, j] + U[i+1, j] + 
                                    U[i, j-1] + U[i, j+1] - 
                                    dx*dy * f(X[i], Y[j]))
                
                error = abs(Unew[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        # Verificar convergência
        if max_error < tol:
            break
            
        # Atualizar para próxima iteração
        U[:, :] = Unew[:, :]
    
    execution_time = time.time() - start_time
    return U, k + 1, execution_time, max_error

######### MÉTODO 2: Jacobi com Relaxamento (Omega) ##########
def jacobi_omega(N, dx, dy, omega=1.2, max_iter=10000, tol=1e-8):
    """Método de Jacobi com fator de relaxamento omega"""
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Aplicar condições de contorno
    U[0, :] = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0] = exact_solution(X, Y[0])
    U[:, -1] = exact_solution(X, Y[-1])
    
    Unew = U.copy()
    
    start_time = time.time()
    for k in range(max_iter):
        max_error = 0
        
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Cálculo do novo valor
                new_val = 0.25 * (U[i-1, j] + U[i+1, j] + 
                                 U[i, j-1] + U[i, j+1] - 
                                 dx*dy * f(X[i], Y[j]))
                
                # Aplicar relaxamento
                Unew[i, j] = omega * new_val + (1 - omega) * U[i, j]
                
                error = abs(Unew[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        if max_error < tol:
            break
            
        U[:, :] = Unew[:, :]
    
    execution_time = time.time() - start_time
    return U, k + 1, execution_time, max_error

######### MÉTODO 3: Jacobi Paralelizado ##########
@jit(nopython=True, parallel=True)
def jacobi_parallel_inner(U, Unew, X, Y, dx, dy, f_values):
    """Função interna paralelizada com Numba"""
    N = U.shape[0]
    max_error = 0.0
    
    for i in prange(1, N - 1):
        for j in range(1, N - 1):
            new_val = 0.25 * (U[i-1, j] + U[i+1, j] + 
                             U[i, j-1] + U[i, j+1] - 
                             dx*dy * f_values[i, j])
            
            Unew[i, j] = new_val
            error = abs(new_val - U[i, j])
            if error > max_error:
                max_error = error
                
    return max_error

def jacobi_parallel(N, dx, dy, max_iter=10000, tol=1e-8):
    """Método de Jacobi paralelizado"""
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Pré-calcular valores da função f
    f_values = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            f_values[i, j] = f(X[i], Y[j])
    
    # Aplicar condições de contorno
    U[0, :] = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0] = exact_solution(X, Y[0])
    U[:, -1] = exact_solution(X, Y[-1])
    
    Unew = U.copy()
    
    start_time = time.time()
    for k in range(max_iter):
        max_error = jacobi_parallel_inner(U, Unew, X, Y, dx, dy, f_values)
        
        if max_error < tol:
            break
            
        U[:, :] = Unew[:, :]
    
    execution_time = time.time() - start_time
    return U, k + 1, execution_time, max_error

######### MÉTODO 4: Solução Direta com Kronecker (Referência) ##########
def solve_kronecker_direct(N, dx, dy):
    """Solução direta usando produto de Kronecker - como referência"""
    # Matriz 1D do operador segunda derivada
    main_diag = -2 * np.ones(N-2)
    off_diag = np.ones(N-3)
    
    D2 = diags([off_diag, main_diag, off_diag], [-1, 0, 1], 
               shape=(N-2, N-2)) / (dx**2)
    
    I = eye(N-2)
    
    # Matriz Laplaciano 2D: ∇² = I ⊗ D2 + D2 ⊗ I
    A = kron(I, D2) + kron(D2, I)
    
    # Vetor fonte (apenas pontos internos)
    F_inner = np.zeros((N-2, N-2))
    for i in range(N-2):
        for j in range(N-2):
            F_inner[i, j] = f(X[i+1], Y[j+1])
    
    F_flat = F_inner.flatten()
    
    # Resolver sistema
    start_time = time.time()
    u_flat = spsolve(A, F_flat)
    execution_time = time.time() - start_time
    
    u_inner = u_flat.reshape((N-2, N-2))
    
    # Montar solução completa
    u_full = np.zeros((N, N))
    u_full[1:-1, 1:-1] = u_inner
    
    # Aplicar condições de contorno
    u_full[0, :] = exact_solution(X[0], Y)
    u_full[-1, :] = exact_solution(X[-1], Y)
    u_full[:, 0] = exact_solution(X, Y[0])
    u_full[:, -1] = exact_solution(X, Y[-1])
    
    return u_full, execution_time



def jacobi_mpi_cart(N, max_iter=10000, tol=1e-8):
    """
    Jacobi paralelizado usando MPI Cartesiano para decomposição 2D
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Criar grid cartesiano 2D
    dims = MPI.Compute_dims(size, 2)
    periods = [True, True]  # Condições periódicas (ou False para Dirichlet)
    reorder = True
    
    # Criar comunicador cartesiano
    cart_comm = comm.Create_cart(dims, periods=periods, reorder=reorder)
    
    # Obter coordenadas deste processo no grid
    coords = cart_comm.Get_coords(rank)
    
    # Obter ranks dos vizinhos
    left, right = cart_comm.Shift(0, 1)   # Direção x
    down, up = cart_comm.Shift(1, 1)      # Direção y
    
    # Parâmetros do problema
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    
    # Determinar subdomínio local
    # Dividir pontos internos entre processos
    points_per_proc_x = N // dims[0]
    points_per_proc_y = N // dims[1]
    
    # Calcular índices locais (incluindo halos)
    start_x = coords[0] * points_per_proc_x
    end_x = start_x + points_per_proc_x + 2  # +2 para halos
    start_y = coords[1] * points_per_proc_y  
    end_y = start_y + points_per_proc_y + 2  # +2 para halos
    
    # Ajustar para processos nas bordas
    if coords[0] == dims[0] - 1:
        end_x = N
    if coords[1] == dims[1] - 1:
        end_y = N
    
    # Tamanho local do subdomínio
    local_nx = end_x - start_x
    local_ny = end_y - start_y
    
    # Criar arrays locais
    U_local = np.zeros((local_nx, local_ny))
    U_new_local = np.zeros((local_nx, local_ny))
    
    # Aplicar condições de contorno globais (apenas processos de borda)
    if coords[0] == 0:  # Borda esquerda
        U_local[0, :] = 0.0
    if coords[0] == dims[0] - 1:  # Borda direita
        U_local[-1, :] = 0.0
    if coords[1] == 0:  # Borda inferior
        U_local[:, 0] = 0.0
    if coords[1] == dims[1] - 1:  # Borda superior
        U_local[:, -1] = 0.0
    
    # Função fonte local
    def f_local(i, j):
        x = (start_x + i) * dx
        y = (start_y + j) * dy
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    # Pré-calcular valores de f
    f_vals = np.zeros((local_nx, local_ny))
    for i in range(1, local_nx - 1):
        for j in range(1, local_ny - 1):
            f_vals[i, j] = f_local(i, j)
    
    # Sincronizar todos os processos
    comm.Barrier()
    start_time = time.time()
    
    for iteration in range(max_iter):
        max_error_local = 0.0
        
        # Trocar dados de halo com vizinhos
        # Enviar/recever bordas para cima/baixo
        if up != MPI.PROC_NULL:
            send_buffer = U_local[:, -2].copy()
            recv_buffer = np.empty(local_nx, dtype=np.float64)
            cart_comm.Sendrecv(send_buffer, up, recv_buffer=recv_buffer, source=down)
            U_local[:, -1] = recv_buffer
        
        if down != MPI.PROC_NULL:
            send_buffer = U_local[:, 1].copy()
            recv_buffer = np.empty(local_nx, dtype=np.float64)
            cart_comm.Sendrecv(send_buffer, down, recv_buffer=recv_buffer, source=up)
            U_local[:, 0] = recv_buffer
        
        # Enviar/receber bordas para esquerda/direita
        if right != MPI.PROC_NULL:
            send_buffer = U_local[-2, :].copy()
            recv_buffer = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(send_buffer, right, recv_buffer=recv_buffer, source=left)
            U_local[-1, :] = recv_buffer
        
        if left != MPI.PROC_NULL:
            send_buffer = U_local[1, :].copy()
            recv_buffer = np.empty(local_ny, dtype=np.float64)
            cart_comm.Sendrecv(send_buffer, left, recv_buffer=recv_buffer, source=right)
            U_local[0, :] = recv_buffer
        
        # Atualizar pontos internos
        for i in range(1, local_nx - 1):
            for j in range(1, local_ny - 1):
                U_new_local[i, j] = 0.25 * (U_local[i-1, j] + U_local[i+1, j] + 
                                           U_local[i, j-1] + U_local[i, j+1] - 
                                           dx*dy * f_vals[i, j])
                
                error = abs(U_new_local[i, j] - U_local[i, j])
                if error > max_error_local:
                    max_error_local = error
        
        # Trocar U_local e U_new_local
        U_local, U_new_local = U_new_local, U_local
        
        # Reduzir erro máximo global
        max_error_global = comm.allreduce(max_error_local, op=MPI.MAX)
        
        if max_error_global < tol:
            if rank == 0:
                print(f"MPI Convergiu em {iteration} iterações")
            break
    
    execution_time = time.time() - start_time
    
    # Coletar resultados em processo 0
    if rank == 0:
        U_global = np.zeros((N, N))
    else:
        U_global = None
    
    # Cada processo envia seu subdomínio para o processo 0
    # (Implementação simplificada - na prática use Gatherv)
    
    return U_local, iteration + 1, execution_time, max_error_global

# Versão simplificada para teste sem MPI
def jacobi_mpi_cart_sequential(N, max_iter=10000, tol=1e-8):
    """
    Versão sequencial para testar a lógica do algoritmo MPI
    """
    # Simular 1 processo
    dims = [1, 1]
    coords = [0, 0]
    
    # Parâmetros
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    
    # Domínio completo
    U = np.zeros((N, N))
    U_new = np.zeros((N, N))
    
    # Condições de contorno
    U[0, :] = 0.0    # Esquerda
    U[-1, :] = 0.0   # Direita
    U[:, 0] = 0.0    # Inferior
    U[:, -1] = 0.0   # Superior
    
    # Função fonte
    def f_func(i, j):
        x = i * dx
        y = j * dy
        return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)
    
    start_time = time.time()
    
    for iteration in range(max_iter):
        max_error = 0.0
        
        # Atualizar pontos internos
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                U_new[i, j] = 0.25 * (U[i-1, j] + U[i+1, j] + 
                                     U[i, j-1] + U[i, j+1] - 
                                     dx*dy * f_func(i, j))
                
                error = abs(U_new[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        U, U_new = U_new, U
        
        if max_error < tol:
            print(f"MPI Sequential convergiu em {iteration} iterações")
            break
    
    execution_time = time.time() - start_time
    return U, iteration + 1, execution_time, max_error

# Função de comparação atualizada
def compare_all_methods():
    """Compara todos os métodos incluindo MPI"""
    print(f"=== COMPARAÇÃO COMPLETA DE MÉTODOS - Malha {N}x{N} ===")
    
    # Solução exata
    def exact_solution(x, y):
        return np.sin(np.pi * x) * np.sin(np.pi * y)
    
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    U_exact = exact_solution(X, Y)
    
    dx = 1.0 / (N - 1)
    dy = dx
    
    # Testar métodos disponíveis
    results = {}
    
    # Jacobi Básico
    U_jacobi, iter_jacobi, time_jacobi, error_jacobi = jacobi_basic(N, dx, dy)
    error_exact_jacobi = np.max(np.abs(U_jacobi - U_exact))
    results['jacobi'] = (iter_jacobi, time_jacobi, error_jacobi, error_exact_jacobi)
    
    # Jacobi Omega
    U_omega, iter_omega, time_omega, error_omega = jacobi_omega(N, dx, dy, omega=1.5)
    error_exact_omega = np.max(np.abs(U_omega - U_exact))
    results['omega'] = (iter_omega, time_omega, error_omega, error_exact_omega)
    
    # Jacobi Paralelizado (Numba)
    U_parallel, iter_parallel, time_parallel, error_parallel = jacobi_parallel(N, dx, dy)
    error_exact_parallel = np.max(np.abs(U_parallel - U_exact))
    results['parallel'] = (iter_parallel, time_parallel, error_parallel, error_exact_parallel)
    
    # MPI (versão sequencial para teste)
    U_mpi, iter_mpi, time_mpi, error_mpi = jacobi_mpi_cart_sequential(N)
    error_exact_mpi = np.max(np.abs(U_mpi - U_exact))
    results['mpi'] = (iter_mpi, time_mpi, error_mpi, error_exact_mpi)
    
    # Solução Direta
    U_direct, time_direct = solve_kronecker_direct(N, dx, dy)
    error_exact_direct = np.max(np.abs(U_direct - U_exact))
    results['direct'] = (0, time_direct, 0, error_exact_direct)
    
    # Resultados
    print("\nMÉTODO               | ITERAÇÕES | TEMPO (s) | ERRO FINAL | ERRO vs EXATA")
    print("-" * 80)
    for method, (iters, time_val, error, error_exact) in results.items():
        method_name = method.upper().ljust(18)
        if method == 'direct':
            print(f"{method_name} | {'-':9} | {time_val:8.4f}  | {'-':10} | {error_exact:.2e}")
        else:
            print(f"{method_name} | {iters:9d} | {time_val:8.4f}  | {error:.2e} | {error_exact:.2e}")
    
    return results

# Para executar com MPI real:
def main_mpi():
    """
    Função principal para execução com MPI
    Executar com: mpirun -n 4 python script.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print("Iniciando Jacobi MPI com decomposição cartesiana...")
    
    N = 100  # Tamanho da malha
    U, iterations, time, error = jacobi_mpi_cart(N)
    
    if rank == 0:
        print(f"MPI Finalizado: {iterations} iterações, tempo: {time:.4f}s, erro: {error:.2e}")

if __name__ == "__main__":
    # Teste sequencial
    N = 50
    results = compare_all_methods()
    
    # Para executar com MPI, use:
    # mpirun -n 4 python seu_script.py
