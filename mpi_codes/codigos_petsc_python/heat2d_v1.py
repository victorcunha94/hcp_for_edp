import numpy as np
from numpy.linalg import solve
import time
import matplotlib.pyplot as plt

def solve_heat_2d_finite_difference():
    ################# Geração da malha ####################
    xl, xr, yb, yt = 0, 1, 0, 1  # Corrigido: yb (bottom), yt (top)
    N = 129  # Aumentei para um problema mais realista
    X = np.linspace(xl, xr, N, endpoint=True)
    Y = np.linspace(yb, yt, N, endpoint=True)
    x, y = np.meshgrid(X, Y, indexing='ij')
    dx = (xr - xl) / (N - 1)  # Corrigido
    dy = (yt - yb) / (N - 1)  # Corrigido
    
    print(f"Malha: {N}x{N}, dx={dx:.4f}, dy={dy:.4f}")
    
    ##### Parâmetros temporais #####
    dt = 0.0001  # Passo temporal
    n_time_steps = 5
    alpha = 1.0  # Coeficiente de difusão térmica
    
    ##### Criação da Matriz U #####
    U = np.zeros((N, N))
    
    # Condição inicial - hot spot no centro
    def initial_condition(x, y):
        center_x, center_y = 0.5, 0.5
        radius = 0.2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return np.where(dist <= radius, 1.0 - (dist / radius)**2, 0.0)
    
    # Aplicar condição inicial
    for i in range(N):
        for j in range(N):
            U[i, j] = initial_condition(X[i], Y[j])
    
    # Condições de contorno de Dirichlet (T = 0 nas bordas)
    def apply_boundary_conditions(U):
        U[0, :] = 0.0   # Left
        U[-1, :] = 0.0  # Right  
        U[:, 0] = 0.0   # Bottom
        U[:, -1] = 0.0  # Top
        return U
    
    U = apply_boundary_conditions(U)
    
    ##### Método implícito  #####
    
    def kron(A, B):
        """Produto de Kronecker"""
        return np.kron(A, B)  # Usando a função do numpy
    
    def create_system_matrix(N, dx, dy, dt, alpha):
        """Cria a matriz do sistema para o método implícito"""
        # Matriz T (operador 1D na direção x)
        main_diag = (1 + 2*alpha*dt/dx**2 + 2*alpha*dt/dy**2) * np.ones(N)
        off_diag = (-alpha*dt/dx**2) * np.ones(N-1)
        
        T = np.diag(main_diag) + np.diag(off_diag, 1) + np.diag(off_diag, -1)
        
        # Matriz identidade
        I = np.eye(N)
        
        # Matriz S para acoplamento na direção y
        main_diag_y = np.zeros(N)
        off_diag_y = (-alpha*dt/dy**2) * np.ones(N-1)
        S = np.diag(main_diag_y) + np.diag(off_diag_y, 1) + np.diag(off_diag_y, -1)
        
        # Matriz do sistema: A = I ⊗ T + S ⊗ I
        A = kron(I, T) + kron(S, I)
        
        # Ajustar condições de contorno na matriz
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    # Ponto de contorno - equação identidade
                    A[idx, :] = 0
                    A[idx, idx] = 1
        
        return A
    
    # Criar matriz do sistema uma vez (é a mesma para todos os passos temporais)
    print("Montando matriz do sistema...")
    start_time = time.time()
    A = create_system_matrix(N, dx, dy, dt, alpha)
    matrix_time = time.time() - start_time
    print(f"Matriz montada em {matrix_time:.4f} segundos")
    
    # Armazenar soluções para visualização
    solutions = [U.copy()]
    
    # Loop temporal
    print("Iniciando simulação temporal...")
    simulation_time = time.time()
    
    for step in range(n_time_steps):
        # Vetor do lado direito
        b = U.flatten()
        
        # Aplicar condições de contorno no vetor b
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    b[idx] = 0.0
        
        # Resolver sistema linear
        U_flat = solve(A, b)
        U = U_flat.reshape(N, N)
        
        # Aplicar condições de contorno (por segurança)
        U = apply_boundary_conditions(U)
        
        # Armazenar a cada 10 passos
        if step % 10 == 0:
            solutions.append(U.copy())
            max_temp = np.max(U)
            min_temp = np.min(U)
            print(f"Step {step:3d}: max T = {max_temp:.6f}, min T = {min_temp:.6f}")
    
    total_time = time.time() - simulation_time
    print(f"Simulação concluída em {total_time:.4f} segundos")
    print(f"Tempo total (matriz + simulação): {matrix_time + total_time:.4f} segundos")
    
    # Visualização
    plot_solutions(solutions, X, Y, n_time_steps)
    
    return total_time + matrix_time

def plot_solutions(solutions, X, Y, n_time_steps):
    """Plot das soluções em diferentes tempos"""
    n_plots = min(5, len(solutions))
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 4))
    
    if n_plots == 1:
        axes = [axes]
    
    for i in range(n_plots):
        idx = i * (len(solutions) // n_plots)
        if idx >= len(solutions):
            idx = len(solutions) - 1
            
        im = axes[i].imshow(solutions[idx].T, extent=[0, 1, 0, 1], origin='lower', 
                           cmap='hot', vmin=0, vmax=1)
        axes[i].set_title(f'Step {idx * (n_time_steps // len(solutions))}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        plt.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig('finite_difference_solution.png', dpi=150, bbox_inches='tight')
    plt.show()

def compare_with_petsc():
    """Função para comparar com a solução PETSc"""
    print("\n" + "="*50)
    print("COMPARAÇÃO PETSc vs DIFERENÇAS FINITAS")
    print("="*50)
    
    # Tempo da solução por diferenças finitas
    fd_time = solve_heat_2d_finite_difference()
    
    print(f"\nTempo Diferenças Finitas: {fd_time:.4f} segundos")
    print("Para comparar com PETSc, execute:")
    print("mpirun -np 4 python petsc_heat2d.py -ksp_monitor")
    print("\nMétricas de comparação:")
    print("1. Tempo de execução")
    print("2. Precisão da solução") 
    print("3. Escalabilidade (para malhas maiores)")
    print("4. Uso de memória")

if __name__ == "__main__":
    compare_with_petsc()
