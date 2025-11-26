import numpy as np
from scipy.sparse import diags, kron as spkron, eye as speye
from scipy.sparse.linalg import spsolve
import time
import matplotlib.pyplot as plt

def solve_heat_2d_finite_difference():
    ################# Geração da malha ####################
    xl, xr, yb, yt = 0, 1, 0, 1
    N = 201
    X = np.linspace(xl, xr, N, endpoint=True)
    Y = np.linspace(yb, yt, N, endpoint=True)
    x, y = np.meshgrid(X, Y, indexing='ij')
    dx = (xr - xl) / (N - 1)
    dy = (yt - yb) / (N - 1)
    
    print(f"Malha: {N}x{N}, dx={dx:.4f}, dy={dy:.4f}")
    
    ##### Parâmetros temporais #####
    dt = 0.0001
    n_time_steps = 50  # Aumentei para ver a evolução
    alpha = 1.0
    
    ##### Criação da Matriz U #####
    U = np.zeros((N, N))
    
    # Condição inicial - VETORIZADO (muito mais eficiente)
    def initial_condition(x, y):
        center_x, center_y = 0.5, 0.5
        radius = 0.2
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        return np.where(dist <= radius, 1.0 - (dist / radius)**2, 0.0)
    
    # Aplicar condição inicial VETORIZADA
    U = initial_condition(x, y)
    
<<<<<<< HEAD
=======
    # Condições de contorno de Dirichlet
>>>>>>> 8abc333789f8ca35c60905fcca5580f94a97abd4
    def apply_boundary_conditions(U):
        U[0, :] = 0.0   # Left
        U[-1, :] = 0.0  # Right  
        U[:, 0] = 0.0   # Bottom
        U[:, -1] = 0.0  # Top
        return U
    
    U = apply_boundary_conditions(U)
    
<<<<<<< HEAD
    
    def kron(A, B):
        return np.kron(A, B)
    
    def create_system_matrix(N, dx, dy, dt, alpha):
        """Cria a matriz do sistema para o método implícito"""
        main_diag = (1 + 2*alpha*dt/dx**2 + 2*alpha*dt/dy**2) * np.ones(N)
        off_diag = (-alpha*dt/dx**2) * np.ones(N-1)
=======
    ##### Método implícito COM MATRIZ ESPARSA #####
    
    def create_sparse_system_matrix(N, dx, dy, dt, alpha):
        """Cria a matriz do sistema ESPARSA"""
        total_points = N * N
>>>>>>> 8abc333789f8ca35c60905fcca5580f94a97abd4
        
        # Coeficientes
        cx = alpha * dt / dx**2
        cy = alpha * dt / dy**2
        center_coeff = 1.0 + 2*cx + 2*cy
        
        # Listas para construir matriz esparsa
        data = []
        rows = []
        cols = []
        
<<<<<<< HEAD
        # Matriz S para acoplamento na direção y
        main_diag_y = np.zeros(N)
        off_diag_y = (-alpha*dt/dy**2) * np.ones(N-1)
        S = np.diag(main_diag_y) + np.diag(off_diag_y, 1) + np.diag(off_diag_y, -1)
        
        A = kron(I, T) + kron(S, I)
        
        # Ajustar condições de contorno na matriz
=======
>>>>>>> 8abc333789f8ca35c60905fcca5580f94a97abd4
        for i in range(N):
            for j in range(N):
                idx = i * N + j
                
                # Condições de contorno
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    # Equação: U = 0
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                else:
                    # Ponto interior - estêncil de 5 pontos
                    rows.append(idx)
                    cols.append(idx)
                    data.append(center_coeff)  # Centro
                    
                    # Vizinho esquerdo (i-1, j)
                    rows.append(idx)
                    cols.append((i-1) * N + j)
                    data.append(-cx)
                    
                    # Vizinho direito (i+1, j)
                    rows.append(idx)
                    cols.append((i+1) * N + j)
                    data.append(-cx)
                    
                    # Vizinho inferior (i, j-1)
                    rows.append(idx)
                    cols.append(i * N + (j-1))
                    data.append(-cy)
                    
                    # Vizinho superior (i, j+1)
                    rows.append(idx)
                    cols.append(i * N + (j+1))
                    data.append(-cy)
        
        # Criar matriz esparsa
        from scipy.sparse import csr_matrix
        A = csr_matrix((data, (rows, cols)), shape=(total_points, total_points))
        
        return A
    
<<<<<<< HEAD
    print("Montando matriz do sistema...")
=======
    # Criar matriz do sistema ESPARSA
    print("Montando matriz do sistema ESPARSA...")
>>>>>>> 8abc333789f8ca35c60905fcca5580f94a97abd4
    start_time = time.time()
    A = create_sparse_system_matrix(N, dx, dy, dt, alpha)
    matrix_time = time.time() - start_time
    print(f"Matriz esparsa montada em {matrix_time:.4f} segundos")
    print(f"Matriz: {A.shape[0]}x{A.shape[1]}, {A.nnz} elementos não-nulos")
    
    # Loop temporal
    print("Iniciando simulação temporal...")
    simulation_time = time.time()
    
    for step in range(n_time_steps):
        # Vetor do lado direito
        b = U.flatten()
        
        # Aplicar condições de contorno no vetor b
        for i in range(N):
            for j in range(N):
                if i == 0 or i == N-1 or j == 0 or j == N-1:
                    idx = i * N + j
                    b[idx] = 0.0
        
        # Resolver sistema linear COM MATRIZ ESPARSA
        U_flat = spsolve(A, b)
        U = U_flat.reshape(N, N)
        
        # Aplicar condições de contorno
        U = apply_boundary_conditions(U)
        
        # Mostrar progresso
        if step % 10 == 0:
            max_temp = np.max(U)
            min_temp = np.min(U)
            print(f"Step {step:3d}: max T = {max_temp:.6f}, min T = {min_temp:.6f}")
    
    total_time = time.time() - simulation_time
    print(f"Simulação concluída em {total_time:.4f} segundos")
    print(f"Tempo total: {matrix_time + total_time:.4f} segundos")
    
    return total_time + matrix_time

def compare_with_petsc():
    """Função para comparar com a solução PETSc"""
    print("\n" + "="*50)
    print("COMPARAÇÃO PETSc vs DIFERENÇAS FINITAS")
    print("="*50)
    
    # Tempo da solução por diferenças finitas
    fd_time = solve_heat_2d_finite_difference()
    
    print(f"\nTempo Diferenças Finitas: {fd_time:.4f} segundos")
    print("Para comparar com PETSc, execute:")
    print("mpirun -np 4 python petsc_unified.py")

if __name__ == "__main__":
    compare_with_petsc()
