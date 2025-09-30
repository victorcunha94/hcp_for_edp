import numpy as np
import time


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
def jacobi_omega(N, dx, dy, omega=1.8, max_iter=10000, tol=1e-8):
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

