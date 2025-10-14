import numpy as np
import time
import matplotlib.pyplot as plt



################# Geração da malha ####################
xl, xr, yb, yt = 0, 1, 0, 1
N = 100  # Aumentado para ver melhor as diferenças
X = np.linspace(xl, xr, N, endpoint=True)
Y = np.linspace(yb, yt, N, endpoint=True)
x, y = np.meshgrid(X, Y, indexing='ij')
dx = (xr - xl) / (N - 1)
dy = (yt - yb) / (N - 1)
#######################################################


# Solução analítica
def f(x, y):
    return np.sin(2*np.pi * x) * np.sin(2*np.pi * y)


# Função fonte
def f(x, y):
    return -2 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)


# Solução analítica para comparação
def exact_solution(x, y):
    return np.sin(2*np.pi * x) * np.sin(2*np.pi * y)


######### MÉTODO: Jacobi Básico ##########
def jacobi_basic(N, dx, dy, max_iter=10000, tol=1e-8):
    """Método de Jacobi básico"""
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Aplicar condições de contorno de Dirichlet
    U[0, :]  = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0]  = exact_solution(X, Y[0])
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



######### MÉTODO: Jacobi com Relaxamento (Omega) ##########
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



# Executar os métodos
print("Executando Jacobi Básico...")
U_basic, iter_basic, time_basic, error_basic = jacobi_basic(N, dx, dy)

print("Executando Jacobi com Relaxamento...")
U_omega, iter_omega, time_omega, error_omega = jacobi_omega(N, dx, dy, omega=1.8)

# Calcular solução analítica
U_exact = exact_solution(x, y)

# Calcular erros em relação à solução analítica
error_analytical_basic = np.abs(U_basic - U_exact)
error_analytical_omega = np.abs(U_omega - U_exact)

# Estatísticas dos erros
max_error_basic = np.max(error_analytical_basic)
max_error_omega = np.max(error_analytical_omega)
rmse_basic = np.sqrt(np.mean(error_analytical_basic**2))
rmse_omega = np.sqrt(np.mean(error_analytical_omega**2))

print("\n=== COMPARAÇÃO DOS MÉTODOS ===")
print(f"Jacobi Básico:")
print(f"  Iterações: {iter_basic}")
print(f"  Tempo: {time_basic:.6f}s")
print(f"  Erro máximo (analítico): {max_error_basic:.2e}")
print(f"  RMSE (analítico): {rmse_basic:.2e}")

print(f"\nJacobi com Relaxamento (ω=1.8):")
print(f"  Iterações: {iter_omega}")
print(f"  Tempo: {time_omega:.6f}s")
print(f"  Erro máximo (analítico): {max_error_omega:.2e}")
print(f"  RMSE (analítico): {rmse_omega:.2e}")

# Visualização comparativa
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Solução Jacobi Básico
im1 = axes[0,0].imshow(U_basic, extent=[0,1,0,1], origin='lower')
axes[0,0].set_title('Jacobi Básico - Solução Numérica')
plt.colorbar(im1, ax=axes[0,0])

# Solução Jacobi com Relaxamento
im2 = axes[0,1].imshow(U_omega, extent=[0,1,0,1], origin='lower')
axes[0,1].set_title('Jacobi com Relaxamento - Solução Numérica')
plt.colorbar(im2, ax=axes[0,1])

# Solução Analítica
im3 = axes[0,2].imshow(U_exact, extent=[0,1,0,1], origin='lower')
axes[0,2].set_title('Solução Analítica')
plt.colorbar(im3, ax=axes[0,2])

# Erros
im4 = axes[1,0].imshow(error_analytical_basic, extent=[0,1,0,1], origin='lower')
axes[1,0].set_title(f'Erro Jacobi Básico\nMax: {max_error_basic:.2e}')
plt.colorbar(im4, ax=axes[1,0])

im5 = axes[1,1].imshow(error_analytical_omega, extent=[0,1,0,1], origin='lower')
axes[1,1].set_title(f'Erro Jacobi Relaxamento\nMax: {max_error_omega:.2e}')
plt.colorbar(im5, ax=axes[1,1])

# Diferença entre os dois métodos
diff_methods = np.abs(U_basic - U_omega)
im6 = axes[1,2].imshow(diff_methods, extent=[0,1,0,1], origin='lower')
axes[1,2].set_title('Diferença entre Métodos')
plt.colorbar(im6, ax=axes[1,2])

plt.tight_layout()
plt.show()

# Gráfico de convergência (perfil em y=0.5)
y_idx = N // 2
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(X, U_basic[:, y_idx], 'b-', label='Jacobi Básico', linewidth=2)
plt.plot(X, U_omega[:, y_idx], 'r--', label='Jacobi Relaxamento', linewidth=2)
plt.plot(X, U_exact[:, y_idx], 'k:', label='Solução Analítica', linewidth=3)
plt.xlabel('x')
plt.ylabel('u(x, y=0.5)')
plt.title('Perfil em y = 0.5')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(X, error_analytical_basic[:, y_idx], 'b-', label='Erro Jacobi Básico')
plt.plot(X, error_analytical_omega[:, y_idx], 'r--', label='Erro Jacobi Relaxamento')
plt.xlabel('x')
plt.ylabel('Erro Absoluto')
plt.title('Erro em y = 0.5')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



