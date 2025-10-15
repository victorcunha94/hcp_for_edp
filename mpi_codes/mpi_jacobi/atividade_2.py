import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt

# Código para resolver o problema de Poisson bidimensional
# em um domínio retângular [a,b] x [c,d] sujeito à
# condições de contorno de tipo Dirichlet.

def norm1_func(f, dx, dy):
    return np.sum(np.abs(f)) * dx*dy

def norm2_func(f, dx, dy):
    f_abs = np.abs(f)
    return np.sqrt( np.sum(f_abs*f_abs) * dx*dy )

def normM_func(f):
    return np.max(np.abs(f))

# =========================================================
# Função para resolver o problema usando o método de Gauss-Seidel
# =========================================================
def solver(U, F, dx, dy, eps, kMax):
    (M, N) = np.shape(U)
    M, N = M-1, N-1
    # Variaveis para armazenar 1/dx^2 e 1/dy^2
    one_dx2 = 1 / dx**2
    one_dy2 = 1 / dy**2
    U0 = np.copy(U)
    for k in range(1, kMax):
        # for i in range(1, M):
        #     for j in range(1, N):
        #         U[i, j] = 0.5 * ( (U[i+1, j] + U[i-1, j]) * one_dx2 + (U[i, j+1] + U[i, j-1]) * one_dy2 - F[i,j] ) / (one_dx2 + one_dy2)

        # Faz a iteração usando a forma vetorizada

        # Itera sobre os pontos tipo red
        i, j = 1, 1
        U[i:M:2, j:N:2] = 0.5 * ( (U[i+1:M+1:2, j:N:2] + U[i-1:M-1:2, j:N:2]) * one_dx2 + (U[i:M:2, j+1:N+1:2] + U[i:M:2, j-1:N-1:2]) * one_dy2 - F[i:M:2, j:N:2] ) / (one_dx2 + one_dy2)
        i, j = 2, 2
        U[i:M:2, j:N:2] = 0.5 * ( (U[i+1:M+1:2, j:N:2] + U[i-1:M-1:2, j:N:2]) * one_dx2 + (U[i:M:2, j+1:N+1:2] + U[i:M:2, j-1:N-1:2]) * one_dy2 - F[i:M:2, j:N:2] ) / (one_dx2 + one_dy2)

        # Itera sobre os pontos tipo black
        i, j = 2, 1
        U[i:M:2, j:N:2] = 0.5 * ( (U[i+1:M+1:2, j:N:2] + U[i-1:M-1:2, j:N:2]) * one_dx2 + (U[i:M:2, j+1:N+1:2] + U[i:M:2, j-1:N-1:2]) * one_dy2 - F[i:M:2, j:N:2] ) / (one_dx2 + one_dy2)
        i, j = 1, 2
        U[i:M:2, j:N:2] = 0.5 * ( (U[i+1:M+1:2, j:N:2] + U[i-1:M-1:2, j:N:2]) * one_dx2 + (U[i:M:2, j+1:N+1:2] + U[i:M:2, j-1:N-1:2]) * one_dy2 - F[i:M:2, j:N:2] ) / (one_dx2 + one_dy2)

        if (la.norm(U - U0, np.inf) < eps):
            print("Processo finalizado em", k, "iterações.", "Erro máximo entre iterações:", la.norm(U - U0, np.inf))
            break
        U0 = np.copy(U)
    return U

# =========================================================
# DADOS DO PROBLEMA
# =========================================================

# Definição da função f(x,y)
def f(x, y):
    return -8 * np.pi**2 * np.sin(2*np.pi * x) * np.sin(2*np.pi * y)

# Domínio do problema [a, b] x [c, d]
a, b = -1.0, 1.0
c, d = -1.0, 1.0

# Tolerância do método iterativo
eps = 1e-10

# Número máximo de iterações
kMax = 1000000

# Definição da solução "exata" do problema
def u_ex(x, y, lim):
    sum = 0
    for n in range(0, lim):
        sum += (-1)**n / (2*n + 1)**3 / np.cosh( ((2*n+1)*np.pi*a) / (2*b) ) * np.cosh( ((2*n+1)*np.pi*x) / (2*b) ) * np.cos( (2*n+1)*np.pi*y / (2*b) )
    
    u = b*b - y*y - 32*b*b / (np.pi**3) * sum

    return u

# =========================================================
# PARÂMETROS NUMÉRICOS
# =========================================================

# Erro entre as soluções
err1 = []
err2 = []
errM = []

# Parâmetros para iterar
ks = np.array([8, 16, 32, 64, 128])

dxs = []
dys = []

for k in ks:
   # Número de pontos internos na malha
    M = k
    N = k

    # Tamanho do espaçamenta
    dx = (b-a) / (M+1)
    dy = (d-c) / (N+1)

    dxs.append(dx)
    dys.append(dy)

    # =========================================================
    # RESOLUÇÃO DO PROBLEMA
    # =========================================================

    # Definição dos pontos em x e y
    x_i = np.linspace(a, b, M+2)
    y_j = np.linspace(c, d, N+2)

    # Definição da malha
    x, y = np.meshgrid(x_i, y_j)

    # Definição da função avaliada nos pontos da malha
    F = -2 * np.ones((M+2, N+2))

    # Matriz com os valores U_ij
    U = np.zeros((M+2, N+2))

    # Solução aproximada por método iterativo
    U = solver(U, F, dx, dy, eps, kMax)

    # Solução "exata" do problema
    U_ex = u_ex(x, y, 100)

    # Erro entre as soluções
    norm1 = norm1_func(U - U_ex, dx, dy)
    norm2 = norm2_func(U - U_ex, dx, dy)
    normM = normM_func(U - U_ex)

    err1.append(norm1)
    err2.append(norm2)
    errM.append(normM)

# =========================================================
# PRINT DO ERRO
# =========================================================

print("==================================================================")
print("Erro 1:", *['{:.5E}'.format(err1[j]) for j in range(0, 5)], sep =', ')
print("Erro 2:", *['{:.5E}'.format(err2[j]) for j in range(0, 5)],sep =', ')
print("Erro Max:", *['{:.5E}'.format(errM[j]) for j in range(0, 5)],sep =', ')
print("==================================================================")

# =========================================================
# PLOT DO ERRO
# =========================================================

# Plot do erro entre as soluções
plt.contourf(x, y, np.abs(U - U_ex))
plt.colorbar(format='%.0e')
plt.title('Erro da solução numérica.')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.show()

h = np.array(dxs)
h2 = h*h

# Plot da norma do erro
plt.loglog(h, err1, '*-', label=r'Norma $1$')
plt.loglog(h, err2, 'o-', label=r'Norma $2$')
plt.loglog(h, errM, 'x-', label=r'Norma $\infty$')
plt.loglog(h, h2/16, '--', label=r'Ordem 2')
plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.legend()
plt.grid()
plt.minorticks_on()
plt.title(r'Erro do método.')
plt.ylabel(r'$|| E ||$')
plt.xlabel(r'$\Delta x, \Delta y$')
plt.show()
