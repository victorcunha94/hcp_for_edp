import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt
from uvw import RectilinearGrid, DataArray


# Creating coordinates
L1, N1 = 1.0, 513
L2, N2 = 1.0, 513
n1, n2 = N1 - 1, N2 - 1


def CreateMesh(L1, L2, N1, N2):
    """Cria malha uniforme"""
    x = np.linspace(0.0, L1, N1)
    y = np.linspace(0.0, L2, N2)
    return x, y


x, y = CreateMesh(L1, L2, N1, N2)


# Post-processing
def WriteSol(x, y, n1, n2, n, tn, P):
    """Escreve solução em formato VTK"""
    filename = 'poisson_sol' + str(n) + '.vtr'
    grid = RectilinearGrid(filename, (x, y), compression=True)
    grid.addCellData(DataArray(P.reshape(n1, n2), range(2), 'Pressure'))
    grid.write()


def jglob(i1, i2, n1):
    """Mapeia índices 2D para 1D"""
    return i1 + i2 * n1


def BuildPoissonSystem(x, y, K_func, f_func):
    """
    Constrói o sistema para a equação de Poisson:
    -∇·(K∇p) = f
    """
    N1, N2 = len(x), len(y)
    n1, n2 = N1 - 1, N2 - 1
    Nc = n1 * n2

    # Pré-calcular geometria
    dx = np.zeros(n1)
    dy = np.zeros(n2)
    xc = np.zeros(n1)
    yc = np.zeros(n2)
    vol = np.zeros(Nc)

    for i in range(n1):
        dx[i] = x[i + 1] - x[i]
        xc[i] = 0.5 * (x[i] + x[i + 1])

    for j in range(n2):
        dy[j] = y[j + 1] - y[j]
        yc[j] = 0.5 * (y[j] + y[j + 1])

    for i in range(n1):
        for j in range(n2):
            g = jglob(i, j, n1)
            vol[g] = dx[i] * dy[j]

    # Inicializar matriz e vetor
    row, col, data = [], [], []
    rhs = np.zeros(Nc)

    print("Montando sistema de Poisson...")

    for i in range(n1):
        for j in range(n2):
            g = jglob(i, j, n1)

            # Termo fonte
            rhs[g] = f_func(xc[i], yc[j]) * vol[g]

            # Coeficiente central
            diag_coef = 0.0

            # Fluxo na face leste (i+1/2, j)
            if i < n1 - 1:
                # Coeficiente na interface
                K_east = 0.5 * (K_func(xc[i], yc[j]) + K_func(xc[i + 1], yc[j]))
                coef = -K_east * dy[j] / (0.5 * dx[i] + 0.5 * dx[i + 1])

                g_east = jglob(i + 1, j, n1)
                row.append(g);
                col.append(g_east);
                data.append(coef)
                diag_coef -= coef
            else:
                # Condição de contorno Neumann homogênea (fluxo zero)
                #pass
                K_face = K_func(xc[i], yc[j])
                T = K_face * dy[j] / (0.5 * dx[i])
                diag_coef += T

            # Fluxo na face oeste (i-1/2, j)
            if i > 0:
                K_west = 0.5 * (K_func(xc[i], yc[j]) + K_func(xc[i - 1], yc[j]))
                coef = -K_west * dy[j] / (0.5 * dx[i] + 0.5 * dx[i - 1])

                g_west = jglob(i - 1, j, n1)
                row.append(g);
                col.append(g_west);
                data.append(coef)
                diag_coef -= coef
            else:
                # Condição de contorno Neumann homogênea
                #pass
                K_face = K_func(xc[i], yc[j])
                T = K_face * dy[j] / (0.5 * dx[i])
                diag_coef += T

            # Fluxo na face norte (i, j+1/2)
            if j < n2 - 1:
                K_north = 0.5 * (K_func(xc[i], yc[j]) + K_func(xc[i], yc[j + 1]))
                coef = -K_north * dx[i] / (0.5 * dy[j] + 0.5 * dy[j + 1])

                g_north = jglob(i, j + 1, n1)
                row.append(g);
                col.append(g_north);
                data.append(coef)
                diag_coef -= coef
            else:
                # Condição de contorno Neumann homogênea
                #pass
                K_face = K_func(xc[i], yc[j])
                T = K_face * dx[i] / (0.5 * dy[j])
                diag_coef += T

            # Fluxo na face sul (i, j-1/2)
            if j > 0:
                K_south = 0.5 * (K_func(xc[i], yc[j]) + K_func(xc[i], yc[j - 1]))
                coef = -K_south * dx[i] / (0.5 * dy[j] + 0.5 * dy[j - 1])

                g_south = jglob(i, j - 1, n1)
                row.append(g);
                col.append(g_south);
                data.append(coef)
                diag_coef -= coef
            else:
                # Condição de contorno Neumann homogênea
                #pass
                K_face = K_func(xc[i], yc[j])
                T = K_face * dx[i] / (0.5 * dy[j])
                diag_coef += T


            # Adicionar termo diagonal
            row.append(g);
            col.append(g);
            data.append(diag_coef)

    # Criar matriz esparsa
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc))
    A = A.tocsr()

    return A, rhs, vol, xc, yc


def K_func(x, y):
    """Coeficiente de difusividade"""
    # Exemplo: coeficiente constante
    return 1.0

    # Exemplo: coeficiente variável
    #return 1.0 + 0.5 * np.sin(2*np.pi*x) * np.cos(2*np.pi*y)


def f_func(x, y):
    """Termo fonte"""
    # Exemplo 1: Fonte senoidal (solução suave)
    return 2.0 * (np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)

    # Exemplo 2: Fonte constante
    # return 1.0

    # Exemplo 3: Fonte pontual
    # if 0.4 <= x <= 0.6 and 0.4 <= y <= 0.6:
    #     return 10.0
    # else:
    #     return 0.0


def analytic_solution(x, y):
    """Solução analítica para f = 2π² sin(πx) sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# Construir e resolver sistema
print("=== SOLUÇÃO DA EQUAÇÃO DE POISSON ===")
print(f"Malha: {n1} x {n2} células")
print(f"Domínio: [0, {L1}] x [0, {L2}]")

A, rhs, vol, xc, yc = BuildPoissonSystem(x, y, K_func, f_func)

print(f"Matriz: {A.shape[0]} x {A.shape[1]}, {A.nnz} elementos não-nulos")

# Resolver sistema
print("Resolvendo sistema linear...")
t0 = time.time()
P = scipy.sparse.linalg.spsolve(A, rhs)
solve_time = time.time() - t0
print(f"Tempo de solução: {solve_time:.3f} segundos")

# Calcular solução analítica e erro
P_analytic = np.zeros_like(P)
for i in range(n1):
    for j in range(n2):
        g = jglob(i, j, n1)
        P_analytic[g] = analytic_solution(xc[i], yc[j])

error = np.abs(P - P_analytic)
L2_error = np.sqrt(np.sum(error ** 2 * vol) / np.sum(vol))
max_error = np.max(error)

print(f"Erro L2: {L2_error:.2e}")
print(f"Erro máximo: {max_error:.2e}")

# Salvar solução
WriteSol(x, y, n1, n2, 0, 0.0, P)

# Visualizar resultados
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Solução numérica
P_plot = P.reshape(n1, n2)
im1 = axes[0, 0].imshow(P_plot.T, extent=[0, L1, 0, L2], origin='lower', cmap='jet')
axes[0, 0].set_title('Solução Numérica')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('y')
plt.colorbar(im1, ax=axes[0, 0])

# Solução analítica
P_ana_plot = P_analytic.reshape(n1, n2)
im2 = axes[0, 1].imshow(P_ana_plot.T, extent=[0, L1, 0, L2], origin='lower', cmap='jet')
axes[0, 1].set_title('Solução Analítica')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('y')
plt.colorbar(im2, ax=axes[0, 1])

# Erro
error_plot = error.reshape(n1, n2)
im3 = axes[1, 0].imshow(error_plot.T, extent=[0, L1, 0, L2], origin='lower', cmap='hot')
axes[1, 0].set_title('Erro Absoluto')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')
plt.colorbar(im3, ax=axes[1, 0])

# Perfil no centro
j_center = n2 // 2
axes[1, 1].plot(xc, P_plot[:, j_center], 'b-', label='Numérica', linewidth=2)
axes[1, 1].plot(xc, P_ana_plot[:, j_center], 'r--', label='Analítica', linewidth=2)
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('p')
axes[1, 1].set_title(f'Perfil em y = {yc[j_center]:.2f}')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('poisson_solution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Simulação concluída!")