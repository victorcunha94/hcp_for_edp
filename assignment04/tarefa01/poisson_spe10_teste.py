import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt

# =============================================================================
# PARÂMETROS REALISTAS PARA RESERVATÓRIO DE PETRÓLEO
# =============================================================================
L1, N1 = 3.5, 220  # Dimensões típicas SPE10: 3.5m x 0.8m
L2, N2 = 0.8, 60
n1, n2 = N1 - 1, N2 - 1
mu = 8.9e-4  # Viscosidade realística do óleo (Pa.s) - CORRIGIDO!
phi = 0.2  # Porosidade típica

# Dimensões do SPE10
Nx_spe, Ny_spe, Nz_spe = 60, 220, 85


def carregar_dados_spe10_corrigido(arquivo='spe_perm.dat'):
    """Carrega os dados de permeabilidade do SPE10 corretamente"""
    total_values = Nx_spe * Ny_spe * Nz_spe * 3

    with open(arquivo, 'r') as file:
        all_values = np.array(file.read().split(), dtype=float)

    assert len(all_values) == total_values, f"Número de valores incorreto"

    kx = np.zeros((Nx_spe, Ny_spe, Nz_spe))
    ky = np.zeros((Nx_spe, Ny_spe, Nz_spe))
    kz = np.zeros((Nx_spe, Ny_spe, Nz_spe))

    index = 0
    for k in range(Nz_spe):
        for j in range(Ny_spe):
            for i in range(Nx_spe):
                kx[i, j, k] = all_values[index]
                ky[i, j, k] = all_values[index + Nx_spe * Ny_spe * Nz_spe]
                kz[i, j, k] = all_values[index + 2 * Nx_spe * Ny_spe * Nz_spe]
                index += 1

    return kx, ky, kz


# Carregar dados
kx, ky, kz = carregar_dados_spe10_corrigido('spe_perm.dat')

# Converter para m² (de mD para m²) e garantir positividade
kx = np.maximum(kx * 9.869233e-16, 1e-20)
ky = np.maximum(ky * 9.869233e-16, 1e-20)
kz = np.maximum(kz * 9.869233e-16, 1e-20)

# Selecionar camadas 33 e 35
K33 = kx[:, :, 32]  # Camada 33 (índice 32)
K35 = kx[:, :, 34]  # Camada 35 (índice 34)


# =============================================================================
# FUNÇÕES PARA RESOLVER EQUAÇÃO DE DARCY ESTACIONÁRIA
# =============================================================================
def CreateMesh(L1, L2, N1, N2):
    """Cria malha computacional com dimensões realísticas"""
    x = np.linspace(0.0, L1, N1)
    y = np.linspace(0.0, L2, N2)
    return x, y


def jglob(i1, i2, n1):
    """Mapeia índices 2D para 1D"""
    return i1 + i2 * n1


def BuildDarcySystem(x, y, K_func, bc_left=1e6, bc_right=0.0):  # Pressões em Pa
    """
    Constrói sistema linear para equação de Darcy ESTACIONÁRIA:
    ∇·(K/μ ∇p) = 0
    """
    N1, N2 = len(x), len(y)
    n1, n2 = N1 - 1, N2 - 1
    Nc = n1 * n2

    # Geometria da malha
    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])

    row, col, data = [], [], []
    rhs = np.zeros(Nc)

    for i in range(n1):
        for j in range(n2):
            g = jglob(i, j, n1)
            diag_coef = 0.0

            # Fluxo na direção x (Leste-Oeste)
            if i < n1 - 1:
                # Interface leste - média harmônica
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i + 1], yc[j])
                K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_e = K_e * dy[j] / (0.5 * dx[i] + 0.5 * dx[i + 1]) / mu
                g_e = jglob(i + 1, j, n1)
                row.append(g);
                col.append(g_e);
                data.append(-Tx_e)
                diag_coef += Tx_e

            if i > 0:
                # Interface oeste - média harmônica
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i - 1], yc[j])
                K_w = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_w = K_w * dy[j] / (0.5 * dx[i] + 0.5 * dx[i - 1]) / mu
                g_w = jglob(i - 1, j, n1)
                row.append(g);
                col.append(g_w);
                data.append(-Tx_w)
                diag_coef += Tx_w

            # Fluxo na direção y (Norte-Sul) - condições de Neumann (fluxo zero)
            if j < n2 - 1:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i], yc[j + 1])
                K_n = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_n = K_n * dx[i] / (0.5 * dy[j] + 0.5 * dy[j + 1]) / mu
                g_n = jglob(i, j + 1, n1)
                row.append(g);
                col.append(g_n);
                data.append(-Ty_n)
                diag_coef += Ty_n

            if j > 0:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i], yc[j - 1])
                K_s = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_s = K_s * dx[i] / (0.5 * dy[j] + 0.5 * dy[j - 1]) / mu
                g_s = jglob(i, j - 1, n1)
                row.append(g);
                col.append(g_s);
                data.append(-Ty_s)
                diag_coef += Ty_s

            # Condições de contorno de Dirichlet nas laterais
            if i == 0:
                # Contorno esquerdo - pressão alta
                K_face = K_func(xc[i], yc[j])
                T_bc = K_face * dy[j] / (0.5 * dx[i]) / mu
                rhs[g] += T_bc * bc_left
                diag_coef += T_bc
            elif i == n1 - 1:
                # Contorno direito - pressão baixa
                K_face = K_func(xc[i], yc[j])
                T_bc = K_face * dy[j] / (0.5 * dx[i]) / mu
                rhs[g] += T_bc * bc_right
                diag_coef += T_bc

            # Termo diagonal
            row.append(g)
            col.append(g)
            data.append(diag_coef)

    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc)).tocsr()
    return A, rhs, xc, yc


def solve_darcy_case(K_func, caso_nome, bc_left=1e6, bc_right=0.0):
    """Resolve caso específico da equação de Darcy estacionária"""
    print(f"\n=== Resolvendo caso: {caso_nome} ===")
    A, rhs, xc, yc = BuildDarcySystem(x, y, K_func, bc_left, bc_right)

    t0 = time.time()
    P = scipy.sparse.linalg.spsolve(A, rhs)
    tempo_solucao = time.time() - t0

    print(f"Tempo de solução: {tempo_solucao:.3f} s")
    print(f"Pressão: min={P.min():.2e} Pa, max={P.max():.2e} Pa")

    return P, xc, yc


# =============================================================================
# ANÁLISE COMPARATIVA
# =============================================================================
def analisar_resultados(P, K, caso_nome, xc, yc):
    """Analisa resultados computados"""
    P_2D = P.reshape((n1, n2)).T

    # Calcular gradiente de pressão e velocidade de Darcy
    dx = x[1] - x[0]
    gradP_x, gradP_y = np.gradient(P_2D, dx, dx)
    v_darcy_x = -K * gradP_x / mu  # Velocidade de Darcy em x

    print(f"\n--- Análise {caso_nome} ---")
    print(f"Gradiente de pressão médio: {np.mean(np.abs(gradP_x)):.2e} Pa/m")
    print(f"Velocidade de Darcy média: {np.mean(np.abs(v_darcy_x)):.2e} m/s")
    print(f"Permeabilidade média: {np.mean(K):.2e} m²")
    print(f"Variação de K: min={K.min():.2e}, max={K.max():.2e} m²")

    return P_2D, v_darcy_x


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================
# Criar malha com dimensões realísticas
x, y = CreateMesh(L1, L2, N1, N2)

# Interpolar permeabilidades para a malha computacional
K33_interp = zoom(K33, (n2 / K33.shape[0], n1 / K33.shape[1]), order=1)
K35_interp = zoom(K35, (n2 / K35.shape[0], n1 / K35.shape[1]), order=1)

# Garantir positividade
K33_interp = np.maximum(K33_interp, 1e-20)
K35_interp = np.maximum(K35_interp, 1e-20)

# Funções de permeabilidade
K_func_33 = lambda x, y: K33_interp[
    min(int(y / L2 * (K33_interp.shape[0] - 1)), K33_interp.shape[0] - 1),
    min(int(x / L1 * (K33_interp.shape[1] - 1)), K33_interp.shape[1] - 1)
]

K_func_35 = lambda x, y: K35_interp[
    min(int(y / L2 * (K35_interp.shape[0] - 1)), K35_interp.shape[0] - 1),
    min(int(x / L1 * (K35_interp.shape[1] - 1)), K35_interp.shape[1] - 1)
]

# Resolver casos com pressões realísticas (1 MPa a 0 MPa)
print("=" * 60)
print("SOLUÇÃO ESTACIONÁRIA - EQUAÇÃO DE DARCY")
print(f"Domínio: {L1} m x {L2} m")
print(f"Condições: p(0,y) = 1 MPa, p({L1},y) = 0 MPa")
print("=" * 60)

P33, xc, yc = solve_darcy_case(K_func_33, "SPE10 - Camada 33", 1e6, 0.0)
P35, _, _ = solve_darcy_case(K_func_35, "SPE10 - Camada 35", 1e6, 0.0)

# Analisar resultados
P33_2D, v33 = analisar_resultados(P33, K33_interp, "Camada 33", xc, yc)
P35_2D, v35 = analisar_resultados(P35, K35_interp, "Camada 35", xc, yc)

# =============================================================================
# VISUALIZAÇÃO - FORMATO PUBLICAÇÃO
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Camada 33
im1 = axes[0, 0].imshow(np.log10(K33_interp), extent=[0, L1, 0, L2],
                        origin='lower', cmap='viridis', aspect='auto')
axes[0, 0].set_title("(a) Permeabilidade - Camada 33", fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('x [m]')
axes[0, 0].set_ylabel('y [m]')
plt.colorbar(im1, ax=axes[0, 0], label='log10(K [m²])')

im2 = axes[0, 1].imshow(P33_2D / 1e6, extent=[0, L1, 0, L2],
                        origin='lower', cmap='plasma', aspect='auto')
axes[0, 1].set_title("(b) Pressão - Camada 33", fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('x [m]')
axes[0, 1].set_ylabel('y [m]')
plt.colorbar(im2, ax=axes[0, 1], label='p [MPa]')

im3 = axes[0, 2].imshow(np.log10(np.abs(v33) + 1e-30), extent=[0, L1, 0, L2],
                        origin='lower', cmap='hot', aspect='auto')
axes[0, 2].set_title("(c) Velocidade Darcy - Camada 33", fontsize=12, fontweight='bold')
axes[0, 2].set_xlabel('x [m]')
axes[0, 2].set_ylabel('y [m]')
plt.colorbar(im3, ax=axes[0, 2], label='log10(v [m/s])')

# Camada 35
im4 = axes[1, 0].imshow(np.log10(K35_interp), extent=[0, L1, 0, L2],
                        origin='lower', cmap='viridis', aspect='auto')
axes[1, 0].set_title("(d) Permeabilidade - Camada 35", fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('x [m]')
axes[1, 0].set_ylabel('y [m]')
plt.colorbar(im4, ax=axes[1, 0], label='log10(K [m²])')

im5 = axes[1, 1].imshow(P35_2D / 1e6, extent=[0, L1, 0, L2],
                        origin='lower', cmap='plasma', aspect='auto')
axes[1, 1].set_title("(e) Pressão - Camada 35", fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('x [m]')
axes[1, 1].set_ylabel('y [m]')
plt.colorbar(im5, ax=axes[1, 1], label='p [MPa]')

im6 = axes[1, 2].imshow(np.log10(np.abs(v35) + 1e-30), extent=[0, L1, 0, L2],
                        origin='lower', cmap='hot', aspect='auto')
axes[1, 2].set_title("(f) Velocidade Darcy - Camada 35", fontsize=12, fontweight='bold')
axes[1, 2].set_xlabel('x [m]')
axes[1, 2].set_ylabel('y [m]')
plt.colorbar(im6, ax=axes[1, 2], label='log10(v [m/s])')

plt.tight_layout()
plt.savefig("comparacao_estacionaria_spe10.png", dpi=300, bbox_inches='tight')
plt.show()

# Perfis comparativos
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
j_center = n2 // 2
plt.plot(xc, P33_2D[j_center, :] / 1e6, 'b-', lw=2, label='Camada 33')
plt.plot(xc, P35_2D[j_center, :] / 1e6, 'r--', lw=2, label='Camada 35')
plt.xlabel('x [m]')
plt.ylabel('p [MPa]')
plt.title('Perfil de Pressão (y central)')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.semilogy(xc, np.abs(v33[j_center, :]), 'b-', lw=2, label='Camada 33')
plt.semilogy(xc, np.abs(v35[j_center, :]), 'r--', lw=2, label='Camada 35')
plt.xlabel('x [m]')
plt.ylabel('|v| [m/s]')
plt.title('Velocidade de Darcy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
from scipy.integrate import trapezoid

fluxo_total_33 = trapezoid(np.abs(v33), dx=dy[0], axis=0)
fluxo_total_35 = trapezoid(np.abs(v35), dx=dy[0], axis=0)
plt.plot(xc, fluxo_total_33, 'b-', lw=2, label='Camada 33')
plt.plot(xc, fluxo_total_35, 'r--', lw=2, label='Camada 35')
plt.xlabel('x [m]')
plt.ylabel('Fluxo [m²/s]')
plt.title('Fluxo Total Integrado')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("perfis_comparativos.png", dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("RESUMO DA COMPARAÇÃO ESTACIONÁRIA")
print("=" * 60)
print(f"Camada 33 - Permeabilidade média: {np.mean(K33):.2e} m² ({np.mean(K33) / 9.869233e-16:.1f} mD)")
print(f"Camada 35 - Permeabilidade média: {np.mean(K35):.2e} m² ({np.mean(K35) / 9.869233e-16:.1f} mD)")
print(f"Razão K33/K35: {np.mean(K33) / np.mean(K35):.2f}")
print(f"Razão fluxo_total_33/fluxo_total_35: {np.mean(fluxo_total_33) / np.mean(fluxo_total_35):.2f}")