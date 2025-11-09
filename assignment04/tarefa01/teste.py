import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# =============================================================================
# PARÂMETROS REALISTAS PARA RESERVATÓRIO DE PETRÓLEO
# =============================================================================
L1, N1 = 220, 2000  # Dimensões típicas SPE10: 3.5m x 0.8m
L2, N2 = 60, 480
n1, n2 = N1 - 1, N2 - 1
mu = 8.9e-4  # Viscosidade realística do óleo (Pa.s)
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
    return A, rhs, xc, yc, dx, dy


def solve_darcy_case(K_func, caso_nome, bc_left=1e6, bc_right=0.0):
    """Resolve caso específico da equação de Darcy estacionária"""
    print(f"\n=== Resolvendo caso: {caso_nome} ===")
    A, rhs, xc, yc, dx, dy = BuildDarcySystem(x, y, K_func, bc_left, bc_right)

    t0 = time.time()
    P = scipy.sparse.linalg.spsolve(A, rhs)
    tempo_solucao = time.time() - t0

    print(f"Tempo de solução: {tempo_solucao:.3f} s")
    print(f"Pressão: min={P.min():.2e} Pa, max={P.max():.2e} Pa")

    return P, xc, yc, dx, dy


# =============================================================================
# CARREGAR E VISUALIZAR CORRETAMENTE OS DADOS DO SPE10
# =============================================================================

def visualizar_camada_spe10_correta(camada=36, componente='kx'):
    """
    Visualiza a camada do SPE10 exatamente como nos trabalhos científicos
    """

    # Carregar dados originais
    kx, ky, kz = carregar_dados_spe10_corrigido('spe_perm.dat')

    # Selecionar camada (índice = camada - 1)
    camada_idx = camada - 1

    if componente == 'kx':
        K_original = kx[:, :, camada_idx]  # Forma: (60, 220)
    elif componente == 'ky':
        K_original = ky[:, :, camada_idx]
    else:  # kz
        K_original = kz[:, :, camada_idx]

    print(f"Forma original da camada {camada}: {K_original.shape}")

    # CORREÇÃO CRÍTICA: Transpor e inverter para visualização correta
    # O SPE10 tem orientação específica que precisa ser ajustada
    K_corrigido = np.flipud(K_original.T)  # Transpor + flip vertical

    print(f"Forma corrigida: {K_corrigido.shape}")

    # Visualização
    plt.figure(figsize=(12, 5))

    # Plot da forma CORRETA (como nos papers)
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(np.log10(K_corrigido), cmap='viridis',
                     extent=[0, 220, 0, 60], aspect='auto')
    plt.title(f'Camada {camada} - {componente.upper()} (Visualização Correta)')
    plt.xlabel('Coord X (220 células)')
    plt.ylabel('Coord Y (60 células)')
    plt.colorbar(im1, label='log10(K [mD])')

    # Adicionar grid para referência
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 221, 20))
    plt.yticks(np.arange(0, 61, 10))

    # Plot da forma ORIGINAL (para comparação)
    plt.subplot(1, 2, 2)
    im2 = plt.imshow(np.log10(K_original), cmap='viridis',
                     extent=[0, 60, 0, 220], aspect='auto')
    plt.title(f'Camada {camada} - {componente.upper()} (Forma Bruta)')
    plt.xlabel('Coord X (60 células)')
    plt.ylabel('Coord Y (220 células)')
    plt.colorbar(im2, label='log10(K [mD])')

    plt.tight_layout()
    plt.show()

    return K_corrigido, K_original


# =============================================================================
# FUNÇÃO PARA RESOLVER DARCY COM ORIENTAÇÃO CORRETA
# =============================================================================

def resolver_darcy_com_orientacao_correta(camada=36, componente='kx'):
    """
    Resolve a equação de Darcy mantendo a orientação correta do SPE10
    """

    # Carregar e preparar dados com orientação correta
    kx, ky, kz = carregar_dados_spe10_corrigido('spe_perm.dat')
    camada_idx = camada - 1

    if componente == 'kx':
        K_original = kx[:, :, camada_idx]
    else:
        K_original = ky[:, :, camada_idx]

    # CORREÇÃO: Manter orientação original do SPE10
    K_corrigido = np.flipud(K_original.T)  # (220, 60) - orientação correta

    # Converter para m²
    K_corrigido = np.maximum(K_corrigido * 9.869233e-16, 1e-20)

    # Dimensões do domínio (em metros)
    Lx, Ly = 3.5, 0.8  # 220 células x 60 células
    Nx, Ny = 220, 60  # Manter resolução original

    # Criar malha
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)

    # Interpolar se necessário (mas manter orientação)
    if (Nx, Ny) != K_corrigido.shape:
        # Interpolar mantendo proporções
        scale_x = Nx / K_corrigido.shape[0]
        scale_y = Ny / K_corrigido.shape[1]
        K_interp = zoom(K_corrigido, (scale_x, scale_y), order=1)
    else:
        K_interp = K_corrigido

    # Função de permeabilidade com orientação correta
    def K_func(x, y):
        i = min(int(x / Lx * (K_interp.shape[0] - 1)), K_interp.shape[0] - 1)
        j = min(int(y / Ly * (K_interp.shape[1] - 1)), K_interp.shape[1] - 1)
        return K_interp[i, j]

    # Resolver sistema de Darcy (usar sua função existente)
    P, xc, yc, dx, dy = solve_darcy_case(K_func, f"SPE10 - Camada {camada}", 1e6, 0.0)

    return P, K_interp, xc, yc


# =============================================================================
# COMPARAÇÃO DIRETA COM SEU CÓDIGO ATUAL
# =============================================================================

def comparar_visualizacoes():
    """
    Compara a visualização atual vs a correta
    """

    # Carregar camada 36
    kx, ky, kz = carregar_dados_spe10_corrigido('spe_perm.dat')

    # Camada 36 (índice 35)
    K36_original = kx[:, :, 35]  # Forma: (60, 220)

    print("=== ANÁLISE DA CAMADA 36 ===")
    print(f"Forma original: {K36_original.shape}")
    print(f"Valores - Min: {K36_original.min():.2f} mD, Max: {K36_original.max():.2f} mD")
    print(f"Média: {K36_original.mean():.2f} mD")

    # Sua visualização atual (provavelmente errada)
    K_atual = zoom(K36_original, (60 / K36_original.shape[0], 220 / K36_original.shape[1]))

    # Visualização correta
    K_correta = np.flipud(K36_original.T)

    # Plot comparativo
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original
    im0 = axes[0].imshow(np.log10(K36_original), cmap='viridis', aspect='auto')
    axes[0].set_title('Forma Bruta (60, 220)')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im0, ax=axes[0])

    # Sua visualização atual
    im1 = axes[1].imshow(np.log10(K_atual), cmap='viridis', aspect='auto')
    axes[1].set_title('Sua Visualização Atual')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[1])

    # Visualização correta
    im2 = axes[2].imshow(np.log10(K_correta), cmap='viridis', aspect='auto')
    axes[2].set_title('Visualização Correta (220, 60)')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.show()

    return K_correta

# =============================================================================
# ANÁLISE COMPARATIVA
# =============================================================================
def analisar_resultados(P, K, caso_nome, xc, yc, dx_val):
    """Analisa resultados computados"""
    P_2D = P.reshape((n1, n2)).T

    # Calcular gradiente de pressão e velocidade de Darcy
    gradP_x, gradP_y = np.gradient(P_2D, dx_val, dx_val)
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

P33, xc, yc, dx, dy = solve_darcy_case(K_func_33, "SPE10 - Camada 33", 1e6, 0.0)
P35, _, _, _, _ = solve_darcy_case(K_func_35, "SPE10 - Camada 35", 1e6, 0.0)

# Analisar resultados
P33_2D, v33 = analisar_resultados(P33, K33_interp, "Camada 33", xc, yc, dx[0])
P35_2D, v35 = analisar_resultados(P35, K35_interp, "Camada 35", xc, yc, dx[0])







# =============================================================================
# PLOTS SEPARADOS - CAMADA 33
# =============================================================================
print("\nGerando plots da Camada 33...")

# Plot 1: Permeabilidade Camada 33
plt.figure(figsize=(11, 3))
im1 = plt.imshow(np.log10(K33_interp), extent=[0, L1, 0, L2],
                 origin='lower', cmap='viridis', aspect='auto')
plt.title("Permeabilidade - Camada 33 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im1, label='log10(K [m²])')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("permeabilidade_camada_33.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Pressão Camada 33
plt.figure(figsize=(11, 3))
im2 = plt.imshow(P33_2D / 1e6, extent=[0, L1, 0, L2],
                 origin='lower', cmap='plasma', aspect='auto')
plt.title("Campo de Pressão - Camada 33 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im2, label='p [MPa]')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("pressao_camada_33.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Velocidade Darcy Camada 33
plt.figure(figsize=(11, 3))
im3 = plt.imshow(np.log10(np.abs(v33) + 1e-30), extent=[0, L1, 0, L2],
                 origin='lower', cmap='hot', aspect='auto')
plt.title("Velocidade de Darcy - Camada 33 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im3, label='log10(v [m/s])')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("velocidade_camada_33.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOTS SEPARADOS - CAMADA 35
# =============================================================================
print("Gerando plots da Camada 35...")

# Plot 4: Permeabilidade Camada 35
plt.figure(figsize=(11, 3))
im4 = plt.imshow(np.log10(K35_interp), extent=[0, L1, 0, L2],
                 origin='lower', cmap='viridis', aspect='auto')
plt.title("Permeabilidade - Camada 35 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im4, label='log10(K [m²])')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("permeabilidade_camada_35.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 5: Pressão Camada 35
plt.figure(figsize=(11, 3))
im5 = plt.imshow(P35_2D / 1e6, extent=[0, L1, 0, L2],
                 origin='lower', cmap='plasma', aspect='auto')
plt.title("Campo de Pressão - Camada 35 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im5, label='p [MPa]')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("pressao_camada_35.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 6: Velocidade Darcy Camada 35
plt.figure(figsize=(11, 3))
im6 = plt.imshow(np.log10(np.abs(v35) + 1e-30), extent=[0, L1, 0, L2],
                 origin='lower', cmap='hot', aspect='auto')
plt.title("Velocidade de Darcy - Camada 35 SPE10", fontsize=14, fontweight='bold')
plt.xlabel('x [m]', fontsize=12)
plt.ylabel('y [m]', fontsize=12)
cbar = plt.colorbar(im6, label='log10(v [m/s])')
cbar.ax.tick_params(labelsize=10)
plt.tight_layout()
plt.savefig("velocidade_camada_35.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# PLOTS COMPARATIVOS
# =============================================================================
print("Gerando plots comparativos...")

# Plot 7: Comparação de Permeabilidades
plt.figure(figsize=(11, 3))

plt.subplot(1, 2, 1)
plt.imshow(np.log10(K33_interp), extent=[0, L1, 0, L2],
           origin='lower', cmap='viridis', aspect='auto')
plt.title("Camada 33", fontsize=12, fontweight='bold')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='log10(K [m²])')

plt.subplot(1, 2, 2)
plt.imshow(np.log10(K35_interp), extent=[0, L1, 0, L2],
           origin='lower', cmap='viridis', aspect='auto')
plt.title("Camada 35", fontsize=12, fontweight='bold')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='log10(K [m²])')

plt.suptitle("Comparação de Permeabilidades - SPE10", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("comparacao_permeabilidades.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 8: Comparação de Pressões
plt.figure(figsize=(11, 3))

plt.subplot(1, 2, 1)
plt.imshow(P33_2D / 1e6, extent=[0, L1, 0, L2],
           origin='lower', cmap='plasma', aspect='auto')
plt.title("Camada 33", fontsize=12, fontweight='bold')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='p [MPa]')

plt.subplot(1, 2, 2)
plt.imshow(P35_2D / 1e6, extent=[0, L1, 0, L2],
           origin='lower', cmap='plasma', aspect='auto')
plt.title("Camada 35", fontsize=12, fontweight='bold')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.colorbar(label='p [MPa]')

plt.suptitle("Comparação de Campos de Pressão - SPE10", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("comparacao_pressoes.png", dpi=300, bbox_inches='tight')
plt.show()

# Plot 9: Perfis Comparativos
plt.figure(figsize=(11, 3))

# Calcular fluxos totais (CORREÇÃO DO ERRO)
fluxo_total_33 = trapezoid(np.abs(v33), dx=dy[0], axis=0)
fluxo_total_35 = trapezoid(np.abs(v35), dx=dy[0], axis=0)

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
plt.plot(xc, fluxo_total_33, 'b-', lw=2, label='Camada 33')
plt.plot(xc, fluxo_total_35, 'r--', lw=2, label='Camada 35')
plt.xlabel('x [m]')
plt.ylabel('Fluxo [m²/s]')
plt.title('Fluxo Total Integrado')
plt.legend()
plt.grid(True)

plt.suptitle("Análise Comparativa - Camadas 33 vs 35", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("perfis_comparativos.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# RESUMO FINAL
# =============================================================================
print("\n" + "=" * 60)
print("RESUMO DA COMPARAÇÃO ESTACIONÁRIA")
print("=" * 60)
print(f"Camada 33 - Permeabilidade média: {np.mean(K33):.2e} m² ({np.mean(K33) / 9.869233e-16:.1f} mD)")
print(f"Camada 35 - Permeabilidade média: {np.mean(K35):.2e} m² ({np.mean(K35) / 9.869233e-16:.1f} mD)")
print(f"Razão K33/K35: {np.mean(K33) / np.mean(K35):.2f}")
print(f"Razão fluxo_total_33/fluxo_total_35: {np.mean(fluxo_total_33) / np.mean(fluxo_total_35):.2f}")
print(f"Velocidade média Camada 33: {np.mean(np.abs(v33)):.2e} m/s")
print(f"Velocidade média Camada 35: {np.mean(np.abs(v35)):.2e} m/s")
print(f"Razão de velocidades: {np.mean(np.abs(v35)) / np.mean(np.abs(v33)):.2f}")

print("\n✅ Análise completa concluída! 9 gráficos gerados:")
print("• 6 gráficos individuais (3 por camada)")
print("• 3 gráficos comparativos")



# =============================================================================
# EXECUTAR ANÁLISE
# =============================================================================

if __name__ == "__main__":
    # 1. Primeiro, veja a diferença nas visualizações
    print("Comparando visualizações...")
    K36_correta = comparar_visualizacoes()

    # 2. Visualização específica da camada 36 (como nos papers)
    print("\nVisualização correta da camada 36:")
    K_corr, K_orig = visualizar_camada_spe10_correta(camada=36, componente='kx')

    # 3. Agora resolva Darcy com orientação correta
    print("\nResolvendo Darcy com orientação correta...")
    P, K, xc, yc = resolver_darcy_com_orientacao_correta(camada=36, componente='kx')