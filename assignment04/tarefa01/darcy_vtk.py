import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time

# =============================================================================
# PARÂMETROS CORRETOS SPE10
# =============================================================================
Lx, Nx = 220, 1220  # 220 células na direção X
Ly, Ny = 60, 260  # 60 células na direção Y
nx, ny = Nx - 1, Ny - 1
mu = 8.9e-4

# Dimensões originais do SPE10 (60 x 220 x 85)
Nx_spe, Ny_spe, Nz_spe = 60, 220, 85


def carregar_dados_spe10(arquivo='spe_perm.dat'):
    total_values = Nx_spe * Ny_spe * Nz_spe * 3

    with open(arquivo, 'r') as file:
        all_values = np.array(file.read().split(), dtype=float)

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


def CreateMesh(Lx, Ly, Nx, Ny):
    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    return x, y


def jglob(i1, i2, n1):
    return i1 + i2 * n1


def BuildDarcySystem(x, y, K_func, bc_left=1e6, bc_right=0.0):
    N1, N2 = len(x), len(y)
    n1, n2 = N1 - 1, N2 - 1
    Nc = n1 * n2

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

            # Fluxo na direção x
            if i < n1 - 1:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i + 1], yc[j])
                K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_e = K_e * dy[j] / (0.5 * dx[i] + 0.5 * dx[i + 1]) / mu
                g_e = jglob(i + 1, j, n1)
                row.append(g);
                col.append(g_e);
                data.append(-Tx_e)
                diag_coef += Tx_e

            if i > 0:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i - 1], yc[j])
                K_w = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_w = K_w * dy[j] / (0.5 * dx[i] + 0.5 * dx[i - 1]) / mu
                g_w = jglob(i - 1, j, n1)
                row.append(g);
                col.append(g_w);
                data.append(-Tx_w)
                diag_coef += Tx_w

            # Fluxo na direção y
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

            # Condições de contorno
            if i == 0:
                K_face = K_func(xc[i], yc[j])
                T_bc = K_face * dy[j] / (0.5 * dx[i]) / mu
                rhs[g] += T_bc * bc_left
                diag_coef += T_bc
            elif i == n1 - 1:
                K_face = K_func(xc[i], yc[j])
                T_bc = K_face * dy[j] / (0.5 * dx[i]) / mu
                rhs[g] += T_bc * bc_right
                diag_coef += T_bc

            row.append(g)
            col.append(g)
            data.append(diag_coef)

    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc)).tocsr()
    return A, rhs, xc, yc, dx, dy


def solve_darcy_case(K_func, caso_nome, bc_left=1e6, bc_right=0.0):
    print(f"Resolvendo: {caso_nome}")
    A, rhs, xc, yc, dx, dy = BuildDarcySystem(x, y, K_func, bc_left, bc_right)

    t0 = time.time()
    P = scipy.sparse.linalg.spsolve(A, rhs)
    print(f"Tempo: {time.time() - t0:.3f} s")

    return P, xc, yc, dx, dy


def salvar_para_paraview(xc, yc, P, K, v, prefixo="solucao"):
    """
    Salva resultados em formato .vts (Structured Grid) para ParaView
    """
    # CORREÇÃO: Sem transposição - manter orientação original
    P_2D = P.reshape((nx, ny))
    K_2D = K
    vx_2D = v

    print(f"Salvando {prefixo} - Dimensões: P={P_2D.shape}, K={K_2D.shape}, v={vx_2D.shape}")

    # Salvar em formato .vts
    with open(f"{prefixo}.vts", 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {len(xc) - 1} 0 {len(yc) - 1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {len(xc) - 1} 0 {len(yc) - 1} 0 0">\n')

        # Pontos da malha
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{xc[i]:.6f} {yc[j]:.6f} 0.0 ')
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')

        # Dados dos pontos
        f.write('      <PointData>\n')

        # Pressão
        f.write('        <DataArray type="Float32" Name="pressure" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{P_2D[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')

        # Permeabilidade (em mD para visualização)
        f.write('        <DataArray type="Float32" Name="permeability" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{(K_2D[i, j] / 9.869233e-16):.6e} ')  # Converter para mD
        f.write('\n        </DataArray>\n')

        # Velocidade
        f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{vx_2D[i, j]:.6e} 0.0 0.0 ')
        f.write('\n        </DataArray>\n')

        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </StructuredGrid>\n')
        f.write('</VTKFile>\n')


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

# Carregar dados SPE10
kx, ky, kz = carregar_dados_spe10('spe_perm.dat')

# Manter cópia em mD para visualização
kx_mD = kx.copy()

# Converter para m² para cálculos
kx = np.maximum(kx * 9.869233e-16, 1e-20)

# Selecionar camadas
K33_orig = kx[:, :, 32]  # Camada 33 - forma (60, 220)
K35_orig = kx[:, :, 34]  # Camada 35 - forma (60, 220)
K33_orig_mD = kx_mD[:, :, 32]  # Para visualização
K35_orig_mD = kx_mD[:, :, 34]

# Criar malha
x, y = CreateMesh(Lx, Ly, Nx, Ny)

# CORREÇÃO: Preparar permeabilidades com orientação correta
K33_corr = np.flipud(K33_orig.T)  # Transpor + flip -> (220, 60)
K35_corr = np.flipud(K35_orig.T)
K33_corr_mD = np.flipud(K33_orig_mD.T)  # Para visualização
K35_corr_mD = np.flipud(K35_orig_mD.T)

# Interpolar para as dimensões da malha de volumes finitos (nx x ny)
K33_interp = zoom(K33_corr, (nx / K33_corr.shape[0], ny / K33_corr.shape[1]), order=1)
K35_interp = zoom(K35_corr, (nx / K35_corr.shape[0], ny / K35_corr.shape[1]), order=1)
K33_interp_mD = zoom(K33_corr_mD, (nx / K33_corr_mD.shape[0], ny / K33_corr_mD.shape[1]), order=1)
K35_interp_mD = zoom(K35_corr_mD, (nx / K35_corr_mD.shape[0], ny / K35_corr_mD.shape[1]), order=1)

K33_interp = np.maximum(K33_interp, 1e-20)
K35_interp = np.maximum(K35_interp, 1e-20)


# Funções de permeabilidade
def K_func_33(x, y):
    i = min(int(x / Lx * (K33_interp.shape[0] - 1)), K33_interp.shape[0] - 1)
    j = min(int(y / Ly * (K33_interp.shape[1] - 1)), K33_interp.shape[1] - 1)
    return K33_interp[i, j]


def K_func_35(x, y):
    i = min(int(x / Lx * (K35_interp.shape[0] - 1)), K35_interp.shape[0] - 1)
    j = min(int(y / Ly * (K35_interp.shape[1] - 1)), K35_interp.shape[1] - 1)
    return K35_interp[i, j]


print("=" * 50)
print("SOLUÇÃO ESTACIONÁRIA - EQUAÇÃO DE DARCY")
print(f"Domínio: {Lx} ft x {Ly} ft (SPE10)")
print(f"Malha: {Nx} x {Ny} pontos, {nx} x {ny} volumes")
print("=" * 50)

# Resolver Camada 33
P33, xc, yc, dx, dy = solve_darcy_case(K_func_33, "SPE10 - Camada 33", 1e6, 0.0)

# Calcular gradiente e velocidade
P33_2D = P33.reshape((nx, ny))
gradP_x33 = np.gradient(P33_2D, dx[0], axis=0)
v33 = -K33_interp * gradP_x33 / mu

print(f"Dimensões Camada 33: P={P33_2D.shape}, K={K33_interp.shape}, v={v33.shape}")

# Resolver Camada 35
P35, _, _, _, _ = solve_darcy_case(K_func_35, "SPE10 - Camada 35", 1e6, 0.0)

# Calcular gradiente e velocidade
P35_2D = P35.reshape((nx, ny))
gradP_x35 = np.gradient(P35_2D, dx[0], axis=0)
v35 = -K35_interp * gradP_x35 / mu

print(f"Dimensões Camada 35: P={P35_2D.shape}, K={K35_interp.shape}, v={v35.shape}")

# Salvar resultados para ParaView
print("\nSalvando resultados para ParaView...")
salvar_para_paraview(xc, yc, P33, K33_interp_mD, v33, "camada_33")
salvar_para_paraview(xc, yc, P35, K35_interp_mD, v35, "camada_35")

# Resumo final
print("\n" + "=" * 50)
print("RESUMO")
print("=" * 50)
print(f"Camada 33 - Permeabilidade média: {np.mean(K33_interp_mD):.1f} mD")
print(f"Camada 35 - Permeabilidade média: {np.mean(K35_interp_mD):.1f} mD")
print(f"Razão K33/K35: {np.mean(K33_interp_mD) / np.mean(K35_interp_mD):.2f}")
print(f"Arquivos gerados: camada_33.vts, camada_35.vts")
print("✅ Simulação concluída!")