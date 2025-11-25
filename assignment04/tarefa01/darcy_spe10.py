import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time

# =============================================================================
# CORREÇÕES APLICADAS
# 1. Viscosidade corrigida: mu = 8.9e-4 Pa·s (água)
# 2. Dimensões da interpolação corrigidas
# 3. Cálculo de vazão corrigido
# 4. Verificação de formas dos arrays
# =============================================================================

# Domínio em metros
Lx, Ly = 670.56, 365.76  # Domínio em metros
Nx, Ny = 60, 220         # Número de pontos na malha
nx, ny = Nx - 1, Ny - 1  # Número de volumes

# Parâmetros físicos CORRIGIDOS
mu = 8.9e-4  # Pa·s (viscosidade da água a ~25°C)

# Condições de contorno especificadas
u_b = 0.036 / 3600  # m/h -> m/s (0.036 m/h conforme especificado originalmente)
p_b = 6e6  # 6 MPa -> Pa

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

def BuildDarcySystemSlab(x, y, K_func, u_inflow=0.0, p_outflow=0.0):
    N1, N2 = len(x), len(y)
    n1, n2 = N1 - 1, N2 - 1
    Nc = n1 * n2

    dx = x[1:] - x[:-1]
    dy = y[1:] - y[:-1]
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])

    row, col, data = [], [], []
    rhs = np.zeros(Nc)
    
    # Primeira passada: construir matriz sem condições de Dirichlet
    for i in range(n1):
        for j in range(n2):
            g = jglob(i, j, n1)
            diag_coef = 0.0

            # Pular células de contorno Dirichlet (serão tratadas depois)
            if i == n1 - 1:  # Saída - condição Dirichlet
                continue

            # Fluxo na direção x (LESTE-OESTE)
            if i < n1 - 1:
                # Se vizinho leste é Dirichlet, tratar separadamente
                if i + 1 == n1 - 1:
                    K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i + 1], yc[j])
                    K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                    T_coef_e = K_e / mu
                    Tx_e = T_coef_e * dy[j] / (0.5 * dx[i] + 0.5 * dx[i + 1])
                    # Contribuição para rhs devido à condição Dirichlet
                    rhs[g] += Tx_e * p_outflow
                    diag_coef += Tx_e
                else:
                    K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i + 1], yc[j])
                    K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                    T_coef_e = K_e / mu
                    Tx_e = T_coef_e * dy[j] / (0.5 * dx[i] + 0.5 * dx[i + 1])
                    g_e = jglob(i + 1, j, n1)
                    row.append(g); col.append(g_e); data.append(-Tx_e)
                    diag_coef += Tx_e

            if i > 0:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i - 1], yc[j])
                K_w = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                T_coef_w = K_w / mu
                Tx_w = T_coef_w * dy[j] / (0.5 * dx[i] + 0.5 * dx[i - 1])
                g_w = jglob(i - 1, j, n1)
                row.append(g); col.append(g_w); data.append(-Tx_w)
                diag_coef += Tx_w

            # Fluxo na direção y (NORTE-SUL)
            if j < n2 - 1:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i], yc[j + 1])
                K_n = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                T_coef_n = K_n / mu
                Ty_n = T_coef_n * dx[i] / (0.5 * dy[j] + 0.5 * dy[j + 1])
                g_n = jglob(i, j + 1, n1)
                row.append(g); col.append(g_n); data.append(-Ty_n)
                diag_coef += Ty_n

            if j > 0:
                K1, K2 = K_func(xc[i], yc[j]), K_func(xc[i], yc[j - 1])
                K_s = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                T_coef_s = K_s / mu
                Ty_s = T_coef_s * dx[i] / (0.5 * dy[j] + 0.5 * dy[j - 1])
                g_s = jglob(i, j - 1, n1)
                row.append(g); col.append(g_s); data.append(-Ty_s)
                diag_coef += Ty_s

            # CONDIÇÕES DE CONTORNO CORRETAS
            # Entrada (x=0): fluxo especificado (Neumann)
            if i == 0:
                rhs[g] += u_inflow * dy[j]

            row.append(g)
            col.append(g)
            data.append(diag_coef)

    # Segunda passada: adicionar condições Dirichlet na saída
    for j in range(n2):
        g_dirichlet = jglob(n1 - 1, j, n1)
        row.append(g_dirichlet)
        col.append(g_dirichlet)
        data.append(1.0)
        rhs[g_dirichlet] = p_outflow

    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc)).tocsr()
    return A, rhs, xc, yc, dx, dy

def solve_darcy_case_slab(K_func, caso_nome, u_inflow=0.0, p_outflow=0.0):
    print(f"Resolvendo: {caso_nome}")
    A, rhs, xc, yc, dx, dy = BuildDarcySystemSlab(x, y, K_func, u_inflow, p_outflow)

    t0 = time.time()
    P = scipy.sparse.linalg.spsolve(A, rhs)
    print(f"Tempo: {time.time() - t0:.3f} s")

    return P, xc, yc, dx, dy

def calcular_velocidade(P, K, dx, dy, mu):
    """Calcular campo de velocidade a partir do campo de pressão usando K/μ"""
    P_2D = P.reshape((nx, ny))
    
    # Gradiente de pressão
    gradP_x = np.gradient(P_2D, dx, axis=0)
    gradP_y = np.gradient(P_2D, dy, axis=1)
    
    # Lei de Darcy: v = -(K/μ) ∇p
    vx = -(K / mu) * gradP_x
    vy = -(K / mu) * gradP_y
    
    # Magnitude da velocidade
    v_mag = np.sqrt(vx**2 + vy**2)
    
    return vx, vy, v_mag, P_2D

def salvar_para_paraview(xc, yc, P, K, vx, vy, v_mag, prefixo="solucao"):
    """
    Salva resultados em formato .vts (Structured Grid) para ParaView
    """
    P_2D = P.reshape((nx, ny))
    K_2D = K

    print(f"Salvando {prefixo}")
    print(f"  Dimensões: P={P_2D.shape}, K={K_2D.shape}")
    print(f"  Range Permeabilidade: [{K_2D.min()/9.869233e-16:.2e}, {K_2D.max()/9.869233e-16:.2e}] mD")
    print(f"  Range Pressão: [{P_2D.min()/1e6:.2f}, {P_2D.max()/1e6:.2f}] MPa")

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

        # Pressão (em MPa para visualização)
        f.write('        <DataArray type="Float32" Name="pressure" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{P_2D[i, j] / 1e6:.6f} ')
        f.write('\n        </DataArray>\n')

        # Permeabilidade (em mD para visualização)
        f.write('        <DataArray type="Float32" Name="permeability" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{(K_2D[i, j] / 9.869233e-16):.6e} ')
        f.write('\n        </DataArray>\n')

        # Coeficiente K/μ
        f.write('        <DataArray type="Float32" Name="K_over_mu" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{(K_2D[i, j] / mu):.6e} ')
        f.write('\n        </DataArray>\n')

        # Velocidade (componentes e magnitude)
        f.write('        <DataArray type="Float32" Name="velocity" NumberOfComponents="3" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{vx[i, j]:.6e} {vy[i, j]:.6e} 0.0 ')
        f.write('\n        </DataArray>\n')
        
        f.write('        <DataArray type="Float32" Name="velocity_magnitude" format="ascii">\n')
        for j in range(len(yc)):
            for i in range(len(xc)):
                f.write(f'{v_mag[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')

        f.write('      </PointData>\n')
        f.write('    </Piece>\n')
        f.write('  </StructuredGrid>\n')
        f.write('</VTKFile>\n')

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

# Carregar dados SPE10
print("Carregando dados SPE10...")
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

print(f"Dimensões originais SPE10: K33_orig {K33_orig.shape}, K35_orig {K35_orig.shape}")

# Criar malha
x, y = CreateMesh(Lx, Ly, Nx, Ny)

# Preparar permeabilidades com orientação correta
K33_corr = np.flipud(K33_orig.T)  # Transpor + flip -> (220, 60)
K35_corr = np.flipud(K35_orig.T)
K33_corr_mD = np.flipud(K33_orig_mD.T)
K35_corr_mD = np.flipud(K35_orig_mD.T)

print(f"Dimensões após transposição: K33_corr {K33_corr.shape}")

# Interpolar para as dimensões da malha de volumes finitos (nx x ny)
# CORREÇÃO: As dimensões estão (220, 60) e queremos (nx, ny) = (59, 219)
K33_interp = zoom(K33_corr, (nx / K33_corr.shape[0], ny / K33_corr.shape[1]), order=1)
K35_interp = zoom(K35_corr, (nx / K35_corr.shape[0], ny / K35_corr.shape[1]), order=1)
K33_interp_mD = zoom(K33_corr_mD, (nx / K33_corr_mD.shape[0], ny / K33_corr_mD.shape[1]), order=1)
K35_interp_mD = zoom(K35_corr_mD, (nx / K35_corr_mD.shape[0], ny / K35_corr_mD.shape[1]), order=1)

print(f"Dimensões após interpolação: K33_interp {K33_interp.shape}")

# Garantir formas corretas
K33_interp = K33_interp[:nx, :ny]
K35_interp = K35_interp[:nx, :ny]
K33_interp_mD = K33_interp_mD[:nx, :ny]
K35_interp_mD = K35_interp_mD[:nx, :ny]

K33_interp = np.maximum(K33_interp, 1e-20)
K35_interp = np.maximum(K35_interp, 1e-20)

print(f"Permeabilidade Camada 33: [{K33_interp_mD.min():.2e}, {K33_interp_mD.max():.2e}] mD")
print(f"Permeabilidade Camada 35: [{K35_interp_mD.min():.2e}, {K35_interp_mD.max():.2e}] mD")

# Funções de permeabilidade
def K_func_33(xp, yp):
    i = min(int(xp / Lx * (K33_interp.shape[0] - 1)), K33_interp.shape[0] - 1)
    j = min(int(yp / Ly * (K33_interp.shape[1] - 1)), K33_interp.shape[1] - 1)
    return K33_interp[i, j]

def K_func_35(xp, yp):
    i = min(int(xp / Lx * (K35_interp.shape[0] - 1)), K35_interp.shape[0] - 1)
    j = min(int(yp / Ly * (K35_interp.shape[1] - 1)), K35_interp.shape[1] - 1)
    return K35_interp[i, j]

print("=" * 70)
print("SOLUÇÃO DO PROBLEMA ELÍPTICO 2D - EQUAÇÃO DE DARCY")
print("=" * 70)
print(f"Domínio: Ω = [0, {Lx}] × [0, {Ly}] m")
print(f"Malha: {Nx} × {Ny} pontos, {nx} × {ny} volumes")
print(f"Viscosidade: μ = {mu:.1e} Pa·s")  # AGORA CORRETA!
print(f"Coeficiente: K/μ")
print(f"Termo fonte: nulo")
print(f"Condições de contorno slab:")
print(f"  - Entrada (x=0): u·n = {u_b*3600:.3f} m/h")
print(f"  - Saída (x=Lx): p = {p_b/1e6:.1f} MPa")
print(f"  - Topo/fundo (y=0, y=Ly): u·n = 0")
print("=" * 70)

# Resolver Camada 33 com condições slab
print("\n>>> CAMADA 33 <<<")
P33, xc, yc, dx, dy = solve_darcy_case_slab(K_func_33, "SPE10 - Camada 33", u_b, p_b)

# Calcular campo de velocidade
vx33, vy33, v_mag33, P33_2D = calcular_velocidade(P33, K33_interp, dx, dy, mu)

print(f"Pressão: [{P33_2D.min()/1e6:.2f}, {P33_2D.max()/1e6:.2f}] MPa")
print(f"Velocidade máxima: {v_mag33.max()*3600:.3f} m/h")
print(f"Coeficiente K/μ médio: {np.mean(K33_interp/mu):.2e} m²/(Pa·s)")

# Resolver Camada 35 com condições slab
print("\n>>> CAMADA 35 <<<")
P35, _, _, dx35, dy35 = solve_darcy_case_slab(K_func_35, "SPE10 - Camada 35", u_b, p_b)

# Calcular campo de velocidade
vx35, vy35, v_mag35, P35_2D = calcular_velocidade(P35, K35_interp, dx35, dy35, mu)

print(f"Pressão: [{P35_2D.min()/1e6:.2f}, {P35_2D.max()/1e6:.2f}] MPa")
print(f"Velocidade máxima: {v_mag35.max()*3600:.3f} m/h") 
print(f"Coeficiente K/μ médio: {np.mean(K35_interp/mu):.2e} m²/(Pa·s)")

# Cálculo CORRETO da vazão
# vx está no centro das células, então na entrada usamos i=0, na saída i=nx-1
Q_out_33 = np.sum(vx33[-1, :] * dy)  # Última coluna (saída)
Q_in_33 = np.sum(vx33[0, :] * dy)    # Primeira coluna (entrada)

Q_out_35 = np.sum(vx35[-1, :] * dy35)
Q_in_35 = np.sum(vx35[0, :] * dy35)

print(f"\nBalanço de vazão Camada 33:")
print(f"  Entrada: {Q_in_33:.6e} m³/s, Saída: {Q_out_33:.6e} m³/s")
print(f"  Diferença: {abs(Q_in_33 - Q_out_33):.2e} m³/s")

print(f"Balanço de vazão Camada 35:")
print(f"  Entrada: {Q_in_35:.6e} m³/s, Saída: {Q_out_35:.6e} m³/s")
print(f"  Diferença: {abs(Q_in_35 - Q_out_35):.2e} m³/s")

# Salvar resultados para ParaView
print("\nSalvando resultados para ParaView...")
salvar_para_paraview(xc, yc, P33, K33_interp, vx33, vy33, v_mag33, "camada_33_elliptic")
salvar_para_paraview(xc, yc, P35, K35_interp, vx35, vy35, v_mag35, "camada_35_elliptic")

# Resumo final
print("\n" + "=" * 70)
print("RESUMO - PROBLEMA ELÍPTICO 2D")
print("=" * 70)
print(f"Camada 33:")
print(f"  - Permeabilidade: {np.mean(K33_interp_mD):.1f} mD (avg)")
print(f"  - K/μ médio: {np.mean(K33_interp/mu):.2e} m²/(Pa·s)")
print(f"  - Vazão saída: {Q_out_33:.3e} m³/s")

print(f"Camada 35:")
print(f"  - Permeabilidade: {np.mean(K35_interp_mD):.1f} mD (avg)")
print(f"  - K/μ médio: {np.mean(K35_interp/mu):.2e} m²/(Pa·s)")
print(f"  - Vazão saída: {Q_out_35:.3e} m³/s")

print(f"Razão K33/K35: {np.mean(K33_interp_mD) / np.mean(K35_interp_mD):.2f}")
print(f"Arquivos gerados: camada_33_elliptic.vts, camada_35_elliptic.vts")
print("Simulação concluída!")
