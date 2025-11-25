import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time

# =============================================================================
# PARÂMETROS CORRETOS SPE10 - DIMENSÕES CORRIGIDAS
# =============================================================================
Lx, Nx = 670.56, 220   # 220 células na direção X (largura) - CORRIGIDO
Ly, Ny = 365.76, 60    # 60 células na direção Y (altura) - CORRIGIDO
nx, ny = Nx, Ny        
mu = 1.0               # Viscosidade [cP]

# Dimensões originais do SPE10 (CORRETAS)
Nx_spe, Ny_spe, Nz_spe = 60, 220, 85  # ATENÇÃO: SPE10 tem 60x220, mas vamos transpor

def carregar_dados_spe10_corrigido(arquivo='spe_perm.dat'):
    """
    Carrega dados SPE10 no formato correto e faz transpose para orientação correta
    """
    total_values = Nx_spe * Ny_spe * Nz_spe
    
    try:
        with open(arquivo, 'r') as file:
            all_values = np.array(file.read().split(), dtype=float)
        
        print(f"Total de valores lidos: {len(all_values)}")
        print(f"Esperado: {total_values * 3}")
        
        if len(all_values) != total_values * 3:
            print("AVISO: Número de valores não corresponde ao esperado!")
        
        # Inicializar arrays
        kx = np.zeros((Nx_spe, Ny_spe, Nz_spe))
        ky = np.zeros((Nx_spe, Ny_spe, Nz_spe))
        kz = np.zeros((Nx_spe, Ny_spe, Nz_spe))
        
        # Carregar Kx (primeiro bloco)
        idx = 0
        for k in range(Nz_spe):
            for j in range(Ny_spe):
                for i in range(Nx_spe):
                    kx[i, j, k] = all_values[idx]
                    idx += 1
        
        # Carregar Ky (segundo bloco)
        for k in range(Nz_spe):
            for j in range(Ny_spe):
                for i in range(Nx_spe):
                    ky[i, j, k] = all_values[idx]
                    idx += 1
        
        # Carregar Kz (terceiro bloco)
        for k in range(Nz_spe):
            for j in range(Ny_spe):
                for i in range(Nx_spe):
                    kz[i, j, k] = all_values[idx]
                    idx += 1
        
        print(f"Permeabilidades carregadas - Kx: {kx.shape}")
        
        # TRANSPOSE para orientação correta: (60, 220) -> (220, 60)
        kx = np.transpose(kx, (1, 0, 2))  # Agora (220, 60, 85)
        ky = np.transpose(ky, (1, 0, 2))
        kz = np.transpose(kz, (1, 0, 2))
        
        print(f"Após transpose - Kx: {kx.shape}")
        
        return kx, ky, kz
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo {arquivo} não encontrado!")
        return None, None, None

def CreateMesh(Lx, Ly, Nx, Ny):
    """Cria malha com orientação correta"""
    dx = Lx / Nx
    dy = Ly / Ny
    xc = np.linspace(dx/2, Lx - dx/2, Nx)  # Centros das células em X
    yc = np.linspace(dy/2, Ly - dy/2, Ny)  # Centros das células em Y
    return xc, yc, dx, dy

def BuildDarcySystem(xc, yc, dx, dy, K_field, bc_left=1e6, bc_right=0.0):
    """Constrói sistema com orientação correta"""
    Nx, Ny = len(xc), len(yc)
    Nc = Nx * Ny
    
    row, col, data = [], [], []
    rhs = np.zeros(Nc)
    
    def idx(i, j):
        return i * Ny + j  # i varia mais rápido (dimensão X)
    
    for i in range(Nx):      # Direção X (220)
        for j in range(Ny):  # Direção Y (60)
            g = idx(i, j)
            diag_coef = 0.0
            
            # Fluxo na direção x (entre células i e i+1)
            if i < Nx - 1:
                K1, K2 = K_field[i, j], K_field[i+1, j]
                K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_e = K_e * dy / dx / mu
                g_e = idx(i+1, j)
                row.append(g); col.append(g_e); data.append(-Tx_e)
                diag_coef += Tx_e
            
            if i > 0:
                K1, K2 = K_field[i, j], K_field[i-1, j]
                K_w = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_w = K_w * dy / dx / mu
                g_w = idx(i-1, j)
                row.append(g); col.append(g_w); data.append(-Tx_w)
                diag_coef += Tx_w
            
            # Fluxo na direção y (entre células j e j+1)
            if j < Ny - 1:
                K1, K2 = K_field[i, j], K_field[i, j+1]
                K_n = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_n = K_n * dx / dy / mu
                g_n = idx(i, j+1)
                row.append(g); col.append(g_n); data.append(-Ty_n)
                diag_coef += Ty_n
            
            if j > 0:
                K1, K2 = K_field[i, j], K_field[i, j-1]
                K_s = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_s = K_s * dx / dy / mu
                g_s = idx(i, j-1)
                row.append(g); col.append(g_s); data.append(-Ty_s)
                diag_coef += Ty_s
            
            # Condições de contorno (agora na direção X correta)
            if i == 0:  # Contorno esquerdo
                T_bc = K_field[i, j] * dy / (0.5 * dx) / mu
                rhs[g] += T_bc * bc_left
                diag_coef += T_bc
            elif i == Nx - 1:  # Contorno direito
                T_bc = K_field[i, j] * dy / (0.5 * dx) / mu
                rhs[g] += T_bc * bc_right
                diag_coef += T_bc
            
            row.append(g); col.append(g); data.append(diag_coef)
    
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc)).tocsr()
    return A, rhs

def salvar_para_paraview_corrigido(xc, yc, P_2D, K_mD, vx, vy, prefixo="solucao"):
    """
    Salva resultados em formato .vts com orientação correta
    """
    Nx, Ny = len(xc), len(yc)
    
    print(f"Salvando {prefixo}.vts - Dimensões: {Nx}x{Ny} (LARGURA x ALTURA)")
    
    # Criar coordenadas dos pontos (Nx+1, Ny+1 pontos)
    x_points = np.linspace(0, Lx, Nx + 1)  # Direção X (largura)
    y_points = np.linspace(0, Ly, Ny + 1)  # Direção Y (altura)
    
    with open(f"{prefixo}.vts", 'w') as f:
        # Cabeçalho XML
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {Nx} 0 {Ny} 0 0">\n')
        f.write(f'    <Piece Extent="0 {Nx} 0 {Ny} 0 0">\n')
        
        # Pontos da malha (coordenadas dos vértices)
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for j in range(Ny + 1):    # Y primeiro (altura)
            for i in range(Nx + 1): # X depois (largura)
                f.write(f'{x_points[i]:.6f} {y_points[j]:.6f} 0.0 ')
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')
        
        # Dados nas células
        f.write('      <CellData>\n')
        
        # Pressão
        f.write('        <DataArray type="Float32" Name="Pressure" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{P_2D[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        # Permeabilidade (em mD)
        f.write('        <DataArray type="Float32" Name="Permeability" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{K_mD[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        # Velocidade X
        f.write('        <DataArray type="Float32" Name="Velocity_X" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{vx[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        # Velocidade Y
        f.write('        <DataArray type="Float32" Name="Velocity_Y" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{vy[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        # Magnitude da velocidade
        velocity_magnitude = np.sqrt(vx**2 + vy**2)
        f.write('        <DataArray type="Float32" Name="Velocity_Magnitude" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{velocity_magnitude[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        f.write('      </CellData>\n')
        f.write('    </Piece>\n')
        f.write('  </StructuredGrid>\n')
        f.write('</VTKFile>\n')
    
    print(f"Arquivo {prefixo}.vts salvo com sucesso!")

# =============================================================================
# EXECUÇÃO PRINCIPAL CORRIGIDA
# =============================================================================

print("=" * 60)
print("SPE10 - ORIENTAÇÃO CORRETA: 220x60 (LARGURA x ALTURA)")
print("=" * 60)

# Carregar dados
kx_mD, ky_mD, kz_mD = carregar_dados_spe10_corrigido('spe_perm.dat')

if kx_mD is None:
    print("Não foi possível carregar os dados.")
    exit()

# Converter para m² (1 mD = 9.869233e-16 m²)
kx = kx_mD * 9.869233e-16
kx = np.maximum(kx, 1e-20)

# Selecionar camadas (já na orientação correta 220x60)
K33_mD = kx_mD[:, :, 32]  # Camada 33 - forma (220, 60)
K35_mD = kx_mD[:, :, 34]  # Camada 35 - forma (220, 60)

K33 = kx[:, :, 32]  # Para cálculos em m²
K35 = kx[:, :, 34]

print(f"\nCamada 33: {K33.shape} - ORIENTAÇÃO CORRETA")
print(f"Camada 35: {K35.shape} - ORIENTAÇÃO CORRETA")
print(f"Permeabilidade Camada 33: {np.min(K33_mD):.2e} a {np.max(K33_mD):.2e} mD")
print(f"Permeabilidade Camada 35: {np.min(K35_mD):.2e} a {np.max(K35_mD):.2e} mD")

# Criar malha com orientação correta
xc, yc, dx, dy = CreateMesh(Lx, Ly, Nx, Ny)
print(f"\nMalha criada: {len(xc)} x {len(yc)} células")
print(f"Largura (X): {Lx} ft, {Nx} células, dx = {dx:.2f} ft")
print(f"Altura (Y): {Ly} ft, {Ny} células, dy = {dy:.2f} ft")

# Resolver para Camada 33
print("\n" + "=" * 40)
print("RESOLVENDO CAMADA 33")
print("=" * 40)

A33, rhs33 = BuildDarcySystem(xc, yc, dx, dy, K33, bc_left=1e6, bc_right=0.0)
P33 = scipy.sparse.linalg.spsolve(A33, rhs33)
P33_2D = P33.reshape((Nx, Ny))

# Calcular velocidade
gradP_x33 = np.gradient(P33_2D, dx, axis=0)
gradP_y33 = np.gradient(P33_2D, dy, axis=1)
vx33 = -K33 * gradP_x33 / mu
vy33 = -K33 * gradP_y33 / mu

print(f"Solução Camada 33: {P33_2D.shape}")
print(f"Pressão: {np.min(P33_2D):.2e} a {np.max(P33_2D):.2e} Pa")

# Resolver para Camada 35
print("\n" + "=" * 40)
print("RESOLVENDO CAMADA 35")
print("=" * 40)

A35, rhs35 = BuildDarcySystem(xc, yc, dx, dy, K35, bc_left=1e6, bc_right=0.0)
P35 = scipy.sparse.linalg.spsolve(A35, rhs35)
P35_2D = P35.reshape((Nx, Ny))

# Calcular velocidade
gradP_x35 = np.gradient(P35_2D, dx, axis=0)
gradP_y35 = np.gradient(P35_2D, dy, axis=1)
vx35 = -K35 * gradP_x35 / mu
vy35 = -K35 * gradP_y35 / mu

print(f"Solução Camada 35: {P35_2D.shape}")
print(f"Pressão: {np.min(P35_2D):.2e} a {np.max(P35_2D):.2e} Pa")

# Salvar resultados para ParaView
print("\n" + "=" * 40)
print("SALVANDO PARA PARAVIEW")
print("=" * 40)

salvar_para_paraview_corrigido(xc, yc, P33_2D, K33_mD, vx33, vy33, "camada_33_horizontal")
salvar_para_paraview_corrigido(xc, yc, P35_2D, K35_mD, vx35, vy35, "camada_35_horizontal")

# Resumo final
print("\n" + "=" * 60)
print("RESUMO FINAL - ORIENTAÇÃO CORRIGIDA")
print("=" * 60)
print(f"Domínio: {Lx} ft (LARGURA) x {Ly} ft (ALTURA)")
print(f"Malha: {Nx} x {Ny} células")
print(f"Camada 33 - Permeabilidade média: {np.mean(K33_mD):.1f} mD")
print(f"Camada 35 - Permeabilidade média: {np.mean(K35_mD):.1f} mD")
print(f"Razão de aspecto: {Lx/Ly:.2f}:1 (LARGURA:ALTURA)")
print("Arquivos gerados: camada_33_horizontal.vts, camada_35_horizontal.vts")
print("=" * 60)

# Verificação visual
print("\nVERIFICAÇÃO VISUAL:")
print("• No ParaView, o domínio deve aparecer HORIZONTAL")
print("• Dimensões: aproximadamente 670 ft (largura) x 365 ft (altura)")
print("• Proporção: quase 2:1 (mais largo que alto)")
