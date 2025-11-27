import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# =============================================================================
# PARÂMETROS CORRETOS SPE10 - DIMENSÕES CORRIGIDAS
# =============================================================================
Lx, Nx = 670.56, 220   # 220 células na direção X (largura)
Ly, Ny = 365.76, 60    # 60 células na direção Y (altura)
Lz, Nz = 85.0, 85      # 85 camadas na direção Z - NOVO: adicionar dimensão Z
nx, ny, nz = Nx, Ny, Nz
mu = 1.0               # Viscosidade [cP]

# Dimensões originais do SPE10
Nx_spe, Ny_spe, Nz_spe = 60, 220, 85

def carregar_dados_spe10_corrigido(arquivo='spe_perm.dat'):
    """
    Carrega dados SPE10 no formato correto
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
        
        print(f"Permeabilidades carregadas - Kx: {kx.shape}, Ky: {ky.shape}, Kz: {kz.shape}")
        
        # TRANSPOSE para orientação correta: (60, 220, 85) -> (220, 60, 85)
        kx = np.transpose(kx, (1, 0, 2))
        ky = np.transpose(ky, (1, 0, 2))
        kz = np.transpose(kz, (1, 0, 2))
        
        print(f"Após transpose - Kx: {kx.shape}, Kz: {kz.shape}")
        
        return kx, ky, kz
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo {arquivo} não encontrado!")
        return None, None, None

def CreateMesh(Lx, Ly, Nx, Ny):
    """Cria malha com orientação correta"""
    dx = Lx / Nx
    dy = Ly / Ny
    xc = np.linspace(dx/2, Lx - dx/2, Nx)
    yc = np.linspace(dy/2, Ly - dy/2, Ny)
    return xc, yc, dx, dy

def BuildDarcySystem_Kz(xc, yc, dx, dy, Kz_field, bc_left=1e6, bc_right=0.0):
    """
    Constrói sistema para permeabilidade Kz (direção vertical)
    MUDANÇA: Usar Kz em vez de Kx
    """
    Nx, Ny = len(xc), len(yc)
    Nc = Nx * Ny
    
    row, col, data = [], [], []
    rhs = np.zeros(Nc)
    
    def idx(i, j):
        return i * Ny + j
    
    for i in range(Nx):
        for j in range(Ny):
            g = idx(i, j)
            diag_coef = 0.0
            
            # Fluxo na direção x (entre células i e i+1) - usando Kz
            if i < Nx - 1:
                K1, K2 = Kz_field[i, j], Kz_field[i+1, j]  # MUDANÇA: Kz em vez de Kx
                K_e = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_e = K_e * dy / dx / mu
                g_e = idx(i+1, j)
                row.append(g); col.append(g_e); data.append(-Tx_e)
                diag_coef += Tx_e
            
            if i > 0:
                K1, K2 = Kz_field[i, j], Kz_field[i-1, j]  # MUDANÇA: Kz em vez de Kx
                K_w = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Tx_w = K_w * dy / dx / mu
                g_w = idx(i-1, j)
                row.append(g); col.append(g_w); data.append(-Tx_w)
                diag_coef += Tx_w
            
            # Fluxo na direção y (entre células j e j+1) - usando Kz
            if j < Ny - 1:
                K1, K2 = Kz_field[i, j], Kz_field[i, j+1]  # MUDANÇA: Kz em vez de Kx
                K_n = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_n = K_n * dx / dy / mu
                g_n = idx(i, j+1)
                row.append(g); col.append(g_n); data.append(-Ty_n)
                diag_coef += Ty_n
            
            if j > 0:
                K1, K2 = Kz_field[i, j], Kz_field[i, j-1]  # MUDANÇA: Kz em vez de Kx
                K_s = 2 * K1 * K2 / (K1 + K2 + 1e-30)
                Ty_s = K_s * dx / dy / mu
                g_s = idx(i, j-1)
                row.append(g); col.append(g_s); data.append(-Ty_s)
                diag_coef += Ty_s
            
            # Condições de contorno
            if i == 0:
                T_bc = Kz_field[i, j] * dy / (0.5 * dx) / mu  # MUDANÇA: Kz em vez de Kx
                rhs[g] += T_bc * bc_left
                diag_coef += T_bc
            elif i == Nx - 1:
                T_bc = Kz_field[i, j] * dy / (0.5 * dx) / mu  # MUDANÇA: Kz em vez de Kx
                rhs[g] += T_bc * bc_right
                diag_coef += T_bc
            
            row.append(g); col.append(g); data.append(diag_coef)
    
    A = scipy.sparse.coo_matrix((data, (row, col)), shape=(Nc, Nc)).tocsr()
    return A, rhs

def salvar_para_paraview_corrigido(xc, yc, P_2D, K_mD, vx, vy, prefixo="solucao"):
    """
    Salva resultados em formato .vts
    """
    Nx, Ny = len(xc), len(yc)
    
    print(f"Salvando {prefixo}.vts - Dimensões: {Nx}x{Ny}")
    
    x_points = np.linspace(0, Lx, Nx + 1)
    y_points = np.linspace(0, Ly, Ny + 1)
    
    with open(f"{prefixo}.vts", 'w') as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {Nx} 0 {Ny} 0 0">\n')
        f.write(f'    <Piece Extent="0 {Nx} 0 {Ny} 0 0">\n')
        
        f.write('      <Points>\n')
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for j in range(Ny + 1):
            for i in range(Nx + 1):
                f.write(f'{x_points[i]:.6f} {y_points[j]:.6f} 0.0 ')
        f.write('\n        </DataArray>\n')
        f.write('      </Points>\n')
        
        f.write('      <CellData>\n')
        
        f.write('        <DataArray type="Float32" Name="Pressure" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{P_2D[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        f.write('        <DataArray type="Float32" Name="Permeability_Kz" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{K_mD[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        f.write('        <DataArray type="Float32" Name="Velocity_X" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{vx[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
        f.write('        <DataArray type="Float32" Name="Velocity_Y" format="ascii">\n')
        for j in range(Ny):
            for i in range(Nx):
                f.write(f'{vy[i, j]:.6e} ')
        f.write('\n        </DataArray>\n')
        
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
# EXECUÇÃO PRINCIPAL - CAMADA 85 Kz
# =============================================================================

print("=" * 60)
print("SPE10 - CAMADA 85 - PERMEABILIDADE Kz")
print("=" * 60)

# Carregar dados
kx_mD, ky_mD, kz_mD = carregar_dados_spe10_corrigido('spe_perm.dat')

if kz_mD is None:
    print("Não foi possível carregar os dados.")
    exit()

# Converter para m² (1 mD = 9.869233e-16 m²)
kz = kz_mD * 9.869233e-16
kz = np.maximum(kz, 1e-20)

# MUDANÇA PRINCIPAL: Selecionar camada 85 em Kz
camada_alvo = 84  # Índice 84 = camada 85 (0-based indexing)
K85_mD = kz_mD[:, :, camada_alvo]  # Camada 85 - forma (220, 60)
K85 = kz[:, :, camada_alvo]        # Para cálculos em m²

print(f"\nCamada 85 - Kz: {K85.shape} - ORIENTAÇÃO CORRETA")
print(f"Permeabilidade Kz - Camada 85: {np.min(K85_mD):.2e} a {np.max(K85_mD):.2e} mD")
print(f"Permeabilidade média: {np.mean(K85_mD):.1f} mD")

# Criar malha
xc, yc, dx, dy = CreateMesh(Lx, Ly, Nx, Ny)
print(f"\nMalha criada: {len(xc)} x {len(yc)} células")

# Resolver para Camada 85 - Kz
print("\n" + "=" * 40)
print("RESOLVENDO CAMADA 85 - Kz")
print("=" * 40)

# MUDANÇA: Usar BuildDarcySystem_Kz em vez de BuildDarcySystem
A85, rhs85 = BuildDarcySystem_Kz(xc, yc, dx, dy, K85, bc_left=1e6, bc_right=0.0)
P85 = scipy.sparse.linalg.spsolve(A85, rhs85)
P85_2D = P85.reshape((Nx, Ny))

# Calcular velocidade
gradP_x85 = np.gradient(P85_2D, dx, axis=0)
gradP_y85 = np.gradient(P85_2D, dy, axis=1)
vx85 = -K85 * gradP_x85 / mu
vy85 = -K85 * gradP_y85 / mu

print(f"Solução Camada 85 - Kz: {P85_2D.shape}")
print(f"Pressão: {np.min(P85_2D):.2e} a {np.max(P85_2D):.2e} Pa")
print(f"Velocidade máxima: {np.max(np.sqrt(vx85**2 + vy85**2)):.2e} m/s")

# Salvar resultados para ParaView
print("\n" + "=" * 40)
print("SALVANDO PARA PARAVIEW")
print("=" * 40)

salvar_para_paraview_corrigido(xc, yc, P85_2D, K85_mD, vx85, vy85, "camada_85_Kz")

# Comparação com outras camadas (opcional)
print("\n" + "=" * 40)
print("COMPARAÇÃO ENTRE CAMADAS")
print("=" + 40)

# Carregar também Kx e Ky para comparação
kx = kx_mD * 9.869233e-16
kx = np.maximum(kx, 1e-20)
ky = ky_mD * 9.869233e-16
ky = np.maximum(ky, 1e-20)

K85_Kx = kx[:, :, camada_alvo]
K85_Ky = ky[:, :, camada_alvo]

print(f"Camada 85 - Kx: {np.mean(kx_mD[:,:,camada_alvo]):.1f} mD")
print(f"Camada 85 - Ky: {np.mean(ky_mD[:,:,camada_alvo]):.1f} mD")
print(f"Camada 85 - Kz: {np.mean(kz_mD[:,:,camada_alvo]):.1f} mD")

# Anisotropia
anisotropia_xy = np.mean(kx_mD[:,:,camada_alvo]) / np.mean(ky_mD[:,:,camada_alvo])
anisotropia_xz = np.mean(kx_mD[:,:,camada_alvo]) / np.mean(kz_mD[:,:,camada_alvo])
anisotropia_yz = np.mean(ky_mD[:,:,camada_alvo]) / np.mean(kz_mD[:,:,camada_alvo])

print(f"\nAnisotropia Kx/Ky: {anisotropia_xy:.3f}")
print(f"Anisotropia Kx/Kz: {anisotropia_xz:.3f}")
print(f"Anisotropia Ky/Kz: {anisotropia_yz:.3f}")

# Resumo final
print("\n" + "=" * 60)
print("RESUMO FINAL - CAMADA 85 Kz")
print("=" * 60)
print(f"Domínio: {Lx} ft (LARGURA) x {Ly} ft (ALTURA)")
print(f"Malha: {Nx} x {Ny} células")
print(f"Camada analisada: 85 (índice {camada_alvo})")
print(f"Permeabilidade Kz média: {np.mean(K85_mD):.1f} mD")
print(f"Permeabilidade Kz mínima: {np.min(K85_mD):.2e} mD")
print(f"Permeabilidade Kz máxima: {np.max(K85_mD):.2e} mD")
print(f"Velocidade máxima: {np.max(np.sqrt(vx85**2 + vy85**2)):.2e} m/s")
print(f"Arquivo gerado: camada_85_Kz.vts")
print("=" * 60)

# Plotagem simples (opcional)
def plot_simple_comparison(K85_mD, P85_2D, vx85, vy85):
    """Plotagem simples para verificação"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('SPE10 - Camada 85 - Permeabilidade Kz', fontsize=16)
    
    # Permeabilidade Kz
    im1 = axes[0, 0].imshow(K85_mD.T, extent=[0, Lx, 0, Ly], origin='lower', 
                           cmap='viridis', aspect='auto')
    axes[0, 0].set_title('Permeabilidade Kz [mD]')
    axes[0, 0].set_xlabel('X [ft]')
    axes[0, 0].set_ylabel('Y [ft]')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Pressão
    im2 = axes[0, 1].imshow(P85_2D.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='jet', aspect='auto')
    axes[0, 1].set_title('Campo de Pressão [Pa]')
    axes[0, 1].set_xlabel('X [ft]')
    axes[0, 1].set_ylabel('Y [ft]')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Magnitude da velocidade
    velocity_magnitude = np.sqrt(vx85**2 + vy85**2)
    im3 = axes[1, 0].imshow(velocity_magnitude.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='hot', aspect='auto')
    axes[1, 0].set_title('Magnitude da Velocidade [m/s]')
    axes[1, 0].set_xlabel('X [ft]')
    axes[1, 0].set_ylabel('Y [ft]')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Vetores de velocidade
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    skip = 8
    axes[1, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                     vx85[::skip, ::skip], vy85[::skip, ::skip], 
                     color='red', scale=0.5)
    axes[1, 1].set_title('Vetores de Velocidade')
    axes[1, 1].set_xlabel('X [ft]')
    axes[1, 1].set_ylabel('Y [ft]')
    axes[1, 1].set_xlim([0, Lx])
    axes[1, 1].set_ylim([0, Ly])
    
    plt.tight_layout()
    plt.show()

print("\nGerando plotagem simples...")
plot_simple_comparison(K85_mD, P85_2D, vx85, vy85)
