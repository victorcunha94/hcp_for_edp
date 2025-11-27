import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import zoom, rotate
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# =============================================================================
# PARÂMETROS CORRETOS SPE10 
# =============================================================================
Lx, Nx = 670.56, 220   # 220 células na direção X (largura)
Ly, Ny = 365.76, 60    # 60 células na direção Y (altura)
nx, ny = Nx, Ny        
mu = 1.0   

# Dimensões originais do SPE10 
Nx_spe, Ny_spe, Nz_spe = 60, 220, 85

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
        
        print(f"Permeabilidades carregadas - Kx: {kx.shape}, Ky: {ky.shape}")
        
        # TRANSPOSE para orientação correta: (60, 220) -> (220, 60)
        kx = np.transpose(kx, (1, 0, 2))  # Agora (220, 60, 85)
        ky = np.transpose(ky, (1, 0, 2))
        kz = np.transpose(kz, (1, 0, 2))
        
        print(f"Após transpose - Kx: {kx.shape}, Ky: {ky.shape}")
        
        return kx, ky, kz
        
    except FileNotFoundError:
        print(f"ERRO: Arquivo {arquivo} não encontrado!")
        return None, None, None

def rotacionar_campo_180(K_field):
    """
    Rotaciona o campo de permeabilidade em 180 graus
    Isso corrige o problema do campo estar espelhado
    """
    return np.rot90(K_field, 2)  # 2 rotações de 90° = 180°

def combinar_perm_horizontal(kx, ky):
    """
    Combina componentes Kx e Ky para permeabilidade horizontal
    Pode usar média geométrica, aritmética, ou selecionar um componente
    """
    # Método 1: Usar apenas Kx (para fluxo predominante em X)
    # return kx
    
    # Método 2: Usar apenas Ky (para fluxo predominante em Y) 
    # return ky
    
    # Método 3: Média geométrica (recomendado para meios anisotrópicos)
    return np.sqrt(kx * ky)
    
    # Método 4: Média aritmética
    # return (kx + ky) / 2

def CreateMesh(Lx, Ly, Nx, Ny):
    """Cria malha com orientação correta"""
    dx = Lx / Nx
    dy = Ly / Ny
    xc = np.linspace(dx/2, Lx - dx/2, Nx)  # Centros das células em X
    yc = np.linspace(dy/2, Ly - dy/2, Ny)  # Centros das células em Y
    return xc, yc, dx, dy

def BuildDarcySystem(xc, yc, dx, dy, K_field, p_inj=1e-5, p_prod=1e-5):
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
            
            # Poços injetor e produtor
            if i == 0 and j == 0:
                rhs[g] += p_inj
            elif i == Nx - 1 and j == Ny - 1:
                rhs[g] -= p_prod
            
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
print("SPE10 - COM ROTAÇÃO E COMPONENTE CORRETO")
print("=" * 60)

# Carregar dados
kx_mD, ky_mD, kz_mD = carregar_dados_spe10_corrigido('spe_perm.dat')

if kx_mD is None:
    print("Não foi possível carregar os dados.")
    exit()

# Converter para m² (1 mD = 9.869233e-16 m²) 
kx = kx_mD * 9.869233e-16
ky = ky_mD * 9.869233e-16

# Selecionar camadas (já na orientação correta 220x60)
K33_kx_mD = kx_mD[:, :, 32]  # Camada 33 Kx
K33_ky_mD = ky_mD[:, :, 32]  # Camada 33 Ky

K35_kx_mD = kx_mD[:, :, 34]  # Camada 35 Kx  
K35_ky_mD = ky_mD[:, :, 34]  # Camada 35 Ky

K36_kx_mD = kx_mD[:, :, 35]  # Camada 36 Kx
K36_ky_mD = ky_mD[:, :, 35]  # Camada 36 Ky

print(f"\nCamada 33 - Kx: {K33_kx_mD.shape}, Ky: {K33_ky_mD.shape}")
print(f"Camada 35 - Kx: {K35_kx_mD.shape}, Ky: {K35_ky_mD.shape}")  
print(f"Camada 36 - Kx: {K36_kx_mD.shape}, Ky: {K36_ky_mD.shape}")

# ROTACIONAR OS CAMPOS EM 180 GRAUS
print("\nRotacionando campos em 180 graus...")
K33_kx_mD_rot = rotacionar_campo_180(K33_kx_mD)
K33_ky_mD_rot = rotacionar_campo_180(K33_ky_mD)
K33_comb_mD = combinar_perm_horizontal(K33_kx_mD_rot, K33_ky_mD_rot)

K35_kx_mD_rot = rotacionar_campo_180(K35_kx_mD)  
K35_ky_mD_rot = rotacionar_campo_180(K35_ky_mD)
K35_comb_mD = combinar_perm_horizontal(K35_kx_mD_rot, K35_ky_mD_rot)

K36_kx_mD_rot = rotacionar_campo_180(K36_kx_mD)
K36_ky_mD_rot = rotacionar_campo_180(K36_ky_mD) 
K36_comb_mD = combinar_perm_horizontal(K36_kx_mD_rot, K36_ky_mD_rot)

# Converter campos rotacionados para m²
K33_comb = K33_comb_mD * 9.869233e-16
K35_comb = K35_comb_mD * 9.869233e-16  
K36_comb = K36_comb_mD * 9.869233e-16

print(f"\nCampos rotacionados e combinados:")
print(f"Camada 33: {K33_comb_mD.shape}")
print(f"Camada 35: {K35_comb_mD.shape}")
print(f"Camada 36: {K36_comb_mD.shape}")

# Criar malha 
xc, yc, dx, dy = CreateMesh(Lx, Ly, Nx, Ny)

# ============== Resolver para Camada 36 (com campo rotacionado)
print("\n" + "=" * 40)
print("RESOLVENDO CAMADA 36 (ROTACIONADA)")
print("=" * 40)

A36, rhs36 = BuildDarcySystem(xc, yc, dx, dy, K36_comb, p_inj=0.036, p_prod=6e6)
P36 = scipy.sparse.linalg.spsolve(A36, rhs36)
P36_2D = P36.reshape((Nx, Ny))

# Calcular velocidade
gradP_x36 = np.gradient(P36_2D, dx, axis=0)
gradP_y36 = np.gradient(P36_2D, dy, axis=1)
vx36 = -K36_comb * gradP_x36 / mu
vy36 = -K36_comb * gradP_y36 / mu

print(f"Solução Camada 36: {P36_2D.shape}")
print(f"Pressão: {np.min(P36_2D):.2e} a {np.max(P36_2D):.2e} Pa")

# Salvar resultados
salvar_para_paraview_corrigido(xc, yc, P36_2D, K36_comb_mD, vx36, vy36, "camada_36_rotacionada")

# Plotar comparação antes/depois da rotação
def plot_comparacao_rotacao(xc, yc, K_original, K_rotacionado, titulo):
    """Plota comparação antes e depois da rotação"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = ax1.contourf(xc, yc, K_original.T, 50, 
                      norm=colors.LogNorm(vmin=np.min(K_original[K_original>0]), 
                                         vmax=np.max(K_original)),
                      cmap='jet')
    ax1.set_title(f'{titulo} - Original')
    ax1.set_xlabel('x [ft]')
    ax1.set_ylabel('y [ft]')
    plt.colorbar(im1, ax=ax1, label='Permeabilidade [mD]')
    
    im2 = ax2.contourf(xc, yc, K_rotacionado.T, 50,
                      norm=colors.LogNorm(vmin=np.min(K_rotacionado[K_rotacionado>0]),
                                         vmax=np.max(K_rotacionado)), 
                      cmap='jet')
    ax2.set_title(f'{titulo} - Rotacionado 180°')
    ax2.set_xlabel('x [ft]')
    ax2.set_ylabel('y [ft]')
    plt.colorbar(im2, ax=ax2, label='Permeabilidade [mD]')
    
    plt.tight_layout()
    plt.savefig(f'comparacao_rotacao_{titulo.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plotar comparações
plot_comparacao_rotacao(xc, yc, K36_kx_mD, K36_kx_mD_rot, "Camada 36 Kx")
plot_comparacao_rotacao(xc, yc, K36_ky_mD, K36_ky_mD_rot, "Camada 36 Ky")

# Plotar campo combinado
plot_permeabilidade(xc, yc, K36_comb_mD, "Camada 36 Combinada e Rotacionada")

# Resumo final
print("\n" + "=" * 60)
print("RESUMO FINAL - COM ROTAÇÃO")
print("=" * 60)
print("✓ Campos rotacionados em 180 graus")
print("✓ Componentes Kx e Ky carregados separadamente") 
print("✓ Permeabilidade combinada usando média geométrica")
print(f"✓ Arquivo gerado: camada_36_rotacionada.vts")
print("=" * 60)
