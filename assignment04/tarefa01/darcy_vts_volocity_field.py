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

def plotar_campo_velocidade(xc, yc, P_2D, K_mD, vx, vy, titulo="Campo de Velocidade"):
    """
    Plota o campo de velocidade com vetores sobrepostos à permeabilidade
    """
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    # CORREÇÃO: Criar meshgrid corretamente
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    
    # Verificar se as coordenadas são estritamente crescentes
    print(f"Verificando coordenadas: X min={np.min(X):.2f}, max={np.max(X):.2f}, crescente={np.all(np.diff(X, axis=0) > 0)}")
    print(f"Verificando coordenadas: Y min={np.min(Y):.2f}, max={np.max(Y):.2f}, crescente={np.all(np.diff(Y, axis=1) > 0)}")
    
    # 1. Permeabilidade com vetores de velocidade
    im1 = axes[0, 0].imshow(K_mD.T, extent=[0, Lx, 0, Ly], origin='lower', 
                           cmap='viridis', aspect='auto')
    # Vetores de velocidade (amostrados)
    skip_x, skip_y = 8, 2  # Amostrar para não ficar muito poluído
    quiv = axes[0, 0].quiver(X[::skip_x, ::skip_y], Y[::skip_x, ::skip_y], 
                            vx[::skip_x, ::skip_y], vy[::skip_x, ::skip_y], 
                            color='red', scale=0.5, scale_units='inches')
    axes[0, 0].set_title('Permeabilidade + Vetores de Velocidade')
    axes[0, 0].set_xlabel('X [ft]')
    axes[0, 0].set_ylabel('Y [ft]')
    plt.colorbar(im1, ax=axes[0, 0], label='Permeabilidade [mD]')
    
    # 2. Magnitude da velocidade
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    im2 = axes[0, 1].imshow(velocity_magnitude.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='hot', aspect='auto')
    axes[0, 1].set_title('Magnitude da Velocidade')
    axes[0, 1].set_xlabel('X [ft]')
    axes[0, 1].set_ylabel('Y [ft]')
    plt.colorbar(im2, ax=axes[0, 1], label='Velocidade [m/s]')
    
    # 3. Linhas de corrente - CORREÇÃO: usar pcolormesh em vez de imshow
    axes[0, 2].pcolormesh(X, Y, K_mD, cmap='viridis', alpha=0.7)
    
    # CORREÇÃO: Criar grid regular para streamplot
    x_stream = np.linspace(0, Lx, Nx)
    y_stream = np.linspace(0, Ly, Ny)
    X_stream, Y_stream = np.meshgrid(x_stream, y_stream, indexing='ij')
    
    # Interpolar velocidades para o grid regular
    from scipy.interpolate import RegularGridInterpolator
    
    # Criar interpoladores
    points_x = (xc, yc)
    interpolator_vx = RegularGridInterpolator(points_x, vx, bounds_error=False, fill_value=0)
    interpolator_vy = RegularGridInterpolator(points_x, vy, bounds_error=False, fill_value=0)
    
    # Criar pontos de avaliação
    eval_points = np.array([[x, y] for x in x_stream for y in y_stream]).reshape(Nx, Ny, 2)
    
    # Interpolar velocidades
    vx_interp = interpolator_vx(eval_points)
    vy_interp = interpolator_vy(eval_points)
    
    # Normalizar para linhas de corrente
    v_mag_interp = np.sqrt(vx_interp**2 + vy_interp**2)
    vx_norm = vx_interp / (v_mag_interp + 1e-15)
    vy_norm = vy_interp / (v_mag_interp + 1e-15)
    
    # Plotar linhas de corrente
    try:
        strm = axes[0, 2].streamplot(X_stream.T, Y_stream.T, vx_norm.T, vy_norm.T, 
                                    color='red', linewidth=1.5, density=2)
        axes[0, 2].set_title('Linhas de Corrente sobre Permeabilidade')
    except Exception as e:
        print(f"Aviso: Não foi possível plotar linhas de corrente: {e}")
        axes[0, 2].set_title('Linhas de Corrente (não disponível)')
    
    axes[0, 2].set_xlabel('X [ft]')
    axes[0, 2].set_ylabel('Y [ft]')
    plt.colorbar(im1, ax=axes[0, 2], label='Permeabilidade [mD]')
    
    # 4. Pressão
    im4 = axes[1, 0].imshow(P_2D.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='jet', aspect='auto')
    axes[1, 0].set_title('Campo de Pressão')
    axes[1, 0].set_xlabel('X [ft]')
    axes[1, 0].set_ylabel('Y [ft]')
    plt.colorbar(im4, ax=axes[1, 0], label='Pressão [Pa]')
    
    # 5. Componente X da velocidade
    im5 = axes[1, 1].imshow(vx.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('Velocidade X')
    axes[1, 1].set_xlabel('X [ft]')
    axes[1, 1].set_ylabel('Y [ft]')
    plt.colorbar(im5, ax=axes[1, 1], label='Velocidade X [m/s]')
    
    # 6. Componente Y da velocidade
    im6 = axes[1, 2].imshow(vy.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='coolwarm', aspect='auto')
    axes[1, 2].set_title('Velocidade Y')
    axes[1, 2].set_xlabel('X [ft]')
    axes[1, 2].set_ylabel('Y [ft]')
    plt.colorbar(im6, ax=axes[1, 2], label='Velocidade Y [m/s]')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plotar_analise_caminhos_fluxo(xc, yc, K_mD, vx, vy, titulo="Análise de Caminhos de Fluxo"):
    """
    Análise detalhada dos caminhos preferenciais de fluxo
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(titulo, fontsize=16, fontweight='bold')
    
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    
    # 1. Caminhos preferenciais - velocidade normalizada pela permeabilidade
    flux_potential = velocity_magnitude / (K_mD + 1e-15)
    im1 = axes[0, 0].imshow(flux_potential.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='YlOrRd', aspect='auto')
    axes[0, 0].set_title('Potencial de Fluxo (|v|/K)')
    axes[0, 0].set_xlabel('X [ft]')
    axes[0, 0].set_ylabel('Y [ft]')
    plt.colorbar(im1, ax=axes[0, 0], label='|v|/K [1/s]')
    
    # 2. Divergência do campo de velocidade (fontes/sumidouros)
    div_v = np.gradient(vx, axis=0) / (Lx/Nx) + np.gradient(vy, axis=1) / (Ly/Ny)
    im2 = axes[0, 1].imshow(div_v.T, extent=[0, Lx, 0, Ly], origin='lower',
                           cmap='RdBu_r', aspect='auto')
    axes[0, 1].set_title('Divergência do Campo de Velocidade')
    axes[0, 1].set_xlabel('X [ft]')
    axes[0, 1].set_ylabel('Y [ft]')
    plt.colorbar(im2, ax=axes[0, 1], label='∇·v [1/s]')
    
    # 3. Vetores coloridos pela magnitude
    skip = 6
    magnitude_at_points = velocity_magnitude[::skip, ::skip]
    quiv = axes[1, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                            vx[::skip, ::skip], vy[::skip, ::skip], 
                            magnitude_at_points, cmap='hot', scale=0.8)
    axes[1, 0].set_title('Vetores Coloridos pela Magnitude')
    axes[1, 0].set_xlabel('X [ft]')
    axes[1, 0].set_ylabel('Y [ft]')
    plt.colorbar(quiv, ax=axes[1, 0], label='Velocidade [m/s]')
    
    # 4. Histograma de direções do fluxo
    angles = np.arctan2(vy.flatten(), vx.flatten()) * 180 / np.pi
    magnitudes = velocity_magnitude.flatten()
    
    # Filtrar magnitudes significativas
    mask = magnitudes > np.percentile(magnitudes, 10)
    angles_filtered = angles[mask]
    
    axes[1, 1].hist(angles_filtered, bins=36, range=(-180, 180), 
                   color='skyblue', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribuição de Direções do Fluxo')
    axes[1, 1].set_xlabel('Ângulo [graus]')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Adicionar estatísticas
    mean_angle = np.mean(angles_filtered)
    axes[1, 1].axvline(mean_angle, color='red', linestyle='--', 
                      label=f'Ângulo médio: {mean_angle:.1f}°')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

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

# Criar malha com orientação correta
xc, yc, dx, dy = CreateMesh(Lx, Ly, Nx, Ny)
print(f"\nMalha criada: {len(xc)} x {len(yc)} células")

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

# =============================================================================
# PLOTAGEM DOS CAMPOS DE VELOCIDADE
# =============================================================================

print("\n" + "=" * 40)
print("GERANDO VISUALIZAÇÕES DOS CAMPOS DE VELOCIDADE")
print("=" * 40)

# Plotar Camada 33
print("Plotando Camada 33...")
plotar_campo_velocidade(xc, yc, P33_2D, K33_mD, vx33, vy33, 
                       "SPE10 - Camada 33 - Campo de Velocidade")

print("Plotando análise de caminhos de fluxo - Camada 33...")
plotar_analise_caminhos_fluxo(xc, yc, K33_mD, vx33, vy33,
                            "SPE10 - Camada 33 - Análise de Caminhos de Fluxo")

# Plotar Camada 35
print("Plotando Camada 35...")
plotar_campo_velocidade(xc, yc, P35_2D, K35_mD, vx35, vy35, 
                       "SPE10 - Camada 35 - Campo de Velocidade")

print("Plotando análise de caminhos de fluxo - Camada 35...")
plotar_analise_caminhos_fluxo(xc, yc, K35_mD, vx35, vy35,
                            "SPE10 - Camada 35 - Análise de Caminhos de Fluxo")

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
print(f"Velocidade máxima Camada 33: {np.max(np.sqrt(vx33**2 + vy33**2)):.2e} m/s")
print(f"Velocidade máxima Camada 35: {np.max(np.sqrt(vx35**2 + vy35**2)):.2e} m/s")
print("Arquivos gerados: camada_33_horizontal.vts, camada_35_horizontal.vts")
print("=" * 60)
