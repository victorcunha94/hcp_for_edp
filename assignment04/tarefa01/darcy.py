import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom



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