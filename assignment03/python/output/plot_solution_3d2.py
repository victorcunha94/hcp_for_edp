#!/usr/bin/env python3
"""
Plot 3D da solução do Jacobi MPI a partir do arquivo CSV


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
import argparse



def load_solution_from_csv(csv_file, N):
    """
    Carrega e reconstrói a solução global a partir do arquivo CSV
    """
    print(f"Carregando arquivo: {csv_file}")
    
    # Carrega o CSV
    df = pd.read_csv(csv_file)
    
    # Encontra a linha que contém a solução final (última linha com dados do domínio)
    solution_rows = df[df['start_x'].notna() & df['start_y'].notna()]
    
    if solution_rows.empty:
        raise ValueError("Arquivo CSV não contém dados de solução")
    
    # Pega a última iteração (solução final)
    final_solution_row = solution_rows.iloc[-1]
    
    print(f"Processo {int(final_solution_row['rank'])}: domínio ({int(final_solution_row['start_x'])}:{int(final_solution_row['end_x'])}) x ({int(final_solution_row['start_y'])}:{int(final_solution_row['end_y'])})")
    
    # Cria matriz para solução global
    global_solution = np.zeros((N, N))
    
    # Para cada processo, reconstruímos sua parte da solução
    for _, row in solution_rows.iterrows():
        rank = int(row['rank'])
        start_x = int(row['start_x'])
        end_x = int(row['end_x'])
        start_y = int(row['start_y'])
        end_y = int(row['end_y'])
        local_nx = int(row['local_nx'])
        local_ny = int(row['local_ny'])
        
        # Aqui precisaríamos dos valores reais de U_local
        # Como não estão no CSV, vamos simular com valores baseados no erro
        # OU você precisa modificar seu código MPI para salvar U_local
        
        # Alternativa: usar valores baseados na posição e no erro final
        dx = 1.0 / (N - 1)
        final_error = float(row['final_error'])
        
        for i_local in range(local_nx):
            for j_local in range(local_ny):
                i_global = start_x + i_local
                j_global = start_y + j_local
                
                if i_global < N and j_global < N:
                    x = i_global * dx
                    y = j_global * dx
                    # Solução numérica aproximada (substitua pelos dados reais quando disponível)
                    global_solution[i_global, j_global] = np.sin(2*np.pi * x) * np.sin(2*np.pi * y) 
    
    return global_solution, solution_rows


def load_solution_with_communication_data(csv_file, N):
    """
<<<<<<< HEAD
    Carrega a solução numérica REAL a partir do CSV
    """
    print(f"Carregando arquivo: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Filtra dados de domínio (onde U_data está presente)
    domain_data = df[df['start_x'].notna() & df['start_y'].notna() & df['U_data'].notna()]
    
    if domain_data.empty:
        raise ValueError("Arquivo CSV não contém dados de solução numérica")
    
    global_solution = np.zeros((N, N))
    
    # Processa cada processo
    for _, row in domain_data.iterrows():
        rank = int(row['rank'])
        start_x = int(row['start_x'])
        end_x = int(row['end_x'])
        start_y = int(row['start_y'])
        end_y = int(row['end_y'])
        local_nx = int(row['local_nx'])
        local_ny = int(row['local_ny'])
        
        print(f"Processo {rank}: ({start_x}:{end_x}) x ({start_y}:{end_y}), tamanho: {local_nx}x{local_ny}")
        
        # Converte string de U_data de volta para array numpy
        if isinstance(row['U_data'], str):
            # Remove colchetes e converte para lista de floats
            u_data_str = row['U_data'].strip('[]')
            u_values = np.fromstring(u_data_str, sep=',')
        else:
            u_values = np.array(row['U_data'])
        
        # Redimensiona para a forma local correta
        try:
            U_local = u_values.reshape((local_nx, local_ny))
            
            # Coloca na solução global
            for i_local in range(local_nx):
                for j_local in range(local_ny):
                    i_global = start_x + i_local
                    j_global = start_y + j_local
                    
                    if i_global < N and j_global < N:
                        global_solution[i_global, j_global] = U_local[i_local, j_local]
                        
        except Exception as e:
            print(f"Erro ao processar dados do processo {rank}: {e}")
            print(f"Tamanho esperado: {local_nx}x{local_ny} = {local_nx * local_ny}")
            print(f"Tamanho recebido: {len(u_values)}")
            continue
    
    # Aplica condições de contorno (bordas = 0)
    global_solution[0, :] = 0.0
    global_solution[-1, :] = 0.0
    global_solution[:, 0] = 0.0
    global_solution[:, -1] = 0.0
    
    return global_solution, domain_data



=======
    Carrega a solução U diretamente do CSV, usando as matrizes locais salvas por cada rank.
    Espera que o CSV tenha colunas: rank, start_x, end_x, start_y, end_y, local_nx, local_ny, e uma
    coluna (ex: 'U' ou 'U_local' ou 'local_matrix') contendo a matriz local como string.
    """
    import ast

<<<<<<< HEAD

=======
    print(f"Carregando arquivo: {csv_file}")
    df = pd.read_csv(csv_file)

    # Filtra as linhas de domínio (cada processo)
    domain_data = df[df['start_x'].notna() & df['start_y'].notna()]
    if domain_data.empty:
        raise ValueError("Arquivo CSV não contém dados de domínio com solução local")

    global_solution = np.zeros((N, N))

    # Detecta o nome da coluna que contém a matriz local
    matrix_col = None
    for c in df.columns:
        if any(k in c.lower() for k in ["u", "matrix", "local"]):
            matrix_col = c
            break

    if matrix_col is None:
        raise ValueError("Nenhuma coluna encontrada contendo a matriz local (ex: 'U_local').")

    print(f"Usando coluna '{matrix_col}' para reconstruir a solução global.")

    for _, row in domain_data.iterrows():
        rank = int(row['rank'])
        start_x = int(row['start_x'])
        end_x = int(row['end_x'])
        start_y = int(row['start_y'])
        end_y = int(row['end_y'])
        local_nx = int(row['local_nx'])
        local_ny = int(row['local_ny'])

        matrix_str = str(row[matrix_col]).strip()
        try:
            # Tenta interpretar como lista literal (Python-style)
            U_local = np.array(ast.literal_eval(matrix_str), dtype=float)
        except Exception:
            # Fallback: tenta converter de string com separadores
            U_local = np.fromstring(matrix_str.replace('\n', ' '), sep=' ')
        # Garante forma correta
        U_local = U_local.reshape((local_nx, local_ny))

        # Copia para o domínio global
        global_solution[start_x:end_x, start_y:end_y] = U_local

        print(f"✅ Rank {rank}: domínio ({start_x}:{end_x}, {start_y}:{end_y}) carregado.")

    return global_solution, domain_data
>>>>>>> 664164ebf050fd4ebdb05dd7cd837323981b0541
>>>>>>> 0f9ae95cc766e899f1ccbcee1992ac32461fe82c
def plot_3d_comparison(global_solution, N, nx, ny, output_file=None):
    """
    Plot 1: Comparação 3D entre solução numérica e analítica
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    analytical = np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)
    
    fig = plt.figure(figsize=(15, 6))
    
    # Solução numérica
    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, global_solution, cmap='viridis', 
                            alpha=0.9, antialiased=True, rstride=1, cstride=1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('U(x,y)')
    ax1.set_title('Solução Numérica 3D')
    fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=20)
    
    # Solução analítica
    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, analytical, cmap='plasma', 
                            alpha=0.9, antialiased=True, rstride=1, cstride=1)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('U(x,y)')
    ax2.set_title('Solução Analítica 3D')
    fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=20)
    
    plt.suptitle(f'Comparação 3D - Malha {N}×{N}, Processos {nx}×{ny}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot 3D salvo como: {output_file}")
    
    plt.show()
    return fig

def plot_2d_comparison(global_solution, N, nx, ny, output_file=None):
    """
    Plot 2: Comparação 2D com contornos
    """
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    analytical = np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Heatmap numérico
    im1 = axes[0,0].imshow(global_solution.T, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='viridis', aspect='equal')
    axes[0,0].set_title('Solução Numérica - Heatmap')
    axes[0,0].set_xlabel('X')
    axes[0,0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Heatmap analítico
    im2 = axes[0,1].imshow(analytical.T, origin='lower', extent=[0, 1, 0, 1], 
                          cmap='plasma', aspect='equal')
    axes[0,1].set_title('Solução Analítica - Heatmap')
    axes[0,1].set_xlabel('X')
    axes[0,1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Contornos numéricos
    contour1 = axes[1,0].contour(X, Y, global_solution, 15, colors='black', alpha=0.7)
    axes[1,0].clabel(contour1, inline=True, fontsize=8)
    im3 = axes[1,0].contourf(X, Y, global_solution, 15, cmap='viridis', alpha=0.8)
    axes[1,0].set_title('Solução Numérica - Contornos')
    axes[1,0].set_xlabel('X')
    axes[1,0].set_ylabel('Y')
    plt.colorbar(im3, ax=axes[1,0])
    
    # Contornos analíticos
    contour2 = axes[1,1].contour(X, Y, analytical, 15, colors='black', alpha=0.7)
    axes[1,1].clabel(contour2, inline=True, fontsize=8)
    im4 = axes[1,1].contourf(X, Y, analytical, 15, cmap='plasma', alpha=0.8)
    axes[1,1].set_title('Solução Analítica - Contornos')
    axes[1,1].set_xlabel('X')
    axes[1,1].set_ylabel('Y')
    plt.colorbar(im4, ax=axes[1,1])
    
    plt.suptitle(f'Comparação 2D - Malha {N}×{N}, Processos {nx}×{ny}', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot 2D salvo como: {output_file}")
    
    plt.show()
    return fig

def plot_convergence_study(csv_files, N_values, output_file=None):
    """
    Plot 3: Estudo de convergência para diferentes tamanhos de malha
    """
    errors_L2 = []
    errors_Linf = []
    
    for csv_file, N in zip(csv_files, N_values):
        try:
            # Carrega solução numérica (ajuste conforme sua implementação)
            global_solution, _ = load_solution_with_communication_data(csv_file, N)
            
            # Calcula solução analítica
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            analytical = np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)
            
            # Calcula erros
            error = np.abs(global_solution - analytical)
            error_L2 = np.sqrt(np.mean(error**2))
            error_Linf = np.max(error)
            
            errors_L2.append(error_L2)
            errors_Linf.append(error_Linf)
            
            print(f"N={N}: L2={error_L2:.2e}, Linf={error_Linf:.2e}")
            
        except Exception as e:
            print(f"Erro ao processar {csv_file}: {e}")
            continue
    
    if len(errors_L2) < 2:
        print("Não há dados suficientes para estudo de convergência")
        return
    
    # Tamanhos da malha
    h_values = [1.0/(N-1) for N in N_values[:len(errors_L2)]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot log-log para taxa de convergência
    ax1.loglog(h_values, errors_L2, 'bo-', linewidth=2, markersize=8, label='Erro L²')
    ax1.loglog(h_values, errors_Linf, 'ro-', linewidth=2, markersize=8, label='Erro L∞')
    
    # Adiciona linhas de referência para taxas teóricas
    h_ref = np.array(h_values)
    ax1.loglog(h_ref, h_ref**2, 'k--', alpha=0.7, label='O(h²)')
    ax1.loglog(h_ref, h_ref, 'k:', alpha=0.7, label='O(h)')
    
    ax1.set_xlabel('Tamanho da malha (h)')
    ax1.set_ylabel('Erro')
    ax1.set_title('Taxa de Convergência (escala log-log)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    # Plot linear para visualização direta
    ax2.plot(N_values[:len(errors_L2)], errors_L2, 'bo-', linewidth=2, markersize=8, label='Erro L²')
    ax2.plot(N_values[:len(errors_L2)], errors_Linf, 'ro-', linewidth=2, markersize=8, label='Erro L∞')
    ax2.set_xlabel('Tamanho da malha (N)')
    ax2.set_ylabel('Erro')
    ax2.set_title('Convergência vs Tamanho da Malha')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Calcula taxas de convergência
    if len(errors_L2) >= 2:
        rates_L2 = []
        rates_Linf = []
        
        for i in range(1, len(errors_L2)):
            rate_L2 = np.log(errors_L2[i-1]/errors_L2[i]) / np.log(h_values[i-1]/h_values[i])
            rate_Linf = np.log(errors_Linf[i-1]/errors_Linf[i]) / np.log(h_values[i-1]/h_values[i])
            rates_L2.append(rate_L2)
            rates_Linf.append(rate_Linf)
        
        # Adiciona texto com taxas
        conv_text = f"Taxas de convergência estimadas:\n\n"
        conv_text += f"Erro L²: {np.mean(rates_L2):.3f} (esperado: ~2.0)\n"
        conv_text += f"Erro L∞: {np.mean(rates_Linf):.3f} (esperado: ~2.0)"
        
        ax2.text(0.05, 0.95, conv_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Estudo de Convergência - Solução Numérica vs Analítica', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot de convergência salvo como: {output_file}")
    
    plt.show()
    return fig


def plot_decomposition(meta_df, N, output_file=None):
    """
    Plota a decomposição de domínio
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cria matriz de cores para os processos
    process_map = -1 * np.ones((N, N))
    
    for _, row in meta_df.iterrows():
        rank = int(row['rank'])
        start_x = int(row['start_x'])
        end_x = int(row['end_x'])
        start_y = int(row['start_y'])
        end_y = int(row['end_y'])
        
        # Marca a área do processo
        for i in range(start_x, end_x):
            for j in range(start_y, end_y):
                if i < N and j < N:
                    process_map[i, j] = rank
    
    # Plot
    im = ax.imshow(process_map.T, origin='lower', cmap='tab10', 
                  extent=[0, 1, 0, 1], aspect='equal')
    
    # Adiciona anotações
    for _, row in meta_df.iterrows():
        rank = int(row['rank'])
        start_x = int(row['start_x'])
        end_x = int(row['end_x'])
        start_y = int(row['start_y'])
        end_y = int(row['end_y'])
        
        center_x = (start_x + end_x - 1) / (2 * (N - 1))
        center_y = (start_y + end_y - 1) / (2 * (N - 1))
        
        ax.text(center_x, center_y, f'P{rank}', ha='center', va='center',
               fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Decomposição de Domínio MPI')
    ax.grid(True, alpha=0.3)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Rank do Processo')
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Decomposição salva como: {output_file}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot 3D da solução do Jacobi MPI")
    parser.add_argument("--csv", type=str, help="Arquivo CSV específico (ex: results_2x2.csv)")
    parser.add_argument("--N", type=int, required=True, help="Tamanho da malha N×N")
    parser.add_argument("--auto-detect", action='store_true', help="Detectar automaticamente arquivos results_*.csv")
    parser.add_argument("--convergence-study", action='store_true', help="Fazer estudo de convergência com múltiplos arquivos")
    
    args = parser.parse_args()
    
    # Encontrar arquivo CSV
    if args.csv:
        csv_files = [args.csv]
    elif args.auto_detect:
        csv_files = glob.glob("results_*.csv")
        if not csv_files:
            print("Nenhum arquivo results_*.csv encontrado!")
            return
    else:
        print("Especifique --csv ou --auto-detect")
        return
    
    for csv_file in csv_files:
        print(f"\n{'='*60}")
        print(f"PROCESSANDO: {csv_file}")
        print(f"{'='*60}")
        
        try:
            # Extrai nx e ny do nome do arquivo
            filename = os.path.basename(csv_file)
            parts = filename.replace('results_', '').replace('.csv', '').split('x')
            if len(parts) == 2:
                nx, ny = int(parts[0]), int(parts[1])
            else:
                nx, ny = 2, 2  # Default
            
            # Carrega a solução
            global_solution, meta_df = load_solution_with_communication_data(csv_file, args.N)

            # Gera os três plots separados
            plot_3d_comparison(global_solution, args.N, nx, ny, f"comparison_3d_{nx}x{ny}_N{args.N}.png")
            plot_2d_comparison(global_solution, args.N, nx, ny, f"comparison_2d_{nx}x{ny}_N{args.N}.png")

            # Gera os três plots separados
            plot_decomposition(meta_df, args.N, output_file=None)
            
            # Plota solução 3D
            #output_3d = f"solucao_numerica_{nx}x{ny}_N{args.N}.png"
            #plot_3d_solution(global_solution, args.N, nx, ny, output_3d)
            
            # Mostra estatísticas do CSV
            print(f"\nESTATÍSTICAS DO CSV:")
            print(f"Arquivo: {csv_file}")
            print(f"Processos: {len(meta_df['rank'].unique())}")
            print(f"Configuração: {nx}×{ny}")
            print(f"Malha: {args.N}×{args.N}")
            
            # Estatísticas de performance
            if 'exec_time' in meta_df.columns:
                total_time = meta_df['exec_time'].sum()
                avg_time = meta_df['exec_time'].mean()
                total_comm = meta_df['comm_time'].sum() if 'comm_time' in meta_df.columns else 0
                print(f"Tempo total: {total_time:.4f}s")
                print(f"Tempo médio: {avg_time:.4f}s")
                if total_comm > 0:
                    print(f"Tempo comunicação: {total_comm:.4f}s")
                    print(f"Overhead: {(total_comm/total_time*100):.1f}%")
            
        except Exception as e:
            print(f"Erro ao processar {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
