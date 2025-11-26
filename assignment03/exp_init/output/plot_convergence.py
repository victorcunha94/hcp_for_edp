#!/usr/bin/env python3
"""
Script para gerar gráfico de taxa de convergência a partir de arquivos CSV
Uso: python3 plot_convergence.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from mpl_toolkits.mplot3d import Axes3D

def load_solution_with_communication_data(csv_file, N):
    """
    Carrega a solução numérica a partir do arquivo CSV e reconstrói a malha global
    """
    try:
        df = pd.read_csv(csv_file)
        
        # Verificar se as colunas necessárias existem
        required_columns = ['rank', 'start_x', 'end_x', 'start_y', 'end_y', 'local_nx', 'local_ny', 'U_data']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"AVISO: Colunas faltando em {csv_file}: {missing_columns}")
            return None, df
        
        # Filtrar processos válidos
        valid_data = df[df['start_x'].notna() & df['start_y'].notna() & df['U_data'].notna()]
        
        if valid_data.empty:
            print(f"AVISO: Nenhum dado válido em {csv_file}")
            return None, df
        
        # Reconstruir malha global
        global_solution = np.zeros((N, N))
        
        for _, row in valid_data.iterrows():
            start_x, end_x = int(row['start_x']), int(row['end_x'])
            start_y, end_y = int(row['start_y']), int(row['end_y'])
            local_nx, local_ny = int(row['local_nx']), int(row['local_ny'])
            
            # Converter U_data de string para array numpy
            if isinstance(row['U_data'], str):
                u_data = np.fromstring(row['U_data'].strip('[]'), sep=',')
            else:
                u_data = np.array(row['U_data'])
            
            # Remodelar para dimensões locais
            u_local = u_data.reshape((local_nx, local_ny))
            
            # Preencher na solução global
            global_solution[start_x:end_x, start_y:end_y] = u_local
        
        return global_solution, df
        
    except Exception as e:
        print(f"Erro ao carregar {csv_file}: {e}")
        return None, pd.DataFrame()

def plot_convergence_study(csv_files, N_values, output_file=None):
    """
    Plot 3: Estudo de convergência para diferentes tamanhos de malha
    Calcula a norma do infinito do módulo da diferença entre solução numérica e analítica
    """
    errors_L2 = []
    errors_Linf = []
    successful_N = []
    
    for csv_file, N in zip(csv_files, N_values):
        try:
            print(f"\nProcessando: {csv_file} com N={N}")
            
            # Carrega solução numérica
            global_solution, meta_df = load_solution_with_communication_data(csv_file, N)
            
            if global_solution is None:
                print(f"  ⚠️  Não foi possível carregar solução de {csv_file}")
                continue
            
            # Calcula solução analítica
            x = np.linspace(0, 1, N)
            y = np.linspace(0, 1, N)
            X, Y = np.meshgrid(x, y, indexing='ij')
            analytical = (1/(8*np.pi**2)) * np.sin(2*np.pi * X) * np.sin(2*np.pi * Y)
            
            # Aplicar condições de contorno de Dirichlet nulas
            analytical[0, :] = 0
            analytical[-1, :] = 0
            analytical[:, 0] = 0
            analytical[:, -1] = 0
            
            # Calcula o módulo da diferença ponto a ponto
            diff_module = np.abs(global_solution - analytical)
            
            # Calcula normas de erro (apenas pontos internos para evitar bordas)
            interior_slice = slice(1, -1)
            error_L2 = np.sqrt(np.mean(diff_module[interior_slice, interior_slice]**2))
            error_Linf = np.max(diff_module[interior_slice, interior_slice])
            
            errors_L2.append(error_L2)
            errors_Linf.append(error_Linf)
            successful_N.append(N)
            
            print(f"  ✅ N={N}:")
            print(f"     Norma L2 = {error_L2:.2e}")
            print(f"     Norma L∞ = {error_Linf:.2e}")
            
        except Exception as e:
            print(f"  ❌ Erro ao processar {csv_file}: {e}")
            continue
    
    if len(errors_L2) < 2:
        print("\n⚠️  Não há dados suficientes para estudo de convergência")
        print("   É necessário pelo menos 2 arquivos CSV com soluções válidas")
        return None
    
    # Preparar dados para plotagem
    h_values = [1.0/(N-1) for N in successful_N]
    
    # Calcula taxas de convergência
    rates_L2 = []
    rates_Linf = []
    
    for i in range(1, len(errors_L2)):
        rate_L2 = np.log(errors_L2[i-1]/errors_L2[i]) / np.log(h_values[i-1]/h_values[i])
        rate_Linf = np.log(errors_Linf[i-1]/errors_Linf[i]) / np.log(h_values[i-1]/h_values[i])
        rates_L2.append(rate_L2)
        rates_Linf.append(rate_Linf)
    
    # PRIMEIRA FIGURA: Escala log-log
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    
    ax1.loglog(h_values, errors_L2, 'bo-', linewidth=3, markersize=10, label='Erro L²', markerfacecolor='blue')
    ax1.loglog(h_values, errors_Linf, 'ro-', linewidth=3, markersize=10, label='Erro L∞', markerfacecolor='red')
    
    # Adiciona linhas de referência para taxas teóricas
    h_ref = np.array(h_values)
    ax1.loglog(h_ref, 0.1 * h_ref**2, 'k--', alpha=0.7, linewidth=2, label='O(h²)')
    ax1.loglog(h_ref, 0.1 * h_ref, 'k:', alpha=0.7, linewidth=2, label='O(h)')
    
    ax1.set_xlabel('Tamanho da malha (h)', fontsize=12)
    ax1.set_ylabel('Erro', fontsize=12)
    ax1.set_title('Taxa de Convergência (escala log-log)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.set_axisbelow(True)
    
    # Adiciona tabela com valores numéricos
    table_data = []
    for i, N in enumerate(successful_N):
        table_data.append([f"N={N}", f"{errors_L2[i]:.2e}", f"{errors_Linf[i]:.2e}"])
    
    table = ax1.table(cellText=table_data,
                     colLabels=['Malha', 'Erro L²', 'Erro L∞'],
                     loc='lower left',
                     bbox=[0.02, 0.02, 0.4, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    
    plt.tight_layout()
    
    # Salva a primeira figura
    if output_file:
        base_name = output_file.replace('.png', '')
        output_file1 = f"{base_name}_loglog.png"
        plt.savefig(output_file1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Plot log-log salvo como: {output_file1}")
    
    plt.show()
    plt.close(fig1)  # Fecha a figura para liberar memória
    
    # SEGUNDA FIGURA: Escala semilogy
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    ax2.semilogy(successful_N, errors_L2, 'bo-', linewidth=3, markersize=10, label='Erro L²', markerfacecolor='blue')
    ax2.semilogy(successful_N, errors_Linf, 'ro-', linewidth=3, markersize=10, label='Erro L∞', markerfacecolor='red')
    
    ax2.set_xlabel('Tamanho da malha (N)', fontsize=12)
    ax2.set_ylabel('Erro', fontsize=12)
    ax2.set_title('Convergência vs Tamanho da Malha', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_axisbelow(True)
    
    # Adiciona texto com taxas
    conv_text = f"Taxas de convergência estimadas:\n\n"
    if rates_L2:
        conv_text += f"Erro L²: {np.mean(rates_L2):.3f} (esperado: ~2.0)\n"
    if rates_Linf:
        conv_text += f"Erro L∞: {np.mean(rates_Linf):.3f} (esperado: ~2.0)"
    
    ax2.text(0.05, 0.95, conv_text, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Salva a segunda figura
    if output_file:
        output_file2 = f"{base_name}_semilogy.png"
        plt.savefig(output_file2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Plot semilogy salvo como: {output_file2}")
    
    plt.show()
    plt.close(fig2)  # Fecha a figura para liberar memória
    
    return {
        'N_values': successful_N,
        'h_values': h_values,
        'errors_L2': errors_L2,
        'errors_Linf': errors_Linf,
        'convergence_rates_L2': rates_L2,
        'convergence_rates_Linf': rates_Linf
    }

def main():
    """
    Função principal que busca automaticamente os arquivos CSV na pasta atual
    """
    print("=" * 60)
    print("ANÁLISE DE CONVERGÊNCIA - MÉTODO DE JACOBI 2D")
    print("=" * 60)
    
    # Buscar arquivos CSV no padrão results_1x1_N*.csv
    csv_patterns = [
        "results_1x1_N*.csv",
        "results_2x2_N*.csv", 
        "results_3x3_N*.csv",
        "results_4x4_N*.csv",
        "results_5x5_N*.csv",
        "results_6x6_N*.csv"
    ]
    
    csv_files = []
    for pattern in csv_patterns:
        csv_files.extend(glob.glob(pattern))
    
    if not csv_files:
        print("❌ Nenhum arquivo CSV encontrado!")
        print("   Certifique-se de que os arquivos estão no padrão:")
        print("   results_1x1_N256.csv, results_2x2_N512.csv, etc.")
        return
    
    print(f"📁 Arquivos CSV encontrados ({len(csv_files)}):")
    for csv_file in sorted(csv_files):
        print(f"   • {csv_file}")
    
    # Extrair valores de N dos nomes dos arquivos
    N_values = []
    valid_csv_files = []
    
    for csv_file in sorted(csv_files):
        try:
            # Extrair N do nome do arquivo (assumindo padrão results_*_N{numero}.csv)
            if '_N' in csv_file:
                N_str = csv_file.split('_N')[-1].replace('.csv', '')
                N = int(N_str)
                N_values.append(N)
                valid_csv_files.append(csv_file)
                print(f"   ✅ {csv_file} → N={N}")
            else:
                print(f"   ⚠️  {csv_file} (padrão de nome não reconhecido)")
        except ValueError:
            print(f"   ⚠️  {csv_file} (não foi possível extrair N)")
    
    if not valid_csv_files:
        print("❌ Nenhum arquivo CSV válido encontrado!")
        return
    
    print(f"\n🎯 Executando análise de convergência com {len(valid_csv_files)} arquivos...")
    
    # Gerar gráfico de convergência
    results = plot_convergence_study(
        csv_files=valid_csv_files,
        N_values=N_values,
        output_file="convergence_study.png"
    )
    
    if results:
        print(f"\n📊 Resultados da análise:")
        print(f"   Malhas analisadas: {results['N_values']}")
        print(f"   Taxa L² média: {np.mean(results['convergence_rates_L2']):.3f}")
        print(f"   Taxa L∞ média: {np.mean(results['convergence_rates_Linf']):.3f}")
        print(f"   (Taxa esperada para Jacobi 2D: ~2.0)")

if __name__ == "__main__":
    main()
