#!/usr/bin/env python3
"""
Analisador de resultados do Jacobi MPI
Gera gráficos de speedup e tabelas de tempos
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_and_process_results():
    """Carrega todos os arquivos results_*.csv e processa os dados"""
    
    # Encontra todos os arquivos results_*.csv
    result_files = glob.glob("results_*.csv")
    
    if not result_files:
        print("Nenhum arquivo results_*.csv encontrado!")
        return None
    
    print(f"Arquivos encontrados: {result_files}")
    
    data_summary = []
    
    for file in result_files:
        # Extrai a configuração do nome do arquivo
        config_name = file.replace('results_', '').replace('.csv', '')
        
        try:
            df = pd.read_csv(file)
            
            # Filtra apenas as linhas de metadados (não as de comunicação)
            meta_df = df[df['iterations'].notna()].copy()
            
            if meta_df.empty:
                print(f"Aviso: {file} não contém dados de metadados")
                continue
            
            # Encontra o tempo sequencial (assume que o menor N é o sequencial)
            # Ou você pode ter um arquivo específico para o sequencial
            seq_time = meta_df['exec_time'].min()  # Aproximação
            
            # Calcula métricas agregadas
            total_processes = meta_df['rank'].nunique()
            avg_exec_time = meta_df['exec_time'].mean()
            max_exec_time = meta_df['exec_time'].max()
            avg_comm_time = meta_df['comm_time'].mean()
            avg_overhead = meta_df['overhead'].mean()
            total_iterations = meta_df['iterations'].iloc[0]  # Assume que todos têm as mesmas iterações
            
            # Calcula speedup (aproximado)
            speedup = seq_time / max_exec_time if max_exec_time > 0 else 0
            
            data_summary.append({
                'config': config_name,
                'processes': total_processes,
                'grid_config': config_name,
                'avg_exec_time': avg_exec_time,
                'max_exec_time': max_exec_time,
                'avg_comm_time': avg_comm_time,
                'avg_overhead': avg_overhead,
                'speedup': speedup,
                'efficiency': speedup / total_processes if total_processes > 0 else 0,
                'iterations': total_iterations,
                'seq_time': seq_time
            })
            
            print(f"Processado {file}: {total_processes} processos, tempo médio: {avg_exec_time:.4f}s")
            
        except Exception as e:
            print(f"Erro ao processar {file}: {e}")
    
    return pd.DataFrame(data_summary)

def plot_speedup(df):
    """Gera gráfico de speedup"""
    if df is None or df.empty:
        print("Nenhum dado para plotar!")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Ordena por número de processos
    df_sorted = df.sort_values('processes')
    
    # Gráfico de speedup
    plt.subplot(2, 2, 1)
    plt.plot(df_sorted['processes'], df_sorted['speedup'], 'bo-', linewidth=2, markersize=8, label='Speedup Medido')
    
    # Linha do speedup ideal
    ideal_speedup = df_sorted['processes']
    plt.plot(df_sorted['processes'], ideal_speedup, 'r--', linewidth=2, label='Speedup Ideal')
    
    plt.xlabel('Número de Processos')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Número de Processos')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gráfico de eficiência
    plt.subplot(2, 2, 2)
    plt.plot(df_sorted['processes'], df_sorted['efficiency'] * 100, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Número de Processos')
    plt.ylabel('Eficiência (%)')
    plt.title('Eficiência vs Número de Processos')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de tempos de execução
    plt.subplot(2, 2, 3)
    x_pos = np.arange(len(df_sorted))
    width = 0.35
    
    plt.bar(x_pos - width/2, df_sorted['avg_exec_time'], width, label='Tempo Execução', alpha=0.7)
    plt.bar(x_pos + width/2, df_sorted['avg_comm_time'], width, label='Tempo Comunicação', alpha=0.7)
    
    plt.xlabel('Configuração')
    plt.ylabel('Tempo (s)')
    plt.title('Tempos de Execução e Comunicação')
    plt.xticks(x_pos, df_sorted['config'], rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de overhead
    plt.subplot(2, 2, 4)
    plt.bar(df_sorted['config'], df_sorted['avg_overhead'] * 100, alpha=0.7)
    plt.xlabel('Configuração')
    plt.ylabel('Overhead de Comunicação (%)')
    plt.title('Overhead vs Configuração')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('speedup_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df):
    """Cria tabela resumo dos resultados"""
    if df is None or df.empty:
        print("Nenhum dado para gerar tabela!")
        return
    
    # Ordena por número de processos
    df_sorted = df.sort_values('processes')
    
    # Seleciona e formata as colunas para a tabela
    summary_table = df_sorted[[
        'config', 'processes', 'avg_exec_time', 'max_exec_time', 
        'avg_comm_time', 'avg_overhead', 'speedup', 'efficiency'
    ]].copy()
    
    # Formata os valores
    summary_table['avg_exec_time'] = summary_table['avg_exec_time'].map('{:.4f}s'.format)
    summary_table['max_exec_time'] = summary_table['max_exec_time'].map('{:.4f}s'.format)
    summary_table['avg_comm_time'] = summary_table['avg_comm_time'].map('{:.4f}s'.format)
    summary_table['avg_overhead'] = summary_table['avg_overhead'].map('{:.2%}'.format)
    summary_table['speedup'] = summary_table['speedup'].map('{:.2f}'.format)
    summary_table['efficiency'] = summary_table['efficiency'].map('{:.2%}'.format)
    
    # Renomeia as colunas para português
    summary_table.columns = [
        'Configuração', 'Processos', 'Tempo Exec Médio', 'Tempo Exec Máximo',
        'Tempo Comm Médio', 'Overhead Médio', 'Speedup', 'Eficiência'
    ]
    
    print("\n" + "="*80)
    print("TABELA RESUMO DOS RESULTADOS")
    print("="*80)
    print(summary_table.to_string(index=False))
    print("="*80)
    
    # Salva a tabela em CSV
    summary_table.to_csv('summary_table.csv', index=False)
    print("\nTabela salva em 'summary_table.csv'")

def plot_communication_analysis(df):
    """Análise detalhada da comunicação"""
    if df is None or df.empty:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Gráfico de proporção comunicação/execução
    plt.subplot(1, 2, 1)
    comm_ratio = (df['avg_comm_time'] / df['avg_exec_time']) * 100
    plt.bar(df['config'], comm_ratio, alpha=0.7, color='orange')
    plt.xlabel('Configuração')
    plt.ylabel('Razão Comunicação/Execução (%)')
    plt.title('Proporção Tempo de Comunicação')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Gráfico de speedup por configuração de grid
    plt.subplot(1, 2, 2)
    
    # Extrai dimensões do grid para análise
    df['nx'] = df['config'].str.split('x').str[0].astype(int)
    df['ny'] = df['config'].str.split('x').str[1].astype(int)
    df['aspect_ratio'] = df['nx'] / df['ny']
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    plt.scatter(df['aspect_ratio'], df['speedup'], s=100, c=colors, alpha=0.7)
    
    for i, row in df.iterrows():
        plt.annotate(row['config'], (row['aspect_ratio'], row['speedup']), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel('Razão de Aspecto (nx/ny)')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Razão de Aspecto do Grid')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('communication_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Função principal"""
    print("Analisador de Resultados do Jacobi MPI")
    print("Carregando arquivos results_*.csv...")
    
    # Carrega e processa os dados
    df = load_and_process_results()
    
    if df is None or df.empty:
        print("Nenhum dado válido encontrado!")
        return
    
    print(f"\nProcessados {len(df)} configurações diferentes")
    
    # Gera os gráficos
    print("\nGerando gráficos...")
    plot_speedup(df)
    plot_communication_analysis(df)
    
    # Gera a tabela resumo
    print("\nGerando tabela resumo...")
    create_summary_table(df)
    
    # Estatísticas adicionais
    print("\n" + "="*50)
    print("ESTATÍSTICAS ADICIONAIS")
    print("="*50)
    print(f"Melhor speedup: {df['speedup'].max():.2f} ({df.loc[df['speedup'].idxmax(), 'config']})")
    print(f"Melhor eficiência: {df['efficiency'].max():.2%} ({df.loc[df['efficiency'].idxmax(), 'config']})")
    print(f"Menor overhead: {df['avg_overhead'].min():.2%} ({df.loc[df['avg_overhead'].idxmin(), 'config']})")
    
    # Salva dados processados
    df.to_csv('processed_results.csv', index=False)
    print(f"\nDados processados salvos em 'processed_results.csv'")

if __name__ == "__main__":
    main()
