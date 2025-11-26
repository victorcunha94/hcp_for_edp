#!/usr/bin/env python3
"""
Análise Comparativa: Jacobi Básico vs Jacobi com Relaxamento
- Norma do erro vs espaçamento
- Tempo computacional vs espaçamento
- Tabela comparativa
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict

################# MÉTODOS JACOBI ####################

# Função fonte
def f(x, y):
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

# Solução analítica para comparação
def exact_solution(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

######### MÉTODO 1: Jacobi Básico ##########
def jacobi_basic(N, dx, dy, max_iter=10000, tol=1e-8):
    """Método de Jacobi básico"""
    # Geração da malha
    xl, xr, yb, yt = 0, 1, 0, 1
    X = np.linspace(xl, xr, N, endpoint=True)
    Y = np.linspace(yb, yt, N, endpoint=True)
    
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Aplicar condições de contorno de Dirichlet
    U[0, :] = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0] = exact_solution(X, Y[0])
    U[:, -1] = exact_solution(X, Y[-1])
    
    Unew = U.copy()
    
    start_time = time.time()
    for k in range(max_iter):
        max_error = 0
        
        # Atualizar pontos internos
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                Unew[i, j] = 0.25 * (U[i-1, j] + U[i+1, j] + 
                                    U[i, j-1] + U[i, j+1] - 
                                    dx*dy * f(X[i], Y[j]))
                
                error = abs(Unew[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        # Verificar convergência
        if max_error < tol:
            break
            
        # Atualizar para próxima iteração
        U[:, :] = Unew[:, :]
    
    execution_time = time.time() - start_time
    
    # Calcular erro em relação à solução exata
    x, y = np.meshgrid(X, Y, indexing='ij')
    U_exact = exact_solution(x, y)
    error_norm = np.max(np.abs(U - U_exact))  # Norma infinito
    
    return U, k + 1, execution_time, max_error, error_norm

######### MÉTODO 2: Jacobi com Relaxamento (Omega) ##########
def jacobi_omega(N, dx, dy, omega=1.8, max_iter=10000, tol=1e-8):
    """Método de Jacobi com fator de relaxamento omega"""
    # Geração da malha
    xl, xr, yb, yt = 0, 1, 0, 1
    X = np.linspace(xl, xr, N, endpoint=True)
    Y = np.linspace(yb, yt, N, endpoint=True)
    
    U = np.zeros((N, N))
    Unew = np.zeros((N, N))
    
    # Aplicar condições de contorno
    U[0, :] = exact_solution(X[0], Y)
    U[-1, :] = exact_solution(X[-1], Y)
    U[:, 0] = exact_solution(X, Y[0])
    U[:, -1] = exact_solution(X, Y[-1])
    
    Unew = U.copy()
    
    start_time = time.time()
    for k in range(max_iter):
        max_error = 0
        
        for i in range(1, N - 1):
            for j in range(1, N - 1):
                # Cálculo do novo valor
                new_val = 0.25 * (U[i-1, j] + U[i+1, j] + 
                                 U[i, j-1] + U[i, j+1] - 
                                 dx*dy * f(X[i], Y[j]))
                
                # Aplicar relaxamento
                Unew[i, j] = omega * new_val + (1 - omega) * U[i, j]
                
                error = abs(Unew[i, j] - U[i, j])
                if error > max_error:
                    max_error = error
        
        if max_error < tol:
            break
            
        U[:, :] = Unew[:, :]
    
    execution_time = time.time() - start_time
    
    # Calcular erro em relação à solução exata
    x, y = np.meshgrid(X, Y, indexing='ij')
    U_exact = exact_solution(x, y)
    error_norm = np.max(np.abs(U - U_exact))  # Norma infinito
    
    return U, k + 1, execution_time, max_error, error_norm

################# ANÁLISE COMPARATIVA ####################

def run_comparative_analysis(N_values: List[int], omega_values: List[float] = None):
    """Executa análise comparativa para diferentes valores de N"""
    if omega_values is None:
        omega_values = [1.0, 1.5, 1.8, 1.9]
    
    results = []
    
    for N in N_values:
        print(f"\nAnalisando malha {N}x{N}...")
        
        # Cálculo do espaçamento
        dx = 1.0 / (N - 1)
        dy = 1.0 / (N - 1)
        
        # Método Jacobi Básico (omega = 1.0)
        U_basic, iter_basic, time_basic, error_basic, norm_basic = jacobi_basic(N, dx, dy)
        
        results.append({
            'N': N,
            'dx': dx,
            'Metodo': 'Jacobi_Basico',
            'Omega': 1.0,
            'Iteracoes': iter_basic,
            'Tempo_Execucao': time_basic,
            'Erro_Convergencia': error_basic,
            'Norma_Erro': norm_basic
        })
        
        print(f"  Jacobi Básico: {iter_basic} iterações, {time_basic:.4f}s, erro: {norm_basic:.2e}")
        
        # Métodos com relaxamento
        for omega in omega_values:
            if omega == 1.0:  # Já calculado acima
                continue
                
            U_omega, iter_omega, time_omega, error_omega, norm_omega = jacobi_omega(
                N, dx, dy, omega=omega)
            
            results.append({
                'N': N,
                'dx': dx,
                'Metodo': 'Jacobi_Relaxamento',
                'Omega': omega,
                'Iteracoes': iter_omega,
                'Tempo_Execucao': time_omega,
                'Erro_Convergencia': error_omega,
                'Norma_Erro': norm_omega
            })
            
            print(f"  Jacobi ω={omega}: {iter_omega} iterações, {time_omega:.4f}s, erro: {norm_omega:.2e}")
    
    return pd.DataFrame(results)

def plot_error_vs_spacing(df: pd.DataFrame):
    """Plota norma do erro em relação ao espaçamento"""
    plt.figure(figsize=(12, 8))
    
    # Gráfico 1: Erro vs dx para diferentes métodos
    plt.subplot(2, 2, 1)
    methods = df['Metodo'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(methods)))
    
    for i, method in enumerate(methods):
        method_data = df[df['Metodo'] == method]
        if method == 'Jacobi_Relaxamento':
            # Para relaxamento, plota cada omega separadamente
            omegas = method_data['Omega'].unique()
            for omega in omegas:
                omega_data = method_data[method_data['Omega'] == omega]
                plt.loglog(omega_data['dx'], omega_data['Norma_Erro'], 'o-', 
                          label=f'{method} (ω={omega})', alpha=0.8)
        else:
            plt.loglog(method_data['dx'], method_data['Norma_Erro'], 's-', 
                      label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Espaçamento (dx)')
    plt.ylabel('Norma do Erro ($L_\\infty$)')
    plt.title('Norma do Erro vs Espaçamento da Malha')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 2: Tempo de execução vs dx
    plt.subplot(2, 2, 2)
    for i, method in enumerate(methods):
        method_data = df[df['Metodo'] == method]
        if method == 'Jacobi_Relaxamento':
            omegas = method_data['Omega'].unique()
            for omega in omegas:
                omega_data = method_data[method_data['Omega'] == omega]
                plt.loglog(omega_data['dx'], omega_data['Tempo_Execucao'], 'o-', 
                          label=f'{method} (ω={omega})', alpha=0.8)
        else:
            plt.loglog(method_data['dx'], method_data['Tempo_Execucao'], 's-', 
                      label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Espaçamento (dx)')
    plt.ylabel('Tempo de Execução (s)')
    plt.title('Tempo Computacional vs Espaçamento')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 3: Iterações vs dx
    plt.subplot(2, 2, 3)
    for i, method in enumerate(methods):
        method_data = df[df['Metodo'] == method]
        if method == 'Jacobi_Relaxamento':
            omegas = method_data['Omega'].unique()
            for omega in omegas:
                omega_data = method_data[method_data['Omega'] == omega]
                plt.semilogx(omega_data['dx'], omega_data['Iteracoes'], 'o-', 
                            label=f'{method} (ω={omega})', alpha=0.8)
        else:
            plt.semilogx(method_data['dx'], method_data['Iteracoes'], 's-', 
                        label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Espaçamento (dx)')
    plt.ylabel('Número de Iterações')
    plt.title('Iterações vs Espaçamento')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Gráfico 4: Eficiência (Tempo/N²) vs N
    plt.subplot(2, 2, 4)
    for i, method in enumerate(methods):
        method_data = df[df['Metodo'] == method]
        if method == 'Jacobi_Relaxamento':
            omegas = method_data['Omega'].unique()
            for omega in omegas:
                omega_data = method_data[method_data['Omega'] == omega]
                efficiency = omega_data['Tempo_Execucao'] / (omega_data['N']**2)
                plt.loglog(omega_data['N'], efficiency, 'o-', 
                          label=f'{method} (ω={omega})', alpha=0.8)
        else:
            efficiency = method_data['Tempo_Execucao'] / (method_data['N']**2)
            plt.loglog(method_data['N'], efficiency, 's-', 
                      label=method, linewidth=2, markersize=8)
    
    plt.xlabel('Tamanho da Malha (N)')
    plt.ylabel('Tempo / N²')
    plt.title('Eficiência Computacional')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('analise_comparativa_jacobi.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(df: pd.DataFrame):
    """Cria tabela resumo dos resultados"""
    # Tabela detalhada
    summary_df = df.copy()
    
    # Formata as colunas para melhor visualização
    summary_df['dx'] = summary_df['dx'].map('{:.2e}'.format)
    summary_df['Tempo_Execucao'] = summary_df['Tempo_Execucao'].map('{:.4f}s'.format)
    summary_df['Norma_Erro'] = summary_df['Norma_Erro'].map('{:.2e}'.format)
    summary_df['Erro_Convergencia'] = summary_df['Erro_Convergencia'].map('{:.2e}'.format)
    
    # Reorganiza as colunas
    summary_df = summary_df[['N', 'dx', 'Metodo', 'Omega', 'Iteracoes', 
                           'Tempo_Execucao', 'Norma_Erro', 'Erro_Convergencia']]
    
    print("\n" + "="*100)
    print("TABELA RESUMO - ANÁLISE COMPARATIVA JACOBI")
    print("="*100)
    print(summary_df.to_string(index=False))
    print("="*100)
    
    # Salva em CSV
    summary_df.to_csv('tabela_comparativa_jacobi.csv', index=False)
    print("\nTabela salva em 'tabela_comparativa_jacobi.csv'")
    
    # Tabela resumida por método
    print("\n" + "="*80)
    print("ESTATÍSTICAS POR MÉTODO")
    print("="*80)
    
    stats = df.groupby(['Metodo', 'Omega']).agg({
        'Iteracoes': ['mean', 'std'],
        'Tempo_Execucao': ['mean', 'std'],
        'Norma_Erro': ['mean', 'std']
    }).round(4)
    
    print(stats)

def main():
    """Função principal"""
    # Valores de N para análise (diferentes espaçamentos)
    N_values = [20, 30, 50, 80, 100, 150, 200]
    
    # Valores de omega para testar
    omega_values = [1.0, 1.2, 1.5, 1.7, 1.8, 1.9]
    
    print("Iniciando análise comparativa Jacobi Básico vs Jacobi com Relaxamento")
    print(f"Malhas analisadas: {N_values}")
    print(f"Valores de omega: {omega_values}")
    
    # Executa a análise
    df_results = run_comparative_analysis(N_values, omega_values)
    
    # Gera os gráficos
    print("\nGerando gráficos...")
    plot_error_vs_spacing(df_results)
    
    # Gera tabelas
    print("\nGerando tabelas...")
    create_summary_table(df_results)
    
    # Análise do melhor método
    best_time = df_results.loc[df_results['Tempo_Execucao'].idxmin()]
    best_error = df_results.loc[df_results['Norma_Erro'].idxmin()]
    
    print("\n" + "="*60)
    print("MELHORES RESULTADOS")
    print("="*60)
    print(f"Menor tempo: {best_time['Metodo']} (ω={best_time['Omega']}) - {best_time['Tempo_Execucao']:.4f}s")
    print(f"Menor erro: {best_error['Metodo']} (ω={best_error['Omega']}) - {best_error['Norma_Erro']:.2e}")
    print("="*60)

if __name__ == "__main__":
    main()
