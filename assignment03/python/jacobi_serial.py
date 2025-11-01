#!/usr/bin/env python3

import numpy as np
import time
import argparse
import pandas as pd
import os
from numpy.linalg import norm




tempo_1 = time.time()
tempo_1 = time.time()


def jacobi_serial(N, max_iter=1000000, tol=1e-9, L=1.0):
    """
    Versão serial do método de Jacobi para comparação de desempenho
    """
    dx = L / (N - 1)
    dx2 = dx * dx
    
    # Alocar arrays (incluindo bordas)
    U = np.zeros((N, N), dtype=np.float64)
    Unew = np.zeros((N, N), dtype=np.float64)
    
    # Pré-calcular f(x,y)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y, indexing='ij')
    f = np.sin(4*np.pi * X) * np.sin(4*np.pi * Y)
    
    # Condições de contorno de Dirichlet nulas já estão garantidas pela inicialização com zeros
    
    t_start = time.perf_counter()
    exec_time = 0.0
    final_error = 0.0

    tempo_1 = 0.0
    tempo_2 = 0.0
    tempo_3 = 0.0


    
    for iteration in range(max_iter):
        t_iter_start = time.perf_counter()

        tempo_1_start = time.perf_counter()
        # Kernel de Jacobi (pontos internos apenas)
        Unew[1:-1, 1:-1] = 0.25 * (
            U[:-2, 1:-1] + U[2:, 1:-1] +
            U[1:-1, :-2] + U[1:-1, 2:] +
            dx2 * f[1:-1, 1:-1]
        )
        tempo_1_end = time.perf_counter()
        tempo_1 += tempo_1_end - tempo_1_start


        # Calcular erro
        tempo_2_start = time.perf_counter()
        # norm_1 = norm(Unew - U, np.inf)
        # norm_2 = norm(Unew, np.inf)
        # current_err = norm_1 / norm_2
        current_err = np.max(np.abs(Unew[1:-1, 1:-1] - U[1:-1, 1:-1])) / np.max(np.abs(Unew[1:-1, 1:-1]))
        final_error = current_err
        tempo_2_end = time.perf_counter()
        tempo_2 += tempo_2_end - tempo_2_start
        

        
        # Swap arrays
        tempo_3_start = time.perf_counter()
        U, Unew = Unew, U
        tempo_3_end = time.perf_counter()
        tempo_3 += tempo_3_end - tempo_3_start



        exec_time += time.perf_counter() - t_iter_start
        
        # Verificar convergência
        if current_err < tol:
            break
    
    total_time = time.perf_counter() - t_start
    print(f"Tempo 1 - {tempo_1}")
    print(f"Tempo 2 - {tempo_2}")
    print(f"Tempo 3 - {tempo_3}")
    meta = dict(
        rank=0,
        method="serial",
        N=N,
        iterations=iteration + 1,
        exec_time=exec_time,
        total_time=total_time,
        final_error=final_error,
        comm_time=0.0,
        overhead=0.0
    )
    
    return meta, U[1:-1, 1:-1]

def main():
    parser = argparse.ArgumentParser(description="Jacobi Serial Solver")
    parser.add_argument("--N", type=int, default=256, help="Tamanho total da grade (N x N pontos)")
    parser.add_argument("--max_iter", type=int, default=1000000, help="Número máximo de iterações")
    parser.add_argument("--tol", type=float, default=1e-9, help="Tolerância para convergência")
    parser.add_argument("--L", type=float, default=1.0, help="Tamanho do domínio")
    parser.add_argument("--output", type=str, default="serial_results.csv", help="Arquivo de saída para resultados")
    
    args = parser.parse_args()
    
    print(f"Executando Jacobi Serial com N={args.N}")
    print(f"Parâmetros: max_iter={args.max_iter}, tol={args.tol}, L={args.L}")
    
    # Executar o solver serial
    meta, solution = jacobi_serial(
        N=args.N,
        max_iter=args.max_iter,
        tol=args.tol,
        L=args.L
    )
    
    # Exibir resultados
    print("\n=== RESULTADOS JACOBI SERIAL ===")
    print(f"Iterações: {meta['iterations']}")
    print(f"Erro final: {meta['final_error']:.8e}")
    print(f"Tempo total: {meta['total_time']:.4f} segundos")
    print(f"Tempo computação: {meta['exec_time']:.4f} segundos")
    print(f"Dimensão da solução: {solution.shape}")
    print("================================\n")
    
    # Salvar resultados em CSV
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Criar DataFrame com os resultados
    results_df = pd.DataFrame([meta])
    
    # Salvar metadados
    output_path = f"output/{args.output}"
    results_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")
    
    # Salvar a solução em arquivo separado
    solution_path = f"output/serial_solution_N{args.N}.csv"
    np.savetxt(solution_path, solution, delimiter=',')
    print(f"Solução salva em: {solution_path}")
    
    return meta

if __name__ == "__main__":
    main()
