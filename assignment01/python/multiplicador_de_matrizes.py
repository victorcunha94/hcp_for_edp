import random
import time
import sys
import numpy as np
import matplotlib.pyplot as plt


def alloc_matrix(n):
    return [[0.0 for _ in range(n)] for _ in range(n)]


def fill_matrix(n, mat):
    for i in range(n):
        row = mat[i]
        for j in range(n):
            row[j] = random.randint(-10000, 10000)


def zero_matrix(n, mat):
    for i in range(n):
        row = mat[i]
        for j in range(n):
            row[j] = 0.0


def mat_mult_ijk(n, A, B, C): #manter
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            sum_ij = 0.0
            for k in range(n):
                sum_ij += Ai[k] * B[k][j]
            Ci[j] = sum_ij


def mat_mult_ikj(n, A, B, C): # manter
    for i in range(n):
        Ci = C[i]
        Ai = A[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]


def mat_mult_jik(n, A, B, C):
    for j in range(n):
        for i in range(n):
            sum_ij = 0.0
            for k in range(n):
                sum_ij += A[i][k] * B[k][j]
            C[i][j] = sum_ij


def mat_mult_jki(n, A, B, C):
    for j in range(n):
        for k in range(n):
            bkj = B[k][j]
            for i in range(n):
                C[i][j] += A[i][k] * bkj


def mat_mult_kij(n, A, B, C):
    for k in range(n):
        Bk = B[k]
        for i in range(n):
            aik = A[i][k]
            Ci = C[i]
            for j in range(n):
                Ci[j] += aik * Bk[j]


def mat_mult_kji(n, A, B, C):
    for k in range(n):
        for j in range(n):
            bkj = B[k][j]
            for i in range(n):
                C[i][j] += A[i][k] * bkj


def measure_cpu_time(func, *args, **kwargs):
    t0 = time.process_time()
    func(*args, **kwargs)
    t1 = time.process_time()
    return t1 - t0


def numpy_mat_mult(n, A, B):
    A_np = np.array(A)
    B_np = np.array(B)
    return A_np @ B_np


def main():
    sizes = [150, 500, 800]  # Tamanhos menores para demonstração
    random.seed(int(time.time()))

    results = {
        'ijk': [], 'ikj': [], 'jik': [], 'jki': [], 'kij': [], 'kji': [], 'numpy': []
    }
    # results = {
    #     'ijk': [], 'ikj': [], 'jik': [], 'jki': [], 'numpy': []
    # }

    for n in sizes:
        print(f"Tamanho da matriz: {n}x{n}")
        A = alloc_matrix(n)
        B = alloc_matrix(n)
        C = alloc_matrix(n)

        fill_matrix(n, A)
        fill_matrix(n, B)

        # Medir tempo para cada permutação
        for method_name in ['ijk', 'ikj', 'jik', 'jki', 'kij', 'kji']:
        #for method_name in ['ijk', 'ikj', 'jik', 'jki']:
            zero_matrix(n, C)
            method_func = globals()[f'mat_mult_{method_name}']
            elapsed = measure_cpu_time(method_func, n, A, B, C)
            results[method_name].append(elapsed)
            print(f"{method_name}: {elapsed:.3f} segundos")

        # Medir tempo para NumPy
        t0 = time.process_time()
        numpy_result = numpy_mat_mult(n, A, B)
        elapsed_numpy = time.process_time() - t0
        results['numpy'].append(elapsed_numpy)
        print(f"numpy: {elapsed_numpy:.3f} segundos")
        print()

    # Criar gráfico
    plt.figure(figsize=(12, 8))

    colors = ['purple', 'orange', 'blue', 'brown', 'green','cyan', 'black']
    line_styles = ['-', ':', '--', ':','-.', '--', '-']

    for i, (method, times) in enumerate(results.items()):
        plt.plot(sizes, times, label=method, color=colors[i],
                 linestyle=line_styles[i], marker='o', linewidth=2)

    plt.xlabel('Tamanho da Matriz (n x n)')
    plt.ylabel('Tempo de Execução (segundos)')
    plt.title('Comparação de Performance: Multiplicação de Matrizes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.yscale('log')  # Escala logarítmica para melhor visualização
    #plt.xscale('log')

    # Salvar gráfico
    plt.savefig('comparacao_multiplicacao_matrizes.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Exibir tabela de resultados
    print("\nTabela de Resultados:")
    print(f"{'Tamanho':<8}", end="")
    for method in results.keys():
        print(f"{method:<10}", end="")
    print()

    for i, n in enumerate(sizes):
        print(f"{n:<8}", end="")
        for method in results.keys():
            print(f"{results[method][i]:<10.3f}", end="")
        print()


if __name__ == "__main__":
    main()

