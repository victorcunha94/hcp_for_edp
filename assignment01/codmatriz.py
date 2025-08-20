import random
import time
import sys


def alloc_matrix(n):
    return [[0.0 for _ in range(n)] for _ in range(n)]

def fill_matrix(n, mat):
    for i in range(n):
        row = mat[i]
        for j in range(n):
            row[j] = random.randint(-10000,10000) #numeros aleatorios em [-10000,10000]

def zero_matrix(n, mat):
    for i in range(n):
        row = mat[i]
        for j in range(n):
            row[j] = 0.0

def mat_mult_ijk(n, A, B, C):
    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for j in range(n):
            sum_ij = 0.0
            for k in range(n):
                sum_ij += Ai[k] * B[k][j]
            Ci[j] += sum_ij

def mat_mult_ikj(n, A, B, C):
    for i in range(n):
        Ci = C[i]
        Ai = A[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]

def measure_cpu_time(func, *args, **kwargs):
    t0 = time.process_time()
    func(*args, **kwargs)
    t1 = time.process_time()
    return t1 - t0

def main():
    sizes = [1000,1500,2000,3000] #tamanho das matrizes [conforme n aumenta, ikj > ijk]
    random.seed(int(time.time()))

    for n in sizes:
        print(f"tamanho da matriz: {n}x{n}")
        A = alloc_matrix(n)
        B = alloc_matrix(n)
        C = alloc_matrix(n)

        fill_matrix(n, A)
        fill_matrix(n, B)

        zero_matrix(n, C)
        elapsed_ijk = measure_cpu_time(mat_mult_ijk, n, A, B, C)
        print(f"modelo ijk: tempo = {elapsed_ijk:.3f} segundos")

        zero_matrix(n, C)
        elapsed_ikj = measure_cpu_time(mat_mult_ikj, n, A, B, C)
        print(f"modelo ikj: tempo = {elapsed_ikj:.3f} segundos")

        print()

if __name__ == "__main__":
    main()