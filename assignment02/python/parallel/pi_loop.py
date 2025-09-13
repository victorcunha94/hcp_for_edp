from numba import njit, prange, config, set_num_threads
import time
import threading
import os



# Configurações
config.THREADING_LAYER = 'omp'
os.environ['OMP_NUM_THREADS'] = '4'  # Define número de threads

pi_valor = 3.141592653589793


@njit(cache=True)
def calc_pi():
    num_steps = 100000
    step = 1.0 / num_steps

    the_sum = 0.0

    for j in prange(num_steps):
        x = (j + 0.5) * step
        the_sum += 4.0 / (1.0 + x * x)

    pi = step * the_sum
    return pi


# Teste com diferentes números de threads
for threads in [1, 2, 4, 8]:
    set_num_threads(threads)
    print(f"\nTestando com {threads} thread(s):")

    start = time.time()
    result = calc_pi()
    elapsed = time.time() - start

    print(f"π = {result:.10f}")
    print(f"Tempo: {elapsed:.4f}s")
    print(f"Threads ativas: {threading.active_count()}")
