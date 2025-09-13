import subprocess
import numpy as np
import matplotlib.pyplot as plt

codigo_seq = "main2_serial.py"
codigo_par = "main2_parallel.py"
n_reps = 3
jobs_list = [1, 2, 4, 8]  # diferentes números de processos para testar
incel_list = [50, 100, 200]  # diferentes dimensões da malha para testar
cores = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']  # cores para cada dimensão

def pegar_tempo(codigo, n_jobs=None, incel=None):
    tempos = []
    for _ in range(n_reps):
        cmd = ["python", codigo]
        if n_jobs is not None:
            cmd.append(str(n_jobs))  # passa n_jobs como argumento
        if incel is not None:
            cmd.append(str(incel))  # passa incel como argumento
        resultado = subprocess.run(cmd, capture_output=True, text=True)
        ultima_linha = resultado.stdout.strip().splitlines()[-1]
        tempos.append(float(ultima_linha))
    return np.mean(tempos)

plt.figure(figsize=(10, 6))

for i, incel in enumerate(incel_list):
    # tempo médio sequencial para esta dimensão da malha
    print(f"\nTestando incel={incel}")
    media_seq = pegar_tempo(codigo_seq, incel=incel)
    print(f"Tempo sequencial (incel={incel}): {media_seq:.3f} s")

    speedups = []
    tempos_par = []

    for n_jobs in jobs_list:
        media_par = pegar_tempo(codigo_par, n_jobs=n_jobs, incel=incel)
        tempos_par.append(media_par)
        speedup = media_seq / media_par
        speedups.append(speedup)
        print(f"  n_jobs={n_jobs}, tempo paralelo={media_par:.3f} s, speedup={speedup:.2f}")

    # Plot da curva de speedup para esta dimensão da malha
    cor = cores[i % len(cores)]
    plt.plot(jobs_list, speedups, 'o-', color=cor, label=f'Tamanho da malha={incel}')
    plt.plot(jobs_list, jobs_list, '--', color='gray', alpha=0.5, label='Speedup ideal' if i == 0 else "")

plt.xlabel('Número de processos')
plt.ylabel('Speedup')
plt.title('Comparação Sequencial vs Paralelo para Diferentes Dimensões da Malha')
plt.grid(True)
plt.legend()
plt.show()
