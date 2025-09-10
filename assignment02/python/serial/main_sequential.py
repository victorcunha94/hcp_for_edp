# main_sequential.py
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import time
import panda as pd
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*
from pc_methods import*

# run_experiments.py
import subprocess
import json

methods = ['AB2', 'AM2', 'PC-AB2-AM2']
incle_values = [50, 100, 200]
process_counts = [1, 2, 4, 8]

experiment_results = {}

for method in methods:
    for incle in incle_values:
        for processes in process_counts:
            cmd = f"sbatch job_submission.sh --method {method} --incle {incle} --processes {processes}"
            result = subprocess.run(cmd, shell=True, capture_output=True)
            # Processar resultados

def calculate_stability_region(tipo, incle=100, xl=-3, xr=1, yb=-3, yt=3, T=1000, tol=1e-8):
    """Função unificada para cálculo de região de estabilidade"""
    # Implementação similar ao seu código, mas organizada
    pass

def benchmark_methods(methods_list, incle=50, runs=10):
    """Benchmark de múltiplos métodos"""
    results = {}
    for metodo in methods_list:
        tempos = []
        for i in range(runs):
            start_time = time.time()
            calculate_stability_region(metodo, incle=incle)
            tempos.append(time.time() - start_time)
        results[metodo] = tempos
    return results


# comparative_analysis.py
def generate_comparison_table(results):
    """Gera tabela comparativa dos métodos"""
    comparison_data = []

    for method, data in results.items():
        avg_time = np.mean(data['times'])
        std_time = np.std(data['times'])
        avg_speedup = np.mean(data['speedups'])

        comparison_data.append({
            'Método': method,
            'Tempo Médio (s)': f"{avg_time:.2f} ± {std_time:.2f}",
            'Speedup Médio': f"{avg_speedup:.2f}",
            'Região Estável': data['stable_region_percentage']
        })

    return pd.DataFrame(comparison_data)