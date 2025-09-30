# run_experiments.py
import subprocess
import sys
import time
from poisson_mpi_analytics import*



def run_mpi_experiments():
    """Executa o código MPI com diferentes números de processos"""
    mesh_sizes = [50, 100]
    process_counts = [1, 2, 4, 8]
    
    for N in mesh_sizes:
        for processes in process_counts:
            if processes == 1:
                # Caso sequencial
                cmd = f"python poisson_sequential.py --N {N}"
            else:
                # Caso paralelo
                cmd = f"mpirun -n {processes} python poisson_mpi_analytics.py --N {N}"
            
            print(f"Executando: {cmd}")
            start_time = time.time()
            
            try:
                result = subprocess.run(cmd.split(), capture_output=True, text=True)
                if result.returncode == 0:
                    print(f"✓ Sucesso: {processes} processos, malha {N}x{N}")
                else:
                    print(f"✗ Erro: {result.stderr}")
            except Exception as e:
                print(f"✗ Exceção: {e}")
            
            elapsed = time.time() - start_time
            print(f"Tempo de execução: {elapsed:.2f}s\n")

if __name__ == "__main__":
    run_mpi_experiments()
    #poisson_mpi_analytics()
    #jacobi_mpi_with_analytics()
