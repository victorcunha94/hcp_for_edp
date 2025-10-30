# compare_solvers.py
import time
import subprocess
import sys

def run_petsc_solution():
    """Executa a solução PETSc e mede o tempo"""
    print("Executando solução PETSc...")
    start_time = time.time()
    
    try:
        # Execute o código PETSc
        result = subprocess.run([
            'mpirun', '-np', '4', 'python3', 'petsc_heat2d.py', 
            '-ksp_monitor', 
            '-ksp_converged_reason',
            '-ksp_type', 'gmres', 
            '-pc_type', 'bjacobi',    
            '-ksp_rtol', '1e-6',
            '-ksp_gmres_restart', '50',
            '-ksp_max_it', '200'
        ], capture_output=True, text=True, timeout=1000)
        
        petsc_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"PETSc executado com sucesso em {petsc_time:.4f} segundos")
            # Extrair informações do output
            for line in result.stdout.split('\n'):
                if 'Time step' in line:
                    print(line)
        else:
            print(f"Erro no PETSc: {result.stderr}")
            petsc_time = float('inf')
            
    except subprocess.TimeoutExpired:
        print("PETSc excedeu o tempo limite")
        petsc_time = float('inf')
    except Exception as e:
        print(f"Erro ao executar PETSc: {e}")
        petsc_time = float('inf')
    
    return petsc_time

if __name__ == "__main__":
    # Importar a função de diferenças finitas
    from heat2d import solve_heat_2d_finite_difference
    
    print("=== COMPARAÇÃO DE DESEMPENHO ===")
    
    # Executar diferenças finitas
    fd_time = solve_heat_2d_finite_difference()
    
    # Executar PETSc (se disponível)
    petsc_time = run_petsc_solution()
    
    print(f"\n=== RESULTADOS ===")
    print(f"Diferenças Finitas: {fd_time:.4f} segundos")
    print(f"PETSc (4 processos): {petsc_time:.4f} segundos")
    
    if petsc_time != float('inf'):
        speedup = fd_time / petsc_time
        print(f"Speedup: {speedup:.2f}x")
