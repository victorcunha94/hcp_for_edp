
import petsc4py
import sys
import numpy as np
import matplotlib.pyplot as plt
import time
import subprocess

petsc4py.init(sys.argv)

from petsc4py import PETSc
from mpi4py import MPI

# Parameters
N_POINTS_X = 129
N_POINTS_Y = 129
TIME_STEP_LENGTH = 0.001
N_TIME_STEPS = 50

def solve_with_petsc():
    """Executa a simulação PETSc e retorna tempo de execução"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"Running 2D Heat Equation with {size} processes")
        print(f"Grid: {N_POINTS_X} x {N_POINTS_Y} = {N_POINTS_X * N_POINTS_Y} DOFs")
    
    h = 1.0 / (N_POINTS_X - 1)
    total_dof = N_POINTS_X * N_POINTS_Y
    
    A = PETSc.Mat().create(comm)
    A.setSizes([total_dof, total_dof])
    A.setType('aij')  # Sparse matrix
    A.setUp()
    
    # Matrix coefficients
    alpha = TIME_STEP_LENGTH / h**2
    center_coeff = 1.0 + 4.0 * alpha
    neighbor_coeff = -alpha
    
    # Determine local range for this process
    local_size = total_dof // size
    remainder = total_dof % size
    
    if rank < remainder:
        local_start = rank * (local_size + 1)
        local_end = local_start + local_size + 1
    else:
        local_start = rank * local_size + remainder
        local_end = local_start + local_size
    
    if rank == 0:
        print(f"Assembling matrix in parallel...")
    
    # Assemble local portion of the matrix
    for global_idx in range(local_start, local_end):
        i = global_idx // N_POINTS_Y
        j = global_idx % N_POINTS_Y
        
        # Boundary conditions: Dirichlet
        if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
            A.setValue(global_idx, global_idx, 1.0)
        else:
            # Interior point - 5-point stencil
            A.setValue(global_idx, global_idx, center_coeff)
            
            # Left neighbor (i-1, j)
            if i > 0:
                left_idx = (i-1) * N_POINTS_Y + j
                A.setValue(global_idx, left_idx, neighbor_coeff)
            
            # Right neighbor (i+1, j)
            if i < N_POINTS_X - 1:
                right_idx = (i+1) * N_POINTS_Y + j
                A.setValue(global_idx, right_idx, neighbor_coeff)
            
            # Bottom neighbor (i, j-1)
            if j > 0:
                bottom_idx = i * N_POINTS_Y + (j-1)
                A.setValue(global_idx, bottom_idx, neighbor_coeff)
            
            # Top neighbor (i, j+1)
            if j < N_POINTS_Y - 1:
                top_idx = i * N_POINTS_Y + (j+1)
                A.setValue(global_idx, top_idx, neighbor_coeff)
    
    A.assemblyBegin()
    A.assemblyEnd()
    
    # Create vectors - also parallel
    b = PETSc.Vec().create(comm)
    b.setSizes(total_dof)
    b.setUp()
    
    x = PETSc.Vec().create(comm)
    x.setSizes(total_dof)
    x.setUp()
    
    # Initial condition - hot spot in the center
    # Only rank 0 initializes the entire array, then we scatter
    if rank == 0:
        initial_condition_global = np.zeros((N_POINTS_X, N_POINTS_Y))
        center_i, center_j = N_POINTS_X // 2, N_POINTS_Y // 2
        radius = min(N_POINTS_X, N_POINTS_Y) // 8
        
        for i in range(N_POINTS_X):
            for j in range(N_POINTS_Y):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist <= radius:
                    initial_condition_global[i, j] = 1.0 - (dist / radius)**2
        
        ic_flat = initial_condition_global.flatten()
    else:
        ic_flat = None
    
    # Scatter initial condition to all processes
    local_ic = np.zeros(local_end - local_start)
    comm.Scatterv(ic_flat, local_ic, root=0)
    
    # Set local portion of the vector
    for local_idx, global_idx in enumerate(range(local_start, local_end)):
        i = global_idx // N_POINTS_Y
        j = global_idx % N_POINTS_Y
        
        # Apply boundary conditions
        if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
            b.setValue(global_idx, 0.0)
        else:
            b.setValue(global_idx, local_ic[local_idx])
    
    b.assemblyBegin()
    b.assemblyEnd()
       
    
    ################## Setup linear solver  ##############################
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)

  
    ksp.setType('gmres')           # GMRES para matrizes não-simétricas
    ksp.setGMRESRestart(50)        # Restart a cada 50 iterações
    # ######################## Setup linear solver##########################
    
    
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)

    # CONFIGURAR SOLVER
    #ksp.setType('minres')
    #ksp.setType('symmlq')
    #ksp.setType('pipebcgs')
    ksp.setType('bcgs') 
    #ksp.setType('cg')               # GC para matrizes não-simétricas
    #ksp.setType('gmres')           # GMRES para matrizes não-simétricas
    #ksp.setGMRESRestart(50)        # Restart a cada 50 iterações

    # Configurar pré-condicionador
    pc = ksp.getPC()
    # pc.setType('bjacobi')          # Block Jacobi para paralelo
    # pc.setType('jacobi')            # Jacobi
    # pc.setType('sor')               # SOR (sequencial)
    # pc.setType('bjacobi')           # Block Jacobi (já está usando)
    # pc.setType('asm')               # Additive Schwarz
    # pc.setType('gasm')              # Restricted Additive Schwarz

    # Configurar tolerâncias
    ksp.setTolerances(rtol=1e-8, atol=1e-9, max_it=1000)
    ksp.setFromOptions()
    #############################################################################
    
    
    
    
    
    
    ######################################################################################
    
    
    
    if rank == 0:
        chosen_solver = ksp.getType()
        print(f"Solving with: {chosen_solver}")
    
    start_time = time.time()
    total_iterations = 0
    
    for time_step in range(N_TIME_STEPS):
        ksp.solve(b, x)

        # RECUPERAR INFORMAÇÕES DO SOLVER
        iterations = ksp.getIterationNumber()
        total_iterations += iterations
        converged_reason = ksp.getConvergedReason()
        
        # Verificar se convergiu
        if converged_reason <= 0:
            if rank == 0:
                print(f"AVISO: Solver não convergiu no passo {time_step}. Razão: {converged_reason}")
        
        if rank == 0 and time_step % 10 == 0:
            residual_norm = ksp.getResidualNorm()
            print(f"Time step {time_step:3d}: {iterations} iterações, resíduo = {residual_norm:.2e}")
        
        
        # Update RHS for next time step
>>>>>>> 8abc333789f8ca35c60905fcca5580f94a97abd4
        x.copy(b)
        
        for global_idx in range(local_start, local_end):
            i = global_idx // N_POINTS_Y
            j = global_idx % N_POINTS_Y
            
            if (i == 0 or i == N_POINTS_X - 1 or j == 0 or j == N_POINTS_Y - 1):
                b.setValue(global_idx, 0.0)
        
        b.assemblyBegin()
        b.assemblyEnd()
        
        if time_step % 10 == 0 or time_step == N_TIME_STEPS - 1:
            if rank == 0:
                solution_global = np.zeros(total_dof)
            else:
                solution_global = None
            
            local_solution = x.getArray()
            comm.Gatherv(local_solution, solution_global, root=0)
            
            if rank == 0:
                solution_2d = solution_global.reshape((N_POINTS_X, N_POINTS_Y))
                max_temp = np.max(solution_2d)
                min_temp = np.min(solution_2d)
                print(f"PETSc - Time step {time_step + 1:3d}: max T = {max_temp:.6f}, min T = {min_temp:.6f}")
                #print(f"{iterations = ksp.getIterationNumber()}"}
    
    petsc_time = time.time() - start_time
    
    if rank == 0:
        print(f"PETSc simulation completed in {petsc_time:.4f} seconds")
        print(f"Total de iterações: {total_iterations}")
        print(f"Média de iterações por passo de tempo: {total_iterations/N_TIME_STEPS:.1f}")
        print(f"Malha: {N_POINTS_X}x{N_POINTS_Y}, dx={h:.6f}, dy={h:.6f}")
        return petsc_time
    else:
        return None
        
        
        
def run_petsc_as_subprocess():
    print("Executando solução PETSc como subprocess...")
    start_time = time.time()
    
    try:
        result = subprocess.run([
            'mpirun', '-np', '4', 'python3', __file__,
            '-ksp_monitor', 
            '-ksp_converged_reason',
            '-ksp_type', 'bcgs', 
            '-pc_type', 'bjacobi',
            '-ksp_rtol', '1e-6',
            '-ksp_max_it', '200'
        ], capture_output=True, text=True, timeout=1000)
        
        petsc_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"PETSc executado com sucesso em {petsc_time:.4f} segundos")
            
            # EXTRAIR INFORMAÇÕES DETALHADAS
            total_iterations = 0
            convergence_info = ""
            
            for line in result.stdout.split('\n'):
                # Extrair iterações do monitor
                if 'KSP preconditioned resid' in line:
                    # Exemplo: "0 KSP preconditioned resid norm 1.000000000000e+00 true resid norm 1.000000000000e+00 ||r(i)||/||b|| 1.000000000000e+00"
                    parts = line.split()
                    if len(parts) > 0:
                        iteration = int(parts[0])
                        total_iterations = max(total_iterations, iteration + 1)  # +1 porque começa em 0
                
                # Extrair razão de convergência
                elif 'Converged reason:' in line:
                    convergence_info = line
                
                # Suas linhas existentes
                elif 'PETSc - Time step' in line or 'PETSc simulation completed' in line:
                    print(f"  {line}")
            
            print(f"  Iterações totais estimadas: {total_iterations}")
            if convergence_info:
                print(f"  {convergence_info}")
                
            return petsc_time, total_iterations
        else:
            print(f"Erro no PETSc: {result.stderr}")
            return float('inf'), 0
            
    except subprocess.TimeoutExpired:
        print("PETSc excedeu o tempo limite")
        return float('inf'), 0
    except Exception as e:
        print(f"Erro ao executar PETSc: {e}")
        return float('inf'), 0
        
        
        
        
def compare_solvers():
    """Compara PETSc vs Diferenças Finitas"""
    print("=== COMPARAÇÃO DE DESEMPENHO ===")
    
    # Verificar se é para executar PETSc diretamente ou como comparação
    if len(sys.argv) > 1 and any(arg.startswith('-ksp') or arg.startswith('-pc') for arg in sys.argv):
        # Modo PETSc direto
        solve_with_petsc()
    else:
        # Modo comparação
        try:
            from heat2d_v1 import solve_heat_2d_finite_difference
            
            # Executar diferenças finitas
            print("\n--- EXECUTANDO DIFERENÇAS FINITAS ---")
            fd_time = solve_heat_2d_finite_difference()
            
            # Executar PETSc
            print("\n--- EXECUTANDO PETSc ---")
            petsc_time = run_petsc_as_subprocess()
            
            # Resultados
            print(f"\n=== RESULTADOS ===")
            print(f"Diferenças Finitas: {fd_time:.4f} segundos")
            print(f"PETSc (4 processos): {petsc_time:.4f} segundos")
            
            if petsc_time != float('inf'):
                speedup = fd_time / petsc_time
                print(f"Speedup: {speedup:.2f}x")
                
        except ImportError:
            print("Não foi possível importar solve_heat_2d_finite_difference")
            print("Executando apenas PETSc...")
            solve_with_petsc()

if __name__ == "__main__":
    # Decide o modo de execução baseado nos argumentos
    if MPI.COMM_WORLD.Get_rank() == 0:
        if len(sys.argv) > 1 and any(arg.startswith('-ksp') or arg.startswith('-pc') for arg in sys.argv):
            # Modo PETSc puro - executado via mpirun
            solve_with_petsc()
        else:
            # Modo comparação
            compare_solvers()
    else:
        # Processos MPI não-rank 0
        if len(sys.argv) > 1 and any(arg.startswith('-ksp') or arg.startswith('-pc') for arg in sys.argv):
            solve_with_petsc()
