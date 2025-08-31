import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import ODEProblem, EulerExplicit, RungeKutta2, RungeKutta4, AdamsBashforth2, AdamsMoulton2
from stability_analysis import func_test, exact_solution_test, estimate_stability_region, plot_stability_region
from plotting_utils import plot_solutions, plot_convergence_order
import os

if __name__ == "__main__":
    # Crie um diretório para salvar os gráficos, se não existir
    output_dir = "output_plots"
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------------------------------------
    # Configurações do Problema Teste
    t_span = (0.0, 10.0)
    initial_condition = 1.0
    h_fixed = 0.01
    lambda_val_stability = None # Será definido como z/h para a análise de estabilidade
    error_margin_stability = 0.1 # Margem de erro para a região de estabilidade

    ode_problem = ODEProblem(func_test, initial_condition=initial_condition, exact_solution=exact_solution_test)

    # Instanciando os métodos
    methods = {
        "Euler Explícito": EulerExplicit(ode_problem),
        "Runge-Kutta de Ordem 2": RungeKutta2(ode_problem),
        "Runge-Kutta de Ordem 4": RungeKutta4(ode_problem),
        "Adams-Bashforth de 2 passos": AdamsBashforth2(ode_problem),
        "Adams-Moulton de 2 passos (PC)": AdamsMoulton2(ode_problem), # Usará solve_predictor_corrector
        "Adams-Moulton de 2 passos (Newton)": AdamsMoulton2(ode_problem) # Usará solve_newton
    }

    # ----------------------------------------------------------------------------------------------------
    # 2. Estimar e Plotar a Região de Estabilidade Absoluta para cada método

    # Região de interesse para Z no plano complexo (para a malha)
    # Aumentada um pouco em relação ao exemplo [-3,1]x[-2,2]
    real_range_z = (-5.5, 3.5)
    imag_range_z = (-5.5, 3.5)
    grid_spacing_z = 0.1

    stability_results = {}

    for name, method_instance in methods.items():
        if "Adams-Moulton" in name:
            # Adams-Moulton 2 (PC)
            if name == "Adams-Moulton de 2 passos (PC)":
                stable_re, stable_im, unstable_re, unstable_im = estimate_stability_region(
                    AdamsMoulton2, t_span, h_fixed, real_range_z, imag_range_z, grid_spacing_z,
                    error_margin_stability, "Adams-Moulton 2 (PC)"
                )
                stability_results["Adams-Moulton 2 (PC)"] = (stable_re, stable_im, unstable_re, unstable_im)
                plot_stability_region(stable_re, stable_im, unstable_re, unstable_im,
                                      "Adams-Moulton 2 (PC)", os.path.join(output_dir, "stability_AM2_PC.png"))
            # Adams-Moulton 2 (Newton)
            elif name == "Adams-Moulton de 2 passos (Newton)":
                stable_re, stable_im, unstable_re, unstable_im = estimate_stability_region(
                    AdamsMoulton2, t_span, h_fixed, real_range_z, imag_range_z, grid_spacing_z,
                    error_margin_stability, "Adams-Moulton 2 (Newton)"
                )
                stability_results["Adams-Moulton 2 (Newton)"] = (stable_re, stable_im, unstable_re, unstable_im)
                plot_stability_region(stable_re, stable_im, unstable_re, unstable_im,
                                      "Adams-Moulton 2 (Newton)", os.path.join(output_dir, "stability_AM2_Newton.png"))
        else:
            # Outros métodos
            method_class = type(method_instance)
            stable_re, stable_im, unstable_re, unstable_im = estimate_stability_region(
                method_class, t_span, h_fixed, real_range_z, imag_range_z, grid_spacing_z,
                error_margin_stability, name
            )
            stability_results[name] = (stable_re, stable_im, unstable_re, unstable_im)
            plot_stability_region(stable_re, stable_im, unstable_re, unstable_im,
                                  name, os.path.join(output_dir, f"stability_{name.replace(' ', '_').replace('-', '_')}.png"))

    # ----------------------------------------------------------------------------------------------------
    # 3. Observar diferenças na região de estabilidade de Adams-Moulton 2

    print("\n--- Observações sobre Adams-Moulton 2 ---")
    print("Compare as regiões de estabilidade de Adams-Moulton 2 (PC) e Adams-Moulton 2 (Newton).")
    print("Normalmente, para o problema linear f(u) = lambda*u, o método de Newton converge para a mesma solução implícita que o PECE idealmente alcançaria.")
    print("As diferenças podem surgir de: tolerância de convergência, número máximo de iterações, ou a sensibilidade do método de Newton a estimativas iniciais ruins em casos não lineares.")
    print("Para este problema linear, espera-se que as regiões sejam muito semelhantes, senão idênticas, se os parâmetros de convergência forem bem ajustados.")
    print("No entanto, a implementação do preditor-corretor usa uma estimativa inicial de Adams-Bashforth, e o corretor itera, enquanto o Newton resolve a equação não linear. A precisão do Newton pode ser maior.")

    # ----------------------------------------------------------------------------------------------------
    # 4. Gerar gráficos das soluções numéricas (lambda = -1, h max na região de estabilidade)

    lambda_val_extreme = -1.0 # Lambda real negativo para convergência
    print(f"\n--- Gerando gráficos de soluções numéricas para lambda = {lambda_val_extreme} ---")

    # Para cada método, encontrar um 'h' máximo que o mantenha estável para lambda = -1.
    # Isso envolve verificar a condição |z| < R_estabilidade, onde z = lambda * h.
    # Para lambda = -1, z = -h. Precisamos que -h esteja na região de estabilidade para z.
    # Exemplo (aproximado, requer a região exata):
    # Euler explícito: |1 + z| < 1 => |-1 + h| < 1 => 0 < h < 2. Max h = 1.9 (para -1)
    # RK2: z deve estar no círculo de raio 1 com centro em (-1,0) (para a estabilidade da psi(z) dada).
    # Para lambda=-1, z=-h. Para h=2, z=-2. (-2,0) está na região de estabilidade de RK2.
    # RK4: região maior, vai permitir h maiores.
    # AB2: Região de estabilidade tem um pedaço no eixo real negativo, algo como h < 1.
    # AM2: Região de estabilidade é maior que AB2.

    # Vamos usar um h conservador para todos, ou um h específico para cada método
    # Para simplicidade, vamos usar um h que é estável para a maioria, e plotar individualmente se necessário.
    # Ou, idealmente, iterar sobre a região de estabilidade para lambda=-1 e encontrar o h_max
    
    # Para este problema linear u' = lambda * u, o fator de amplificação psi(z) é conhecido.
    # Podemos calcular o h_max para lambda=-1 de forma analítica se as regiões fossem simples.
    # Como as regiões são complexas, e para seguir o espírito do exercício,
    # vamos usar um h que sabemos ser estável para lambda=-1, ou determinar empiricamente.
    # Para lambda=-1, z = -h. Queremos que -h esteja na região de estabilidade.

    h_for_extreme_case = 0.1 # Um valor que geralmente é estável para a maioria.
    # Para Euler, se lambda = -1, z = -0.1. |1 - 0.1| = 0.9 < 1. Estável.
    # Para outros métodos, z=-0.1 também deve estar na região de estabilidade.

    all_t_values = []
    all_u_values = []
    all_method_names = []

    for name, method_instance in methods.items():
        if "Adams-Moulton" in name:
            if name == "Adams-Moulton de 2 passos (PC)":
                t_vals, u_vals = method_instance.solve_predictor_corrector(t_span, h_for_extreme_case, lambda_val_extreme)
            elif name == "Adams-Moulton de 2 passos (Newton)":
                t_vals, u_vals = method_instance.solve_newton(t_span, h_for_extreme_case, lambda_val_extreme)
        else:
            t_vals, u_vals = method_instance.solve(t_span, h_for_extreme_case, lambda_val_extreme)
        
        all_t_values.append(t_vals)
        all_u_values.append(u_vals)
        all_method_names.append(name)

    plot_solutions(all_t_values, all_u_values, exact_solution_test, lambda_val_extreme,
                   all_method_names, f'Soluções Numéricas para $\lambda = {lambda_val_extreme}$, h = {h_for_extreme_case}',
                   os.path.join(output_dir, 'numerical_solutions_lambda_minus_1.png'))
    

    # ----------------------------------------------------------------------------------------------------
    # 5. Gerar gráficos de ordem de convergência temporal
    print("\n--- Gerando gráficos de ordem de convergência temporal ---")

    # Valores de h a serem testados (pelo menos 5 malhas diferentes)
    h_values_to_test = np.array([0.2, 0.1, 0.05, 0.025, 0.0125])
    lambda_conv = -1.0 # Usar lambda = -1 para a análise de convergência

    all_h_values_conv = []
    all_errors_conv = []
    all_method_names_conv = []

    for name, method_instance in methods.items():
        errors_for_method = []
        for h_conv in h_values_to_test:
            if "Adams-Moulton" in name:
                if name == "Adams-Moulton de 2 passos (PC)":
                    t_vals, u_vals = method_instance.solve_predictor_corrector(t_span, h_conv, lambda_conv)
                elif name == "Adams-Moulton de 2 passos (Newton)":
                    t_vals, u_vals = method_instance.solve_newton(t_span, h_conv, lambda_conv)
            else:
                t_vals, u_vals = method_instance.solve(t_span, h_conv, lambda_conv)
            
            u_exact_tf = exact_solution_test(t_span[1], lambda_conv)
            error_tf = np.abs(u_vals[-1] - u_exact_tf)
            errors_for_method.append(error_tf)
        
        all_h_values_conv.append(h_values_to_test)
        all_errors_conv.append(errors_for_method)
        all_method_names_conv.append(name)

    plot_convergence_order(all_h_values_conv, all_errors_conv, all_method_names_conv,
                           f'Ordem de Convergência Temporal para $\lambda = {lambda_conv}$',
                           os.path.join(output_dir, 'convergence_order_plot.png'))

    # ----------------------------------------------------------------------------------------------------
    # 6. Comentários sobre os métodos
    print("\n--- Comentários Finais ---")
    print("A escolha do método mais vantajoso depende das características do problema e dos requisitos.")
    print("- **Euler Explícito:** Mais simples, mas exige passo 'h' pequeno para estabilidade e precisão. Ordem de convergência 1.")
    print("- **Runge-Kutta de Ordem 2 e 4:** Maior precisão (ordem 2 e 4, respectivamente) e regiões de estabilidade maiores que Euler Explícito.")
    print("  RK4 é um 'cavalo de batalha' para muitos problemas, oferecendo bom equilíbrio entre precisão e custo computacional por passo.")
    print("- **Adams-Bashforth de 2 passos (AB2):** Método de passo múltiplo explícito de ordem 2. Requer um método de passo único para iniciar.")
    print("  Pode ser mais eficiente que RK2 por requerer menos avaliações da função 'f' por passo, mas tem restrições de estabilidade.")
    print("- **Adams-Moulton de 2 passos (AM2):** Método de passo múltiplo implícito de ordem 2 (global).")
    print("  Oferece melhor estabilidade que AB2 e RK2 para o mesmo 'h'.")
    print("  A implementação via Preditor-Corretor")
