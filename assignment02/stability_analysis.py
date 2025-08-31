import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import ODEProblem, EulerExplicit, RungeKutta2, RungeKutta4, AdamsBashforth2, AdamsMoulton2

def func_test(t, u, lambda_val):
    return lambda_val * u

def exact_solution_test(t, lambda_val):
    return np.exp(lambda_val * t)

def estimate_stability_region(method_class, t_span, h, real_range, imag_range, grid_spacing, error_margin, method_name):
    print(f"Estimating stability region for {method_name}...")
    problem = ODEProblem(func_test, initial_condition=1.0, exact_solution=exact_solution_test)
    method = method_class(problem)

    re_vals = np.arange(real_range[0], real_range[1] + grid_spacing, grid_spacing)
    im_vals = np.arange(imag_range[0], imag_range[1] + grid_spacing, grid_spacing)

    stable_points_re = []
    stable_points_im = []
    unstable_points_re = []
    unstable_points_im = []

    # Para facilitar a paralelização, esta iteração principal seria um bom candidato
    for re_z in re_vals:
        for im_z in im_vals:
            z = re_z + 1j * im_z
            lambda_val = z / h

            # Para Adams-Moulton, precisamos especificar a abordagem
            if isinstance(method, AdamsMoulton2):
                if method_name == "Adams-Moulton 2 (PC)":
                    _, u_numeric = method.solve_predictor_corrector(t_span, h, lambda_val)
                elif method_name == "Adams-Moulton 2 (Newton)":
                    _, u_numeric = method.solve_newton(t_span, h, lambda_val)
                else:
                    raise ValueError("Unknown Adams-Moulton method name")
            else:
                _, u_numeric = method.solve(t_span, h, lambda_val)

            # Comparar a solução numérica no tempo final com a solução exata
            u_exact_tf = exact_solution_test(t_span[1], lambda_val)
            u_numeric_tf = u_numeric[-1]

            if np.abs(u_numeric_tf - u_exact_tf) < error_margin:
                stable_points_re.append(re_z)
                stable_points_im.append(im_z)
            else:
                unstable_points_re.append(re_z)
                unstable_points_im.append(im_z)
    print(f"Finished stability region for {method_name}.")
    return stable_points_re, stable_points_im, unstable_points_re, unstable_points_im

def plot_stability_region(stable_points_re, stable_points_im, unstable_points_re, unstable_points_im, method_name, filename="stability_region.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(unstable_points_re, unstable_points_im, color='pink', s=10, label='Instável')
    plt.scatter(stable_points_re, stable_points_im, color='black', s=10, label='Estável')
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.xlabel('Parte Real de z')
    plt.ylabel('Parte Imaginária de z')
    plt.title(f'Região de Estabilidade Absoluta: {method_name}')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(filename)
    plt.close()
    print(f"Saved stability region plot to {filename}")