import matplotlib.pyplot as plt
import numpy as np

def plot_solutions(t_values_list, u_values_list, exact_solution_func, lambda_val, method_names, title, filename):
    plt.figure(figsize=(10, 6))
    t_exact = t_values_list[0] # Assumimos que todos os t_values são os mesmos
    u_exact = exact_solution_func(t_exact, lambda_val)
    plt.plot(t_exact, u_exact.real, 'k--', label='Solução Exata (Real)')
    plt.plot(t_exact, u_exact.imag, 'k:', label='Solução Exata (Imag)')

    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for i, (t_vals, u_vals) in enumerate(zip(t_values_list, u_values_list)):
        plt.plot(t_vals, u_vals.real, colors[i % len(colors)] + '-', label=f'{method_names[i]} (Real)')
        plt.plot(t_vals, u_vals.imag, colors[i % len(colors)] + '--', label=f'{method_names[i]} (Imag)')

    plt.xlabel('t')
    plt.ylabel('u(t)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Saved solution plot to {filename}")

def plot_convergence_order(h_values_list, errors_list, method_names, title, filename):
    plt.figure(figsize=(10, 6))
    for i, (h_vals, errs) in enumerate(zip(h_values_list, errors_list)):
        plt.loglog(h_vals, errs, 'o-', label=method_names[i])

    # Plotar linhas de referência para as ordens esperadas
    # Euler Ex.: O(h^1), RK2: O(h^2), RK4: O(h^4), AB2: O(h^2), AM2: O(h^3) (para PECE) ou O(h^2) para o implícito dependendo do erro local.
    # Vamos usar as ordens teóricas de erro global: Euler (1), RK2 (2), AB2 (2), AM2 (2) para o problema linear, RK4 (4)
    # Para Adams-Moulton2, o erro local é O(h^3), então o global é O(h^2).

    # Exemplo de linhas de referência (ajustar conforme os métodos)
    h_ref = np.array([1e-2, 1e-1])
    plt.loglog(h_ref, 0.1 * h_ref**1, 'k--', label='Ordem 1 (Ref)')
    plt.loglog(h_ref, 0.1 * h_ref**2, 'g--', label='Ordem 2 (Ref)')
    plt.loglog(h_ref, 0.1 * h_ref**3, 'b--', label='Ordem 3 (Ref)') # AM2 local order
    plt.loglog(h_ref, 0.1 * h_ref**4, 'r--', label='Ordem 4 (Ref)')


    plt.xlabel('Tamanho do Passo (h)')
    plt.ylabel('Erro Global Máximo (|u_numerico(tf) - u_exato(tf)|)')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig(filename)
    plt.close()
    print(f"Saved convergence order plot to {filename}")