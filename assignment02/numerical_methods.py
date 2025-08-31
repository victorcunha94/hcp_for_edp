import numpy as np

# Problema teste: u'(t) = lambda * u(t), u(0) = 1
# Solução exata: u(t) = exp(lambda * t)

class ODEProblem:
    def __init__(self, func, initial_condition, exact_solution=None):
        self.func = func
        self.initial_condition = initial_condition
        self.exact_solution = exact_solution

# Métodos Numéricos de Passo Único

class EulerExplicit:
    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def solve(self, t_span, h, lambda_val):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)
        u_values[0] = self.ode_problem.initial_condition

        for i in range(num_steps):
            u_values[i+1] = u_values[i] + h * self.ode_problem.func(t_values[i], u_values[i], lambda_val)
        return t_values, u_values

class RungeKutta2:
    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def solve(self, t_span, h, lambda_val):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)
        u_values[0] = self.ode_problem.initial_condition

        for i in range(num_steps):
            k1 = self.ode_problem.func(t_values[i], u_values[i], lambda_val)
            k2 = self.ode_problem.func(t_values[i] + h, u_values[i] + h * k1, lambda_val)
            u_values[i+1] = u_values[i] + (h / 2) * (k1 + k2)
        return t_values, u_values

class RungeKutta4:
    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def solve(self, t_span, h, lambda_val):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)
        u_values[0] = self.ode_problem.initial_condition

        for i in range(num_steps):
            k1 = self.ode_problem.func(t_values[i], u_values[i], lambda_val)
            k2 = self.ode_problem.func(t_values[i] + h/2, u_values[i] + (h/2) * k1, lambda_val)
            k3 = self.ode_problem.func(t_values[i] + h/2, u_values[i] + (h/2) * k2, lambda_val)
            k4 = self.ode_problem.func(t_values[i] + h, u_values[i] + h * k3, lambda_val)
            u_values[i+1] = u_values[i] + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
        return t_values, u_values

# Métodos Numéricos de Passo Múltiplo

class AdamsBashforth2:
    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    def solve(self, t_span, h, lambda_val):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)

        # Usar Euler Explícito para o primeiro passo (u_1)
        u_values[0] = self.ode_problem.initial_condition
        u_values[1] = u_values[0] + h * self.ode_problem.func(t_values[0], u_values[0], lambda_val)

        for i in range(1, num_steps):
            f_n = self.ode_problem.func(t_values[i], u_values[i], lambda_val)
            f_n_minus_1 = self.ode_problem.func(t_values[i-1], u_values[i-1], lambda_val)
            u_values[i+1] = u_values[i] + (h / 2) * (3 * f_n - f_n_minus_1)
        return t_values, u_values

class AdamsMoulton2:
    def __init__(self, ode_problem):
        self.ode_problem = ode_problem

    # Para a abordagem Preditor-Corretor (PECE)
    def solve_predictor_corrector(self, t_span, h, lambda_val, max_iter=10, tol=1e-6):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)

        # Inicialização com Euler Explícito ou RK2
        u_values[0] = self.ode_problem.initial_condition
        # Usar RK2 para os dois primeiros passos (u_1, u_2)
        rk2 = RungeKutta2(self.ode_problem)
        _, u_init = rk2.solve((t0, t0 + 2*h), h, lambda_val)
        u_values[1:3] = u_init[1:3]

        for i in range(2, num_steps):
            # Preditor (Adams-Bashforth de 2 passos)
            f_n = self.ode_problem.func(t_values[i], u_values[i], lambda_val)
            f_n_minus_1 = self.ode_problem.func(t_values[i-1], u_values[i-1], lambda_val)
            u_pred = u_values[i] + (h / 2) * (3 * f_n - f_n_minus_1)

            # Corretor (Adams-Moulton de 2 passos)
            u_corr = u_pred # Primeira estimativa para a iteração
            for _ in range(max_iter):
                f_n_plus_1_corr = self.ode_problem.func(t_values[i+1], u_corr, lambda_val)
                u_next_candidate = u_values[i] + (h / 12) * (5 * f_n_plus_1_corr + 8 * f_n - f_n_minus_1)
                if np.abs(u_next_candidate - u_corr) < tol:
                    break
                u_corr = u_next_candidate
            u_values[i+1] = u_corr
        return t_values, u_values

    # Para a abordagem com Método de Newton
    def solve_newton(self, t_span, h, lambda_val, max_iter=100, tol=1e-8):
        t0, tf = t_span
        num_steps = int((tf - t0) / h)
        t_values = np.linspace(t0, tf, num_steps + 1)
        u_values = np.zeros(num_steps + 1, dtype=np.complex128)

        # Inicialização com Euler Explícito ou RK2
        u_values[0] = self.ode_problem.initial_condition
        # Usar RK2 para os dois primeiros passos (u_1, u_2)
        rk2 = RungeKutta2(self.ode_problem)
        _, u_init = rk2.solve((t0, t0 + 2*h), h, lambda_val)
        u_values[1:3] = u_init[1:3]

        # Definir a função G(u_n+1) para o Método de Newton
        # u_{n+1} - u_n - (h/12) * (5*f(t_{n+1}, u_{n+1}) + 8*f(t_n, u_n) - f(t_{n-1}, u_{n-1})) = 0
        # G(U) = U - u_n - (h/12) * (5*f(t_{n+1}, U) + 8*f(t_n, u_n) - f(t_{n-1}, u_{n-1}))
        # Para o problema teste f(t, u, lambda) = lambda * u
        # G(U) = U - u_n - (h/12) * (5*lambda*U + 8*lambda*u_n - lambda*u_{n-1})

        for i in range(2, num_steps):
            u_n = u_values[i]
            u_n_minus_1 = u_values[i-1]
            t_n = t_values[i]
            t_n_plus_1 = t_values[i+1]

            # Preditor para a estimativa inicial de Newton
            f_n = self.ode_problem.func(t_n, u_n, lambda_val)
            f_n_minus_1 = self.ode_problem.func(t_values[i-1], u_n_minus_1, lambda_val)
            u_guess = u_n + (h / 2) * (3 * f_n - f_n_minus_1) # Adams-Bashforth 2

            u_k = u_guess # Estimativa inicial para Newton

            for _ in range(max_iter):
                # G(U_k) = U_k - u_n - (h/12) * (5*lambda*U_k + 8*lambda*u_n - lambda*u_{n-1})
                G_uk = u_k - u_n - (h/12) * (5 * lambda_val * u_k + 8 * lambda_val * u_n - lambda_val * u_n_minus_1)

                # G'(U_k) = 1 - (h/12) * 5 * lambda
                dG_duk = 1 - (h/12) * 5 * lambda_val

                if dG_duk == 0: # Evitar divisão por zero, pode acontecer para certos lambda/h
                    break

                u_next = u_k - G_uk / dG_duk
                if np.abs(u_next - u_k) < tol:
                    u_k = u_next
                    break
                u_k = u_next
            u_values[i+1] = u_k
        return t_values, u_values