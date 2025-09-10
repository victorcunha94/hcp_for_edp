import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*



def preditor_corrector_AB_AM(Un, Un1, Un2=None, Un3=None, z=None,
                             preditor_order=2, corretor_order=2, n_correcoes=1):
    """
    Preditor-Corretor seguindo P - E - (C - E)^n
    - Un, Un1, Un2, Un3: históricos como [real, imag], onde Un é o mais antigo e Un3 o mais recente
    - z: [real, imag] (h * lambda)
    - preditor_order: 1..4 (usa suas funções AB1..AB4)
    - corretor_order: 1..4 (aplica corretores na forma iterativa, como no C)
    - n_correcoes: número de iterações (nCE)
    Retorna: u_est final como [real, imag]
    """
    if z is None:
        raise ValueError("z não pode ser None (passe z = [real, imag]).")

    # montar histórico (apenas entradas não-None)
    history = [Un, Un1, Un2, Un3]
    hist = [h for h in history if h is not None]
    if len(hist) < 2:
        raise ValueError("É necessário pelo menos dois valores históricos (Un, Un1).")

    # helpers elementares
    def add(a,b): return [a[0]+b[0], a[1]+b[1]]
    def sub(a,b): return [a[0]-b[0], a[1]-b[1]]
    def scale(a, alpha): return [alpha*a[0], alpha*a[1]]

    # --- PREDITOR: escolhe a função AB adequada usando os últimos elementos (mais recente é hist[-1]) ---
    if preditor_order == 1:
        u_pred = AB1(hist[-1], z)
    elif preditor_order == 2:
        if len(hist) < 2: raise ValueError("AB2 requer 2 estados históricos.")
        u_pred = AB2(hist[-1], hist[-2], z)
    elif preditor_order == 3:
        if len(hist) < 3: raise ValueError("AB3 requer 3 estados históricos.")
        u_pred = AB3(hist[-1], hist[-2], hist[-3], z)
    elif preditor_order == 4:
        if len(hist) < 4: raise ValueError("AB4 requer 4 estados históricos.")
        u_pred = AB4(hist[-1], hist[-2], hist[-3], hist[-4], z)
    else:
        raise NotImplementedError("preditor_order deve ser 1..4")

    # --- ESTIMATIVA inicial (E): f_pred = z * u_pred ---
    f_arr = prod(z, u_pred)       # retorna array-like [re,im]
    f_est = [float(f_arr[0]), float(f_arr[1])]

    # quem é "u_n" (o passo mais recente antes do preditor)? -> hist[-1]
    u_n = hist[-1]
    u_nm1 = hist[-2] if len(hist) >= 2 else None
    u_nm2 = hist[-3] if len(hist) >= 3 else None

    # --- LOOP CORRETOR: (C then E) repeated n_correcoes times ---
    u_est = u_pred
    for _ in range(n_correcoes):
        if corretor_order == 1:
            # implicit Euler fixed-point (como no C)
            # u_est = u_n + f_est
            u_est = add(u_n, f_est)

        elif corretor_order == 2:
            # trapezoid (AM2) iterativo: u_est = u_n + 0.5*(z*u_n + f_est)
            z_un_arr = prod(z, u_n)
            z_un = [float(z_un_arr[0]), float(z_un_arr[1])]
            avg = [0.5 * (z_un[0] + f_est[0]), 0.5 * (z_un[1] + f_est[1])]
            u_est = add(u_n, avg)

        elif corretor_order == 3:
            # AM3 iterative as in C:
            # u_est = u_n + (5/12) f_est + (2/3) z*u_n - (1/12) z*u_{n-1}
            if u_nm1 is None:
                raise ValueError("AM3 requer u_{n-1} disponível (Un2).")
            z_un_arr = prod(z, u_n); z_un = [float(z_un_arr[0]), float(z_un_arr[1])]
            z_um1_arr = prod(z, u_nm1); z_um1 = [float(z_um1_arr[0]), float(z_um1_arr[1])]
            term = add(scale(f_est, 5.0/12.0), scale(z_un, 2.0/3.0))
            term = sub(term, scale(z_um1, 1.0/12.0))
            u_est = add(u_n, term)

        elif corretor_order == 4:
            # AM4 iterative as in C:
            # u_est = u_n + (9/24) f_est + (19/24) z*u_n - (5/24) z*u_{n-1} + (1/24) z*u_{n-2}
            if u_nm1 is None or u_nm2 is None:
                raise ValueError("AM4 requer u_{n-1} e u_{n-2} disponíveis (Un2, Un1).")
            z_un_arr = prod(z, u_n); z_un = [float(z_un_arr[0]), float(z_un_arr[1])]
            z_um1_arr = prod(z, u_nm1); z_um1 = [float(z_um1_arr[0]), float(z_um1_arr[1])]
            z_um2_arr = prod(z, u_nm2); z_um2 = [float(z_um2_arr[0]), float(z_um2_arr[1])]
            term = add(scale(f_est, 9.0/24.0), scale(z_un, 19.0/24.0))
            term = sub(term, scale(z_um1, 5.0/24.0))
            term = add(term, scale(z_um2, 1.0/24.0))
            u_est = add(u_n, term)

        else:
            raise NotImplementedError("corretor_order deve ser 1..4")

        # --- E (re-avaliação) após a correção ---
        f_arr = prod(z, u_est)
        f_est = [float(f_arr[0]), float(f_arr[1])]

    # u_est já é o U^{próximo} (e f_est é f(U^{próximo}))
    return u_est




# --- helpers de conversão ---
def to_c(u):
    """array [real, imag] -> python complex"""
    return complex(u[0], u[1])

def to_arr(u):
    """python complex -> array [real, imag]"""
    return np.array([u.real, u.imag])

# --- preditores (para f(u) = z * u, com z já representando h*lambda) ---
def AB_pred(u_nm3, u_nm2, u_nm1, u_n, z, order):
    if order == 1:
        return u_n + z * u_n
    if order == 2:
        # AB2: (1 + 3/2 z) u_n - (1/2 z) u_{n-1}
        return (1.0 + 1.5 * z) * u_n - 0.5 * z * u_nm1
    if order == 3:
        # AB3: u_n + z*(23/12 u_n - 16/12 u_{n-1} + 5/12 u_{n-2})
        return u_n + z * ((23.0/12.0)*u_n - (16.0/12.0)*u_nm1 + (5.0/12.0)*u_nm2)
    if order == 4:
        # AB4: u_n + z*(55/24 u_n - 59/24 u_{n-1} + 37/24 u_{n-2} - 9/24 u_{n-3})
        return u_n + z * ((55.0/24.0)*u_n - (59.0/24.0)*u_nm1
                         + (37.0/24.0)*u_nm2 - (9.0/24.0)*u_nm3)
    raise ValueError("preditor order must be 1..4")

# --- corretores iterativos (usando a forma iterativa com f_est = z * u_est) ---
def AM_correct_iter(u_nm3, u_nm2, u_nm1, u_n, u_pred, z, order, n_iter):
    # u_pred is complex initial estimate
    u_est = u_pred
    for _ in range(n_iter):
        f_est = z * u_est  # avaliação em t_{n+1} com a estimativa
        if order == 1:
            # implicit Euler fixed-point style: u = u_n + f_est
            u_est = u_n + f_est
        elif order == 2:
            # trapezoid / AM2: u = u_n + 1/2 (z u_n + f_est)
            u_est = u_n + 0.5 * (z * u_n + f_est)
        elif order == 3:
            # AM3 iterative form (as in C): u = u_n + z*(5/12 u_est + 2/3 u_n - 1/12 u_{n-1})
            u_est = u_n + z * ((5.0/12.0) * u_est + (2.0/3.0) * u_n - (1.0/12.0) * u_nm1)
        elif order == 4:
            # AM4 iterative form (as in C): u = u_n + z*(9/24 u_est + 19/24 u_n -5/24 u_{n-1} + 1/24 u_{n-2})
            u_est = u_n + z * ((9.0/24.0) * u_est + (19.0/24.0) * u_n
                               - (5.0/24.0) * u_nm1 + (1.0/24.0) * u_nm2)
        else:
            raise ValueError("corretor order must be 1..4")
    return u_est

# --- função principal ---
def preditor_corrector_AB_AM_v3(Un, Un1, Un2=None, Un3=None, z_arr = None, preditor_order=2, corretor_order=2, n_correcoes=2):
    """
    Inputs:
      Un, Un1, Un2, Un3 : arrays [real, imag]
        correspondem a U^n, U^{n+1}, U^{n+2}, U^{n+3} (Un é o mais antigo)
      z_arr : [real, imag] representando z = h * lambda
      preditor_order, corretor_order : inteiros 1..4
      n_correcoes : número de iterações do corretor (nCE)
    Returns:
      Un4 : array [real, imag] correspondente a U^{n+4}
    """
    # converter para complex
    z = complex(z_arr[0], z_arr[1])
    u_n = to_c(Un)
    u_n1 = to_c(Un1)
    u_n2 = to_c(Un2)
    u_n3 = to_c(Un3)

    # Observação da convenção: na formulação ABk usamos
    # u_nm3 = u_{n-3}, u_nm2 = u_{n-2}, u_nm1 = u_{n-1}, u_n = u_n
    # Para produzir u_{n+1} (ou u_{n+4} na nomenclatura AB4), chamamos AB_pred com:
    #   u_nm3 = u_{n-3}, u_nm2 = u_{n-2}, u_nm1 = u_{n-1}, u_n = u_n (ajuste conforme ordem)
    # Dado que recebemos Un..Un3 com Un = U^n (o mais antigo) e Un3 = U^{n+3} (o mais recente),
    # a ordem a passar para AB_pred será: (u_n, u_n1, u_n2, u_n3) rotacionada como abaixo.

    # Mapear argumentos para AB_pred que assume (u_nm3,u_nm2,u_nm1,u_n)
    # Queremos o preditor para o próximo passo após u_n3, portanto:
    u_nm3 = u_n      # U^n  -> será u_{n-3} na chamada (relativa)
    u_nm2 = u_n1     # U^{n+1}
    u_nm1 = u_n2     # U^{n+2}
    u_n_current = u_n3  # U^{n+3} (o mais recente)
    # Nota: a fórmula de AB_pred está escrita de forma que 'u_n_current' é o termo mais recente.

    # preditor
    u_pred = AB_pred(u_nm3, u_nm2, u_nm1, u_n_current, z, preditor_order)

    # corretor iterativo (PECE style) usando a forma iterativa das AM (como no C)
    u_corr = AM_correct_iter(u_nm3, u_nm2, u_nm1, u_n_current, u_pred, z, corretor_order, n_correcoes)

    # retornar no formato [real, imag]
    return to_arr(u_corr)


def preditor_corrector_AB_AM_V2(Un1, Un, z, preditor_order=2, corretor_order=2, n_correcoes=1):
    """
    Implementação CORRIGIDA seguindo a lógica do código C
    """
    # Manter histórico (assumindo Un = u_n, Un1 = u_{n-1})
    u_n = Un.copy()
    u_nm1 = Un1.copy()

    # Para métodos de ordem superior, precisamos de mais histórico
    # (No código C, isso é gerenciado pelo loop principal)

    # -------------------------------
    # Predictor step
    # -------------------------------
    if preditor_order == 0:  # Euler
        u_pred = [u_n[0] + z[0] * u_n[0] - z[1] * u_n[1],
                  u_n[1] + z[0] * u_n[1] + z[1] * u_n[0]]

    elif preditor_order == 2:  # AB2
        # u_pred = (1 + 1.5z)*u_n - (0.5z)*u_nm1
        term1 = prod([1 + 1.5 * z[0], 1.5 * z[1]], u_n)
        term2 = prod([0.5 * z[0], 0.5 * z[1]], u_nm1)
        u_pred = [term1[0] - term2[0], term1[1] - term2[1]]

    elif preditor_order == 3:  # AB3
        # u_pred = u_n + z*((23/12)*u_n - (16/12)*u_nm1 + (5/12)*u_nm2)
        # (Implementação similar)
        pass

    else:  # Default to Euler
        u_pred = [u_n[0] + z[0] * u_n[0] - z[1] * u_n[1],
                  u_n[1] + z[0] * u_n[1] + z[1] * u_n[0]]

    # -------------------------------
    # PECE loop (Corrector)
    # -------------------------------
    u_est = u_pred

    for _ in range(n_correcoes):
        # EVALUATION step: f(u_est) = z * u_est
        f_est = prod(z, u_est)

        # Corrector step
        if corretor_order == 7:  # Euler Implícito
            # u_est = u_n + f_est
            u_est = [u_n[0] + f_est[0], u_n[1] + f_est[1]]

        elif corretor_order == 8 or corretor_order == 19:  # Trapezoidal/AM2
            # u_est = u_n + 0.5*(z*u_n + f_est)
            f_n = prod(z, u_n)
            term = [0.5 * (f_n[0] + f_est[0]), 0.5 * (f_n[1] + f_est[1])]
            u_est = [u_n[0] + term[0], u_n[1] + term[1]]

        elif corretor_order == 22:  # AM3
            # u_est = u_n + (5/12)*f_est + (2/3)*(z*u_n) - (1/12)*(z*u_nm1)
            f_n = prod(z, u_n)
            f_nm1 = prod(z, u_nm1)
            term1 = [5 / 12 * f_est[0], 5 / 12 * f_est[1]]
            term2 = [2 / 3 * f_n[0], 2 / 3 * f_n[1]]
            term3 = [1 / 12 * f_nm1[0], 1 / 12 * f_nm1[1]]
            u_est = [u_n[0] + term1[0] + term2[0] - term3[0],
                     u_n[1] + term1[1] + term2[1] - term3[1]]

        elif corretor_order == 23:  # AM4
            # u_est = u_n + (9/24)*f_est + (19/24)*(z*u_n) - (5/24)*(z*u_nm1) + (1/24)*(z*u_nm2)
            f_n = prod(z, u_n)
            f_nm1 = prod(z, u_nm1)
            # Para u_nm2, precisaríamos de mais histórico
            # Esta é uma limitação da interface atual
            f_nm2 = [0, 0]  # Placeholder - precisa ser ajustado

            term1 = [9 / 24 * f_est[0], 9 / 24 * f_est[1]]
            term2 = [19 / 24 * f_n[0], 19 / 24 * f_n[1]]
            term3 = [5 / 24 * f_nm1[0], 5 / 24 * f_nm1[1]]
            term4 = [1 / 24 * f_nm2[0], 1 / 24 * f_nm2[1]]

            u_est = [u_n[0] + term1[0] + term2[0] - term3[0] + term4[0],
                     u_n[1] + term1[1] + term2[1] - term3[1] + term4[1]]

    return u_est

def preditor_corrector_AB_AM_v1(Un1, Un, z, preditor_order=2, corretor_order=2, n_correcoes=1):
    """
    Método Preditor-Corretor com Adams-Bashforth como preditor e Adams-Moulton como corretor
    Formato compatível com BDF2: recebe Un1, Un e retorna Un2

    Parameters:
    Un1: estado U^{n+1} [real, imag]
    Un: estado U^n [real, imag]
    z: parâmetro complexo [real, imag]
    preditor_order: ordem do método Adams-Bashforth (2, 3, 4)
    corretor_order: ordem do método Adams-Moulton (1, 2, 3, 4)
    n_correcoes: número de iterações do corretor

    Returns:
    Un2: próximo estado U^{n+2} [real, imag]
    """
    # Criar histórico com os dois estados disponíveis
    history = [Un.copy(), Un1.copy()]

    # PREDITOR: Adams-Bashforth
    if preditor_order == 1:
        Un_pred = AB1(history[-1], z)
    if preditor_order == 2:
        Un_pred = AB2(history[-1], history[-2], z)
    elif preditor_order == 3:
        # Para ordem 3, precisamos de mais um estado anterior
        # Usar Euler implícito para estimar U^{n-1}
        Un_prev = euler_implict(history[-2], z)
        history.insert(0, Un_prev)
        Un_pred = AB3(history[-1], history[-2], history[-3], z)
    elif preditor_order == 4:
        # Para ordem 4, precisamos de mais dois estados anteriores
        Un_prev1 = euler_implict(history[-2], z)
        Un_prev2 = euler_implict(Un_prev1, z)
        history.insert(0, Un_prev2)
        history.insert(0, Un_prev1)
        Un_pred = AB4(history[-1], history[-2], history[-3], history[-4], z)
    else:
        raise ValueError("Ordem do preditor deve ser 2, 3 ou 4")

    # CORRETOR: Adams-Moulton (aplicado iterativamente)
    Un_corr = Un_pred

    for _ in range(n_correcoes):
        # Preparar histórico para o corretor
        corr_history = history.copy()
        corr_history.append(Un_corr)

        if corretor_order == 1:
            Un_corr = AM1(corr_history[-1], z)
        elif corretor_order == 2:
            Un_corr = AM2(corr_history[-1], corr_history[-2], z)
        elif corretor_order == 3:
            Un_corr = AM3(corr_history[-1], corr_history[-2], corr_history[-3], z)
        elif corretor_order == 4:
            Un_corr = AM4(corr_history[-1], corr_history[-2], corr_history[-3], corr_history[-4], z)
        else:
            raise ValueError("Ordem do corretor deve ser entre 1 e 4")

    return Un_corr