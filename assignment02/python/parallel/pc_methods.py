import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*


@njit(cache=True)
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
