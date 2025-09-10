import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*
from pc_methods import*
import time
from joblib import Parallel, delayed

tol = 1e-08
T = 1000
tipo = "PC-AB3-AM3"
incle = 100

#dimensoes leveque:

#euler exp/AB2345
"""
xl = -3
xr = 1
yb = -2
yt = 2
"""

#euler imp
"""
xl = -1
xr = 3
yb = -2
yt = 2
"""

#runge-kutta
"""
xl = -5
xr = 2
yb = -4
yt = 4
"""

#BDF
"""
xl = -10
xr = 30
yb = -20
yt = 20
"""

#trapezio/ponto medio
"""
xl = -2
xr = 2
yb = -2
yt = 2
"""

#AM2345
"""
xl = -6
xr = 1
yb = -4
yt = 4
"""

#PECE

xl = -2
xr = 2
yb = -2
yt = 2


# Função para processar um ponto individual
def process_point(h, k, tipo):
    real_z = xl + (h*(np.abs(xr - xl)/incle))
    img_z  = yb + (k*(np.abs(yb - yt)/incle))
    z      = np.array([real_z, img_z])

    # Inicialização para todos os métodos
    Un = np.array([1, 0])

    # ---------------- Métodos de passo único ----------------
    if tipo == "euler_exp":
        for n in range(T):
            Un1 = euler_explicit(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    elif tipo == "euler_imp":
        for n in range(T):
            Un1 = euler_implict(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    elif tipo == "trapezio":
        for n in range(T):
            Un1 = trapezio(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    # ---------------- Métodos Runge-Kutta ----------------
    elif tipo == "RK2":
        for n in range(T):
            Un1 = runge_kutta2(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    elif tipo == "RK3":
        for n in range(T):
            Un1 = RK3(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    elif tipo == "RK4":
        for n in range(T):
            Un1 = RK4(Un, z)
            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
            Un = Un1
        return (real_z, img_z, False)

    # ---------------- Métodos BDF ----------------
    elif tipo == "BDF2":
        Un1 = euler_implict(Un, z)
        for n in range(T):
            Un2 = BDF2(Un1, Un, z)
            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1 = Un1, Un2
        return (real_z, img_z, False)

    elif tipo == "BDF3":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
        for n in range(T):
            Un3 = BDF3(Un2, Un1, Un, z)
            if linalg.norm(Un3, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un3, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2 = Un1, Un2, Un3
        return (real_z, img_z, False)

    elif tipo == "BDF4":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
        Un3 = BDF3(Un2, Un1, Un, z)
        for n in range(T):
            Un4 = BDF4(Un3, Un2, Un1, Un, z)
            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3 = Un1, Un2, Un3, Un4
        return (real_z, img_z, False)

    elif tipo == "BDF5":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
        Un3 = BDF3(Un2, Un1, Un, z)
        Un4 = BDF4(Un3, Un2, Un1, Un, z)
        for n in range(T):
            Un5 = BDF5(Un4, Un3, Un2, Un1, Un, z)
            if linalg.norm(Un5, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un5, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3, Un4 = Un1, Un2, Un3, Un4, Un5
        return (real_z, img_z, False)

    elif tipo == "BDF6":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
        Un3 = BDF3(Un2, Un1, Un, z)
        Un4 = BDF4(Un3, Un2, Un1, Un, z)
        Un5 = BDF5(Un4, Un3, Un2, Un1, Un, z)
        for n in range(T):
            Un6 = BDF6(Un5, Un4, Un3, Un2, Un1, Un, z)
            if linalg.norm(Un6, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un6, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3, Un4, Un5 = Un1, Un2, Un3, Un4, Un5, Un6
        return (real_z, img_z, False)

    # ---------------- Adams-Bashforth ----------------
    elif tipo == 'AB2':
        Un1 = euler_explicit(Un, z)
        for n in range(T):
            Un2 = AB2(Un1, Un, z)
            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1 = Un1, Un2
        return (real_z, img_z, False)

    elif tipo == 'AB3':
        Un1 = euler_explicit(Un, z)
        Un2 = AB2(Un1, Un, z)
        for n in range(T):
            Un3 = AB3(Un2, Un1, Un, z)
            if linalg.norm(Un3, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un3, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2 = Un1, Un2, Un3
        return (real_z, img_z, False)

    elif tipo == 'AB4':
        Un1 = euler_explicit(Un, z)
        Un2 = AB2(Un1, Un, z)
        Un3 = AB3(Un2, Un1, Un, z)
        for n in range(T):
            Un4 = AB4(Un3, Un2, Un1, Un, z)
            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3 = Un1, Un2, Un3, Un4
        return (real_z, img_z, False)

    elif tipo == 'AB5':
        Un1 = AB1(Un, z)
        Un2 = AB2(Un1, Un, z)
        Un3 = AB3(Un2, Un1, Un, z)
        Un4 = AB4(Un3, Un2, Un1, Un, z)
        for n in range(T):
            Un5 = AB5(Un4, Un3, Un2, Un1, Un, z)
            if linalg.norm(Un5, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un5, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3, Un4 = Un1, Un2, Un3, Un4, Un5
        return (real_z, img_z, False)

    # ---------------- Adams-Moulton ----------------
    elif tipo == 'AM2':
        Un1 = AM1(Un, z)  # AM1 é equivalente a Euler implícito
        for n in range(T):
            Un2 = AM2(Un1, Un, z)
            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1 = Un1, Un2
        return (real_z, img_z, False)

    elif tipo == 'AM3':
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)
        for n in range(T):
            Un3 = AM3(Un2, Un1, Un, z)
            if linalg.norm(Un3, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un3, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2 = Un1, Un2, Un3
        return (real_z, img_z, False)

    elif tipo == 'AM4':
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)
        Un3 = AM3(Un2, Un1, Un, z)
        for n in range(T):
            Un4 = AM4(Un3, Un2, Un1, Un, z)
            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3 = Un1, Un2, Un3, Un4
        return (real_z, img_z, False)

    elif tipo == 'AM5':
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)
        Un3 = AM3(Un2, Un1, Un, z)
        Un4 = AM4(Un3, Un2, Un1, Un, z)
        for n in range(T):
            Un5 = AM5(Un4, Un3, Un2, Un1, Un, z)
            if linalg.norm(Un5, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un5, 2) > 1/tol:
                return (real_z, img_z, False)
            Un, Un1, Un2, Un3, Un4 = Un1, Un2, Un3, Un4, Un5
        return (real_z, img_z, False)

    # ---------------- Preditor-Corretor ----------------
    elif tipo == 'PC-AB1-AM1':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)

        for n in range(T):
            Un1 = preditor_corrector_AB_AM(Un, Un1, z=z,
                                           preditor_order=1, corretor_order=1, n_correcoes=1)
            Un = Un1

            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1 / tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB2-AM2':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)

        for n in range(T):
            Un2 = preditor_corrector_AB_AM(Un, Un1, z=z,
                                           preditor_order=2, corretor_order=2, n_correcoes=1)
            Un = Un1
            Un1 = Un2

            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1 / tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB3-AM3':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        Un2 = euler_implict(Un1, z)

        for n in range(T):
            Un3 = preditor_corrector_AB_AM(Un, Un1, Un2, z=z,
                                           preditor_order=3, corretor_order=3, n_correcoes=1)
            Un = Un1
            Un1 = Un2
            Un2 = Un3

            if linalg.norm(Un3, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un3, 2) > 1 / tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB4-AM4':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        Un2 = euler_implict(Un1, z)
        Un3 = euler_implict(Un2, z)

        for n in range(T):
            Un4 = preditor_corrector_AB_AM(Un, Un1, Un2, Un3, z=z,
                                           preditor_order=4, corretor_order=4, n_correcoes=1)
            Un = Un1
            Un1 = Un2
            Un2 = Un3
            Un3 = Un4

            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1 / tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)


# Configuração básica
fig, ax = plt.subplots(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
ax.set_xlim(xl, xr)
ax.set_ylim(yb, yt)
plt.xlabel('Eixo Re(z)')
plt.ylabel('Eixo Im(z)')
plt.title(f'Região de Estabilidade - {tipo}')

# Paralelização com Joblib
t0 = time.time()

# Executar todos os pontos em paralelo
results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(process_point)(h, k, tipo)
    for h in range(incle)
    for k in range(incle)
)

# Processar resultados e plotar
for real_z, img_z, is_stable in results:
    if is_stable:
        if tipo in ['AM2', 'AM3']:
            plt.plot(real_z, img_z, 'bo', markersize=0.5)
        elif tipo in ['PC-AB3-AM3']:
            plt.plot(real_z, img_z, 'bo', markersize=2)
        elif tipo in ['PC-AB4-AM4']:
            plt.plot(real_z, img_z, 'bo', markersize=2)
        elif tipo in ['RK2', 'RK3', 'RK4']:
            plt.plot(real_z, img_z, 'bo', markersize=1)
        else:
            plt.plot(real_z, img_z, 'bo', markersize=0.5)

t1 = time.time()
print(f"{t1 - t0:.2f}")

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()