import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*
from pc_methods import*
from joblib import Parallel, delayed
import time

tol = 1e-08
T = 2000
tipo = "BDF2"

incle = 200
xl = -11
xr = 30
yb = -20
yt = 20

# Configuração básica
fig, ax = plt.subplots(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
ax.set_xlim(xl - 1, xr + 1)
ax.set_ylim(yb - 1, yt + 1)
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Plano Cartesiano')

# Função para processar um ponto individual
def process_point(h, k, tipo):
    real_z = xl + (h*(np.abs(xr - xl)/incle))
    img_z  = yb + (k*(np.abs(yb - yt)/incle))
    z      = np.array([real_z, img_z])
    Un     = np.array([1,0])

    # Inicialização dependendo do método
    if tipo == "BDF2":
        Un1 = euler_implict(Un, z)
    elif tipo == "BDF3":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
    elif tipo == "BDF4":
        Un1 = euler_implict(Un, z)
        Un2 = BDF2(Un1, Un, z)
        Un3 = BDF3(Un2, Un1, Un, z)

    # ---------------- Iterações ----------------
    for n in range(T):
        if tipo == "euler_exp":
            Un1 = euler_explicit(Un, z)
            Un = Un1
        elif tipo == "euler_imp":
            Un1 = euler_implict(Un, z)
            Un = Un1
        elif tipo == "trapezio":
            Un1 = trapezio(Un, z)
            Un = Un1
        elif tipo == "BDF2":
            Un2 = BDF2(Un1, Un, z)
            Un = Un1
            Un1 = Un2
        elif tipo == "BDF3":
            Un3 = BDF3(Un2, Un1, Un, z)
            Un, Un1, Un2 = Un1, Un2, Un3
        elif tipo == "BDF4":
            Un4 = BDF4(Un3, Un2, Un1, Un, z)
            Un, Un1, Un2, Un3 = Un1, Un2,  Un3, Un4

        # ---------------- Condições de estabilidade ----------------
        if linalg.norm(Un1,2) < tol:
            return (real_z, img_z, True)
        elif linalg.norm(Un1,2) > 1/tol:
            return (real_z, img_z, False)

    if tipo == 'AB3':
        Un = np.array([1, 0])
        Un1 = euler_explicit(Un, z)
        Un2 = AB2(Un1, Un, z)

        for n in range(T):
            Un3 = AB3(Un2, Un1, Un, z)
            Un = Un1
            Un1 = Un2
            Un2 = Un3

            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'AB4':
        Un = np.array([1, 0])
        Un1 = euler_explicit(Un, z)
        Un2 = AB2(Un1, Un, z)
        Un3 = AB3(Un2, Un1, Un, z)

        for n in range(T):
            Un4 = AB4(Un3, Un2, Un1, Un, z)
            Un = Un1
            Un1 = Un2
            Un2 = Un3
            Un3 = Un4

            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'AM2':
        Un = np.array([1, 0])
        Un1 = AM1(Un, z)

        for n in range(T):
            Un2 = AM2_correct(Un1, Un, z)
            Un = Un1
            Un1 = Un2

            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'AM3':
        Un = np.array([1, 0])
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)

        for n in range(T):
            Un3 = AM3(Un2, Un1, Un, z)
            Un = Un1
            Un1 = Un2
            Un2 = Un3

            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'AM4':
        Un = np.array([1, 0])
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)
        Un3 = AM3(Un2, Un1, Un, z)

        for n in range(T):
            Un4 = AM4(Un3, Un2, Un1, Un, z)
            Un = Un1
            Un1 = Un2
            Un2 = Un3
            Un3 = Un4

            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'AM5':
        Un = np.array([1, 0])
        Un1 = AM1(Un, z)
        Un2 = AM2(Un1, Un, z)
        Un3 = AM3(Un2, Un1, Un, z)
        Un4 = AM4(Un3, Un2, Un1, Un, z)

        for n in range(T):
            Un5 = AM5(Un4, Un3, Un2, Un1, Un, z)
            Un = Un1
            Un1 = Un2
            Un2 = Un3
            Un3 = Un4
            Un4 = Un5

            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB1-AM1':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)

        for n in range(T):
            Un2 = preditor_corrector_AB_AM(Un1, Un, z, preditor_order=2, corretor_order=1, n_correcoes=1)
            Un = Un1

            if linalg.norm(Un1, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un1, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB2-AM2':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)

        for n in range(T):
            Un2 = preditor_corrector_AB_AM(Un1, Un, z, preditor_order=2, corretor_order=2, n_correcoes=1)
            Un = Un1
            Un1 = Un2

            if linalg.norm(Un2, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un2, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB3-AM3':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        Un2_initial = euler_implict(Un1, z)

        for n in range(T):
            Un3 = preditor_corrector_AB_AM(Un2_initial, Un1, z, preditor_order=3, corretor_order=3, n_correcoes=1)
            Un = Un1
            Un1 = Un2_initial
            Un2_initial = Un3

            if linalg.norm(Un3, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un3, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    elif tipo == 'PC-AB4-AM4':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        Un2 = euler_implict(Un1, z)
        Un3_initial = euler_implict(Un2, z)

        for n in range(T):
            Un4 = preditor_corrector_AB_AM(Un3_initial, Un2, z, preditor_order=4, corretor_order=4, n_correcoes=2)
            Un = Un1
            Un1 = Un2
            Un2 = Un3_initial
            Un3_initial = Un4

            if linalg.norm(Un4, 2) < tol:
                return (real_z, img_z, True)
            elif linalg.norm(Un4, 2) > 1/tol:
                return (real_z, img_z, False)
        return (real_z, img_z, False)

    return (real_z, img_z, False)

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
            plt.plot(real_z, img_z, 'ro', markersize=0.5)
        elif tipo in ['PC-AB3-AM3']:
            plt.plot(real_z, img_z, 'ro', markersize=2)
        elif tipo in ['PC-AB4-AM4']:
            plt.plot(real_z, img_z, 'go', markersize=2)
        else:
            plt.plot(real_z, img_z, 'bo', markersize=0.5)

t1 = time.time()
print(f"{t1 - t0:.2f}")

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()