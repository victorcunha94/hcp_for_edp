import numpy as np
from numpy import linalg
import pandas as pd
import matplotlib.pyplot as plt
from utils_tools import *
from bdf_methods import *
from rk_methods import *
from adams_bashforth_moulton_methods import *
from pc_methods import *
from joblib import Parallel, delayed

####################### PARÂMETROS ###############
tol = 1e-08
T = 1000

incle = 10
xl = -2
xr = 2
yb = -2
yt = 2

##################### TIPO ##########################
tipo = "BDF2"  # Altere para o método desejado
#####################################################

# Lista para armazenar os dados
data = []

# Configuração básica
fig, ax = plt.subplots(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
ax.set_xlim(xl - 1, xr + 1)
ax.set_ylim(yb - 1, yt + 1)
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.title(f'{tipo}')


def process_point(real_z, img_z, z, tipo):
    """Processa um ponto e retorna o valor de estabilidade"""
    if tipo == 'TR_BDF2':
        Un = np.array([1, 0])
        for n in range(T):
            Un1 = TR_BDF2_explicit(Un, z)
            Un = Un1
            if linalg.norm(Un1, 2) < tol:
                return 1  # Estável
            elif linalg.norm(Un1, 2) > 1 / tol:
                return 0  # Instável
        return 0.5  # Indefinido

    elif tipo == 'BDF2':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        for n in range(T):
            Un2 = BDF2(Un1, Un, z)
            Un = Un1
            Un1 = Un2
            if linalg.norm(Un2, 2) < tol:
                return 1
            elif linalg.norm(Un2, 2) > 1 / tol:
                return 0
        return 0.5

    # Adicione aqui os outros métodos seguindo o mesmo padrão...
    # (BDF3, BDF4, RK4, RK3, AB2, AB3, AB4, AM2, AM3, AM4, AM5, PC-AB1-AM1, etc.)

    elif tipo == 'PC-AB4-AM4':
        Un = np.array([1, 0])
        Un1 = euler_implict(Un, z)
        Un2 = euler_implict(Un1, z)
        Un3_initial = euler_implict(Un2, z)

        for n in range(T):
            Un4 = preditor_corrector_AB_AM(Un, Un1, Un2, Un3=Un3_initial, z=z,
                                           preditor_order=4, corretor_order=4, n_correcoes=1)
            Un = Un1
            Un1 = Un2
            Un2 = Un3_initial
            Un3_initial = Un4

            if linalg.norm(Un4, 2) < tol:
                return 1
            elif linalg.norm(Un4, 2) > 1 / tol:
                return 0
        return 0.5

    # Padrão para métodos não implementados
    return 0.5


# Processa todos os pontos
for h in range(incle + 1):
    print(f"Processando linha {h}/{incle}")
    for k in range(incle + 1):
        real_z = xl + (h * (np.abs(xr - xl) / incle))
        img_z = yb + (k * (np.abs(yb - yt) / incle))
        z = np.array([real_z, img_z])

        # Calcula a estabilidade
        stable_value = process_point(real_z, img_z, z, tipo)

        # Adiciona aos dados
        data.append({
            'x': real_z,
            'y': img_z,
            'stable': stable_value
        })

        # Plota o ponto
        if stable_value == 1:
            plt.plot(real_z, img_z, 'bo', markersize=1)
        elif stable_value == 0:
            plt.plot(real_z, img_z, 'ro', markersize=1)
        else:
            plt.plot(real_z, img_z, 'go', markersize=1)

# Salva os dados em CSV
df = pd.DataFrame(data)
df.to_csv(tipo + '.csv', index=False)
print(f"Dados salvos em '{tipo}.csv'")

plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig(f'{tipo}_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Agora plota usando o formato do primeiro programa
plt.figure(figsize=(8, 8))
scatter = plt.scatter(
    df["x"], df["y"],
    c=df["stable"], cmap="plasma",
    s=0.5, marker="."
)

plt.xlabel("Real part (x)")
plt.ylabel("Imaginary part (y)")
plt.title(f"Stability in the Complex Plane - {tipo}")

cbar = plt.colorbar(scatter)
cbar.set_label("Stability value")

plt.savefig(f'{tipo}_colorplot.png', dpi=300, bbox_inches='tight')
plt.show()