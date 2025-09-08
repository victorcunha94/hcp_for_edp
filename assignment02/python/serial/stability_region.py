import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*

tol = 1e-08
T = 1000

#tipo= 'RK3'

tipo= "AB4"


def TR_BDF2 (Un,z):
  num   = [1,0] + z/4
  den   = [1,0] - z/4
  Ustar = prod(div(num,den),Un)
  num = [1,0]
  den = [3,0]-z
  Un1 = prod(div(num,den),4*Ustar - Un)
  return Un1


def PreditorCorretor(Un, z, n_correcoes=1):
    """
    Método Preditor-Corretor: Euler Explícito (Preditor) + Euler Implícito (Corretor)
    Para a equação teste: y' = z*y

    Parameters:
    Un: estado atual [real, imag]
    z: parâmetro complexo [real, imag]
    n_correcoes: número de iterações do corretor

    Returns:
    Un1: próximo estado [real, imag]
    """
    h = 1.0  # passo fixo para análise de estabilidade

    # PREDITOR: Euler Explícito
    # y_{n+1}^{(0)} = y_n + h * f(t_n, y_n) = y_n + h * z * y_n
    preditor = prod([1, 0], Un)  # termo y_n
    termo_preditor = prod(prod(z, [h, 0]), Un)  # termo h*z*y_n
    Un1_pred = [preditor[0] + termo_preditor[0], preditor[1] + termo_preditor[1]]

    # CORRETOR: Euler Implícito (aplicado iterativamente)
    Un1_cor = Un1_pred

incle = 200
xl = -5
xr = 5
yb = -5
yt = 5

# incle = 100
# xl = -10
# xr = 30
# yb = -20
# yt = 20




# Configuração básica
fig, ax = plt.subplots(figsize=(8, 8))
plt.axhline(y=0, color='k', linestyle='-', alpha=0.5)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.5)
ax.set_xlim(xl - 1, xr + 1)
ax.set_ylim(yb - 1, yt + 1)
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Plano Cartesiano')


if tipo =='BDF2':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_implict(Un, z)
        for n in range(T):
          Un2 = BDF2(Un1, Un, z)
          Un  = Un1
          Un1 = Un2
        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break
elif tipo == 'BDF3':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_implict(Un, z)
        Un2    = BDF2(Un1, Un, z)
        for n in range(T):
          Un3 = BDF3(Un2, Un1, Un, z)
          Un  = Un1
          Un1 = Un2
          Un3 = Un2
        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break

elif tipo == 'BDF4':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_implict(Un, z)
        Un2    = BDF2(Un1, Un, z)
        for n in range(T):
          Un3 = BDF3(Un2, Un1, Un, z)
          Un  = Un1
          Un1 = Un2
          Un3 = Un2
        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break



if tipo =='RK4':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])

        for n in range(T):
          Un1 = RK4(Un, z)
          Un  = Un1

        # print(Un)
          if linalg.norm(Un1, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un1, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break


if tipo =='RK3':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])

        for n in range(T):
          Un1 = RK3(Un, z)
          Un  = Un1

        # print(Un)
          if linalg.norm(Un1, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un1, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break


if tipo =='PC_EulerExplicito+Implicito':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])

        for n in range(T):
          Un1 = PreditorCorretor(Un, z, 1)
          Un  = Un1

        # print(Un)
          if linalg.norm(Un1, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un1, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break


if tipo =='AB2':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_explicit(Un, z)

        for n in range(T):
          Un2 = AB2(Un1, Un, z)
          Un  = Un1
          Un1 = Un2
        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=2)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break



if tipo =='AB3':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_explicit(Un, z)
        Un2    = AB2(Un1, Un, z)

        for n in range(T):
          Un3 = AB3(Un2, Un1, Un, z)
          Un  = Un1
          Un1 = Un2
          Un2 = Un3

        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=0.5)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break

if tipo =='AB4':
    for h in range(incle):
      print(f"h = {h}")
      for k in range(incle):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = euler_explicit(Un, z)
        Un2    = AB2(Un1, Un, z)
        Un3    = AB3(Un2, Un1, Un, z)

        for n in range(T):
          Un4 = AB4(Un3, Un2, Un1, Un, z)
          Un  = Un1
          Un1 = Un2
          Un2 = Un3
          Un3 = Un4


        # print(Un)
          if linalg.norm(Un4, 2) < tol :
            plt.plot(real_z, img_z, 'bo', markersize=0.5)
            break
          elif linalg.norm(Un4, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
