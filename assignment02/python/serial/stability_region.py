import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*
from pc_methods import*

tol = 1e-08
T = 1000

#tipo= 'RK3'

tipo = "euler_implict"


def TR_BDF2 (Un,z):
  num   = [1,0] + z/4
  den   = [1,0] - z/4
  Ustar = prod(div(num,den),Un)
  num = [1,0]
  den = [3,0]-z
  Un1 = prod(div(num,den),4*Ustar - Un)
  return Un1


incle = 500
xl = -8
xr = 3
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


for h in range(incle):
  print(f"h = {h}")
  for k in range(incle):
    real_z = xl + (h*(np.abs(xr - xl)/incle))
    img_z  = yb + (k*(np.abs(yb - yt)/incle))
    z      = np.array([real_z, img_z])
    Un     = np.array([1, 0])
    # Un1    = euler_implict(Un, z)
    # Un2 = BDF2(Un1, Un, z)
    # Un3 = BDF3(Un2, Un1, Un, z)
    # Un4 = BDF4(Un3, Un2, Un1, Un, z)
    # Un5 = BDF5(Un4, Un3, Un2, Un1, Un, z)
    #Un1    = trapezio(Un, z)

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


if tipo =='AM2':
    for h in range(incle + 1):
      print(f"h = {h}")
      for k in range(incle + 1):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = AM1(Un, z)



        for n in range(T):
          Un2    = AM2_correct(Un1, Un, z)
          Un  = Un1
          Un1 = Un2

        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'ro', markersize=0.5)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break

if tipo =='AM3':
    for h in range(incle + 1):
      print(f"h = {h}")
      for k in range(incle + 1):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = AM1(Un, z)
        Un2    = AM2(Un1, Un, z)



        for n in range(T):
          Un3    = AM3(Un2, Un1, Un, z)
          Un  = Un1
          Un1 = Un2
          Un2 = Un3

        # print(Un)
          if linalg.norm(Un2, 2) < tol :
            plt.plot(real_z, img_z, 'ro', markersize=0.5)
            break
          elif linalg.norm(Un2, 2) > 1/tol:
            #plt.plot(real_z, img_z, 'ko', markersize=2)
            break

if tipo =='AM4':
    for h in range(incle + 1):
      print(f"h = {h}")
      for k in range(incle + 1):
        real_z = xl + (h*(np.abs(xr - xl)/incle))
        img_z  = yb + (k*(np.abs(yb - yt)/incle))
        z      = np.array([real_z, img_z])
        Un     = np.array([1, 0])
        Un1    = AM1(Un, z)
        Un2    = AM2(Un1, Un, z)
        Un3    = AM3(Un2, Un1, Un, z)

        for n in range(T):
          Un4 = AM4(Un3, Un2, Un1, Un, z)
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


if tipo == 'PC-AB1-AM1':  # Preditor-Corretor AB1-AM1
    for h in range(incle + 1):
        print(f"h = {h}")
        for k in range(incle + 1):
            real_z = xl + (h * (np.abs(xr - xl) / incle))
            img_z = yb + (k * (np.abs(yb - yt) / incle))
            z = np.array([real_z, img_z])
            Un = np.array([1, 0])
            Un1 = euler_implict(Un, z)


            for n in range(T):
                Un2 = preditor_corrector_AB_AM(Un1, Un, z, preditor_order=2, corretor_order=1, n_correcoes=1)
                Un = Un1


                if linalg.norm(Un1, 2) < tol:
                    plt.plot(real_z, img_z, 'bo', markersize=2)
                    break
                elif linalg.norm(Un1, 2) > 1 / tol:
                    break


if tipo == 'PC-AB2-AM2':  # Preditor-Corretor AB2-AM2
    for h in range(incle + 1):
        print(f"h = {h}")
        for k in range(incle + 1):
            real_z = xl + (h * (np.abs(xr - xl) / incle))
            img_z = yb + (k * (np.abs(yb - yt) / incle))
            z = np.array([real_z, img_z])
            Un = np.array([1, 0])
            Un1 = euler_implict(Un, z)

            for n in range(T):
                Un2 = preditor_corrector_AB_AM(Un1, Un, z, preditor_order=2, corretor_order=2, n_correcoes=1)
                Un = Un1
                Un1 = Un2

                if linalg.norm(Un2, 2) < tol:
                    plt.plot(real_z, img_z, 'bo', markersize=2)
                    break
                elif linalg.norm(Un2, 2) > 1 / tol:
                    break

if tipo == 'PC-AB3-AM3':  # Preditor-Corretor AB3-AM3
    for h in range(incle):
        print(f"h = {h}")
        for k in range(incle):
            real_z = xl + (h * (np.abs(xr - xl) / incle))
            img_z = yb + (k * (np.abs(yb - yt) / incle))
            z = np.array([real_z, img_z])
            Un = np.array([1, 0])
            Un1 = euler_implict(Un, z)
            # Para ordem 3, precisamos de mais um estado inicial
            Un2_initial = euler_implict(Un1, z)

            for n in range(T):
                Un3 = preditor_corrector_AB_AM(Un2_initial, Un1, z, preditor_order=3, corretor_order=3, n_correcoes=1)
                Un = Un1
                Un1 = Un2_initial
                Un2_initial = Un3

                if linalg.norm(Un3, 2) < tol:
                    plt.plot(real_z, img_z, 'ro', markersize=2)
                    break
                elif linalg.norm(Un3, 2) > 1 / tol:
                    break

if tipo == 'PC-AB4-AM4':  # Preditor-Corretor AB4-AM4
    for h in range(incle):
        print(f"h = {h}")
        for k in range(incle):
            real_z = xl + (h * (np.abs(xr - xl) / incle))
            img_z = yb + (k * (np.abs(yb - yt) / incle))
            z = np.array([real_z, img_z])
            Un = np.array([1, 0])
            Un1 = euler_implict(Un, z)
            Un2 = euler_implict(Un1, z)
            # Para ordem 4, precisamos de mais um estado inicial
            Un3_initial = euler_implict(Un2, z)

            for n in range(T):
                Un4 = preditor_corrector_AB_AM(Un3_initial, Un2, z, preditor_order=4, corretor_order=4, n_correcoes=2)
                Un = Un1
                Un1 = Un2
                Un2 = Un3_initial
                Un3_initial = Un4

                if linalg.norm(Un4, 2) < tol:
                    plt.plot(real_z, img_z, 'go', markersize=2)
                    break
                elif linalg.norm(Un4, 2) > 1 / tol:
                    break

plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
