import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt


tol = 1e-08
T = 1000

tipo= 'BDF2'

def prod(a, b):
  c = np.array([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]])
  return c

def div_a (a, b):
  den = b[0]**2 + b[1]**2
  c   = np.array([(a[0] * b[0] + a[1] * b[1])/den , (a[1] * b[0] - a[0] * b[1])/den] )
  return c

def div(a, b):
    den = b[0]**2 + b[1]**2
    if abs(den) < 1e-14:  # Se denominador ~ 0, trata como instável
        return np.array([np.inf, np.inf])
    return np.array([
        (a[0] * b[0] + a[1] * b[1]) / den,
        (a[1] * b[0] - a[0] * b[1]) / den
    ])


# a = np.array([1,0])
# b = np.array([3,4])

# print(div(a,b))

def runge_kutta(Un, z):
    Ustar = Un + prod([0.5, 0],prod(Un, z))
    Un1 = Un + prod(Ustar, z)
    return Un1

def euler_explicit(Un, z):
  Un1 = Un + prod(z,Un)
  return Un1

def euler_implict(Un, z):
  den = [1,0] - z
  Un1 = prod(div([1,0],den),Un)
  return Un1

def trapezio(Un, z):
  num = [1,0] + z/2
  den = [1,0] - z/2
  Un1 = prod(div(num,den),Un)
  return Un1

def BDF2(Un1,Un,z):
  num = [1,0]
  den = [3,0]-2*z
  Un2 = prod(div(num,den),4*Un1 - Un)
  return Un2

def BDF3(Un2, Un1, Un, z):
  num1 = [18,0] 
  num2 = [9,0]
  num3 = [2,0]
  den = [11,0] - 6*z
  Un3 = prod(div([18,0],den),Un2) - prod(div([9,0],den),Un1) + prod(div([2,0],den),Un)
  return Un3

def BDF4(Un3, Un2, Un1, Un, z):
  num1 = [48,0] 
  num2 = [36,0]
  num3 = [16,0]
  num4 = [3,0]
  den = [25,0] - 12*z
  Un4 = prod(div([48,0],den),Un3) - prod(div([36,0],den),Un2) \
        + prod(div([16,0],den),Un1) - prod(div([3,0],den),Un)
  return Un4

def BDF5(Un4, Un3, Un2, Un1, Un, z):
  num1 = [300,0] 
  num2 = [300,0]
  num3 = [200,0]
  num4 = [75,0]
  num5 = [12,0]
  den  = [137,0] - 60*z
  Un5 = prod(div(num1,den),Un4) - prod(div(num2,den),Un3) \
        + prod(div(num3,den),Un2) - prod(div(num4,den),Un1) \
        + prod(div(num5,den),Un)
  return Un5

def BDF6(Un5, Un4, Un3, Un2, Un1, Un, z):
  num1 = [360, 0]
  num2 = [450, 0]
  num3 = [400, 0]
  num4 = [225, 0]
  num5 = [72 , 0]
  num6 = [10 , 0]
  den  = [147, 0] - 60*z
  Un6  =  prod(div(num1, den), Un5) - prod(div(num2, den), Un4)\
	+ prod(div(num3, den), Un3) - prod(div(num4, den), Un2)\
	+ prod(div(num5, den), Un1) - prod(div(num6, den), Un)
  return Un6



def TR_BDF2 (Un,z):
  num   = [1,0] + z/4
  den   = [1,0] - z/4
  Ustar = prod(div(num,den),Un)
  num = [1,0]
  den = [3,0]-z
  Un1 = prod(div(num,den),4*Ustar - Un)
  return Un1

incle = 100
xl = -10
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




plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
