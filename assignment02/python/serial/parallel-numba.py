import numpy as np
from numpy import linalg
import time
import matplotlib.pyplot as plt
from numba import njit, prange
from numba import set_num_threads, get_num_threads

tol = 1e-08
T = 3000
incle = 100
metodo = "BDF4"
set_num_threads(8)

@njit
def prod(a, b):
    return np.array([a[0]*b[0] - a[1]*b[1], a[0]*b[1] + a[1]*b[0]])

@njit
def div(a, b):
    den = b[0]**2 + b[1]**2
    if abs(den) < 1e-14:
        return np.array([np.inf, np.inf])
    return np.array([(a[0]*b[0] + a[1]*b[1])/den, (a[1]*b[0] - a[0]*b[1])/den])

@njit
def euler_explicit(Un, z):
    return Un + prod(z, Un)

@njit
def euler_implict(Un, z):
    den = np.array([1.0,0.0]) - z
    return prod(div(np.array([1.0,0.0]), den), Un)

@njit
def trapezio(Un, z):
    num = np.array([1.0,0.0]) + z/2
    den = np.array([1.0,0.0]) - z/2
    return prod(div(num, den), Un)

@njit
def BDF2(Un1, Un, z):
    den = np.array([3.0,0.0]) - 2*z
    return prod(div(np.array([1.0,0.0]), den), 4*Un1 - Un)

@njit
def BDF3(Un2, Un1, Un, z):
    den = np.array([11.0,0.0]) - 6*z
    return prod(div(np.array([18.0,0.0]), den), Un2) \
- prod(div(np.array([9.0,0.0]), den), Un1) \
+ prod(div(np.array([2.0,0.0]), den), Un)

@njit
def BDF4(Un3, Un2, Un1, Un, z):
    den = np.array([25.0,0.0]) - 12*z
    return prod(div(np.array([48.0,0.0]), den), Un3) \
- prod(div(np.array([36.0,0.0]), den), Un2) \
+ prod(div(np.array([16.0,0.0]), den), Un1) \
- prod(div(np.array([3.0,0.0]), den), Un)

@njit(parallel=True)
def compute_mask(incle, T, tol, metodo):
    mask = np.zeros((incle, incle), dtype=np.bool_)
    for h in prange(incle):
        for k in range(incle):
            real_z = -10 + (h*(40/incle))
            img_z  = -20 + (k*(40/incle))
            z      = np.array([real_z, img_z])
            Un     = np.array([1.0, 0.0])

            if metodo == "BDF2":
                Un1 = euler_implict(Un, z)
            elif metodo == "BDF3":
                Un1 = euler_implict(Un, z)
                Un2 = BDF2(Un1, Un, z)
            elif metodo == "BDF4":
                Un1 = euler_implict(Un, z)
                Un2 = BDF2(Un1, Un, z)
                Un3 = BDF3(Un2, Un1, Un, z)

            for n in range(T):
                if metodo == "euler_exp":
                    Un1 = euler_explicit(Un, z)
                    Un = Un1
                elif metodo == "euler_imp":
                    Un1 = euler_implict(Un, z)
                    Un = Un1
                elif metodo == "trapezio":
                    Un1 = trapezio(Un, z)
                    Un = Un1
                elif metodo == "BDF2":
                    Un2 = BDF2(Un1, Un, z)
                    Un, Un1 = Un1, Un2
                elif metodo == "BDF3":
                    Un3 = BDF3(Un2, Un1, Un, z)
                    Un, Un1, Un2 = Un1, Un2, Un3
                elif metodo == "BDF4":
                    Un4 = BDF4(Un3, Un2, Un1, Un, z)
                    Un, Un1, Un2, Un3 = Un1, Un2, Un3, Un4

                norm = np.sqrt(Un1[0]**2 + Un1[1]**2)
                if norm < tol:
                    mask[h,k] = True
                    break
                elif norm > 1/tol:
                    break
    return mask

start_time = time.time()
mask = compute_mask(incle, T, tol, metodo)
end_time = time.time()
print(f"{end_time - start_time:.2f}")

x = -10 + np.arange(incle)*(40/incle)
y = -20 + np.arange(incle)*(40/incle)
X, Y = np.meshgrid(x, y, indexing='ij')

plt.figure(figsize=(8,8))
plt.axhline(0, color="k", alpha=0.5)
plt.axvline(0, color="k", alpha=0.5)
plt.scatter(X[mask], Y[mask], s=1, c='b')
plt.xlabel("Re(z)")
plt.ylabel("Im(z)")
plt.title(f"Região de estabilidade - {metodo}")
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()