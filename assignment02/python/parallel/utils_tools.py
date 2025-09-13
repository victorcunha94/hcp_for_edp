import numpy as np
from numba import njit, prange




@njit(cache=True)
def complex_add(a, b):
    """Adição de números complexos representados como [real, imag]"""
    return np.array([a[0] + b[0], a[1] + b[1]])



@njit(cache=True)
def prod(a, b):
  c = np.array([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]])
  return c

@njit(cache=True)
def div(a, b):
    den = b[0]**2 + b[1]**2
    if abs(den) < 1e-14:
        return np.array([np.inf, np.inf])
    return np.array([
        (a[0] * b[0] + a[1] * b[1]) / den,
        (a[1] * b[0] - a[0] * b[1]) / den
    ])

@njit(cache=True)
def div_a (a, b):
  den = b[0]**2 + b[1]**2
  c   = np.array([(a[0] * b[0] + a[1] * b[1])/den , (a[1] * b[0] - a[0] * b[1])/den] )
  return c




# Funções auxiliares para operações com números complexos como arrays [real, imag]
@njit(cache=True)
def complex_prod(a, b):
    """Multiplicação de números complexos representados como [real, imag]"""
    real = a[0] * b[0] - a[1] * b[1]
    imag = a[0] * b[1] + a[1] * b[0]
    return np.array([real, imag])


@njit(cache=True)
def complex_div(a, b):
    """Divisão de números complexos representados como [real, imag]"""
    denominator = b[0] * b[0] + b[1] * b[1]
    real = (a[0] * b[0] + a[1] * b[1]) / denominator
    imag = (a[1] * b[0] - a[0] * b[1]) / denominator
    return np.array([real, imag])


@njit(cache=True)
def complex_sub(a, b):
    """Subtração de números complexos representados como [real, imag]"""
    return np.array([a[0] - b[0], a[1] - b[1]])


@njit(cache=True)
def complex_norm(a):
    """Norma de número complexo representado como [real, imag]"""
    return np.sqrt(a[0] * a[0] + a[1] * a[1])

