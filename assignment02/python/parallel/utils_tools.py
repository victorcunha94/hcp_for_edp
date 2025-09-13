import numpy as np


def prod(a, b):
  c = np.array([a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]])
  return c


def div(a, b):
    den = b[0]**2 + b[1]**2
    if abs(den) < 1e-14:
        return np.array([np.inf, np.inf])
    return np.array([
        (a[0] * b[0] + a[1] * b[1]) / den,
        (a[1] * b[0] - a[0] * b[1]) / den
    ])


def div_a (a, b):
  den = b[0]**2 + b[1]**2
  c   = np.array([(a[0] * b[0] + a[1] * b[1])/den , (a[1] * b[0] - a[0] * b[1])/den] )
  return c
