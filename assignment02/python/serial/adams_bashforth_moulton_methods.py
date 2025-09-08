from utils_tools import*



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

def AB2(Un1, Un, z):
  den = [2, 0]
  num1 = [2, 0] + 3*z
  parcela1 = div(num1, den)
  parcela2 = div(z, den)
  Un2 = prod(parcela1, Un1) - prod(parcela2, Un)

  return Un2

def AB2(Un1, Un, z):
  den = [2, 0]
  num1 = [2, 0] + 3*z
  parcela1 = div(num1, den)
  parcela2 = div(z, den)
  Un2 = prod(parcela1, Un1) - prod(parcela2, Un)

  return Un2


def AB3(Un2, Un1, Un, z):
    """
    Adams-Bashforth de 3ª ordem
    Fórmula: U^{n+3} = U^{n+2} + h/12 [23f(U^{n+2}) - 16f(U^{n+1}) + 5f(U^n)]
    Para f(U) = z*U
    """
    h = 1.0  # passo fixo para análise de estabilidade
    den = [12, 0]  # denominador 12

    # Termos do numerador
    term1 = prod([23 * h, 0], z)  # 23h*z
    term2 = prod([-16 * h, 0], z)  # -16h*z
    term3 = prod([5 * h, 0], z)  # 5h*z

    # Coeficientes para cada U
    coef_Un2 = [1, 0]  # coeficiente de U^{n+2}
    coef_Un2 = [coef_Un2[0] + div(term1, den)[0], coef_Un2[1] + div(term1, den)[1]]

    coef_Un1 = div(term2, den)  # coeficiente de U^{n+1}
    coef_Un = div(term3, den)  # coeficiente de U^n

    # Calcular U^{n+3}
    Un3 = prod(coef_Un2, Un2)
    Un3 = [Un3[0] + prod(coef_Un1, Un1)[0], Un3[1] + prod(coef_Un1, Un1)[1]]
    Un3 = [Un3[0] + prod(coef_Un, Un)[0], Un3[1] + prod(coef_Un, Un)[1]]

    return Un3



def AB4(Un3, Un2, Un1, Un, z):
    """
    Adams-Bashforth de 4ª ordem
    Fórmula: U^{n+4} = U^{n+3} + h/24 [55f(U^{n+3}) - 59f(U^{n+2}) + 37f(U^{n+1}) - 9f(U^n)]
    Para f(U) = z*U
    """
    h = 1.0  # passo fixo
    den = [24, 0]  # denominador 24

    # Termos do numerador
    term1 = prod([55 * h, 0], z)  # 55h*z
    term2 = prod([-59 * h, 0], z)  # -59h*z
    term3 = prod([37 * h, 0], z)  # 37h*z
    term4 = prod([-9 * h, 0], z)  # -9h*z

    # Coeficientes para cada U
    coef_Un3 = [1, 0]  # coeficiente de U^{n+3}
    coef_Un3 = [coef_Un3[0] + div(term1, den)[0], coef_Un3[1] + div(term1, den)[1]]

    coef_Un2 = div(term2, den)  # coeficiente de U^{n+2}
    coef_Un1 = div(term3, den)  # coeficiente de U^{n+1}
    coef_Un = div(term4, den)  # coeficiente de U^n

    # Calcular U^{n+4}
    Un4 = prod(coef_Un3, Un3)
    Un4 = [Un4[0] + prod(coef_Un2, Un2)[0], Un4[1] + prod(coef_Un2, Un2)[1]]
    Un4 = [Un4[0] + prod(coef_Un1, Un1)[0], Un4[1] + prod(coef_Un1, Un1)[1]]
    Un4 = [Un4[0] + prod(coef_Un, Un)[0], Un4[1] + prod(coef_Un, Un)[1]]

    return Un4



#####################################################################
###################Implicit Adams–Moulton methods####################
#####################################################################

def AM1(Un, z):
    """
    Adams-Moulton 1-step (Método do Trapézio)
    U^{n+1} = U^n + k/2 (f(U^n) + f(U^{n+1}))
    Para f(U) = z*U
    """
    h = 1.0  # passo fixo

    # Rearranjando: U^{n+1} - (h*z/2)U^{n+1} = U^n + (h*z/2)U^n
    # U^{n+1} (1 - h*z/2) = U^n (1 + h*z/2)

    coef_left = [1, 0]  # 1
    term_left = prod([-h / 2, 0], z)  # -h*z/2
    coef_left = [coef_left[0] + term_left[0], coef_left[1] + term_left[1]]

    coef_right = [1, 0]  # 1
    term_right = prod([h / 2, 0], z)  # h*z/2
    coef_right = [coef_right[0] + term_right[0], coef_right[1] + term_right[1]]

    # U^{n+1} = [ (1 + h*z/2) / (1 - h*z/2) ] * U^n
    G = div(coef_right, coef_left)
    Un1 = prod(G, Un)

    return Un1


def AM2(Un1, Un, z):
    """
    Adams-Moulton 2-step
    U^{n+2} = U^{n+1} + k/12 (-f(U^n) + 8f(U^{n+1}) + 5f(U^{n+2}))
    Para f(U) = z*U
    """
    h = 1.0

    # Rearranjando: U^{n+2} - (5h*z/12)U^{n+2} = U^{n+1} + (h*z/12)(-U^n + 8U^{n+1})

    coef_left = [1, 0]  # 1
    term_left = prod([-5 * h / 12, 0], z)  # -5h*z/12
    coef_left = [coef_left[0] + term_left[0], coef_left[1] + term_left[1]]

    coef_right1 = [1, 0]  # coeficiente de U^{n+1}
    term_right1 = prod([8 * h / 12, 0], z)  # 8h*z/12
    coef_right1 = [coef_right1[0] + term_right1[0], coef_right1[1] + term_right1[1]]

    coef_right2 = prod([-h / 12, 0], z)  # coeficiente de U^n: -h*z/12

    # U^{n+2} = [coef_right1/coef_left]U^{n+1} + [coef_right2/coef_left]U^n
    term1 = div(coef_right1, coef_left)
    term2 = div(coef_right2, coef_left)

    Un2 = prod(term1, Un1)
    Un2 = [Un2[0] + prod(term2, Un)[0], Un2[1] + prod(term2, Un)[1]]

    return Un2


def AM3(Un2, Un1, Un, z):
    """
    Adams-Moulton 3-step
    U^{n+3} = U^{n+2} + k/24 (f(U^n) - 5f(U^{n+1}) + 19f(U^{n+2}) + 9f(U^{n+3}))
    Para f(U) = z*U
    """
    h = 1.0

    # Rearranjando: U^{n+3} - (9h*z/24)U^{n+3} = U^{n+2} + (h*z/24)(U^n - 5U^{n+1} + 19U^{n+2})

    coef_left = [1, 0]  # 1
    term_left = prod([-9 * h / 24, 0], z)  # -9h*z/24
    coef_left = [coef_left[0] + term_left[0], coef_left[1] + term_left[1]]

    coef_right1 = [1, 0]  # coeficiente de U^{n+2}
    term_right1 = prod([19 * h / 24, 0], z)  # 19h*z/24
    coef_right1 = [coef_right1[0] + term_right1[0], coef_right1[1] + term_right1[1]]

    coef_right2 = prod([-5 * h / 24, 0], z)  # coeficiente de U^{n+1}: -5h*z/24
    coef_right3 = prod([h / 24, 0], z)  # coeficiente de U^n: h*z/24

    # U^{n+3} = [coef_right1/coef_left]U^{n+2} + [coef_right2/coef_left]U^{n+1} + [coef_right3/coef_left]U^n
    term1 = div(coef_right1, coef_left)
    term2 = div(coef_right2, coef_left)
    term3 = div(coef_right3, coef_left)

    Un3 = prod(term1, Un2)
    Un3 = [Un3[0] + prod(term2, Un1)[0], Un3[1] + prod(term2, Un1)[1]]
    Un3 = [Un3[0] + prod(term3, Un)[0], Un3[1] + prod(term3, Un)[1]]

    return Un3


def AM4(Un3, Un2, Un1, Un, z):
    """
    Adams-Moulton 4-step
    U^{n+4} = U^{n+3} + k/720 (-19f(U^n) + 106f(U^{n+1}) - 264f(U^{n+2}) + 646f(U^{n+3}) + 251f(U^{n+4}))
    Para f(U) = z*U
    """
    h = 1.0

    # Rearranjando: U^{n+4} - (251h*z/720)U^{n+4} = U^{n+3} + (h*z/720)(-19U^n + 106U^{n+1} - 264U^{n+2} + 646U^{n+3})

    coef_left = [1, 0]  # 1
    term_left = prod([-251 * h / 720, 0], z)  # -251h*z/720
    coef_left = [coef_left[0] + term_left[0], coef_left[1] + term_left[1]]

    coef_right1 = [1, 0]  # coeficiente de U^{n+3}
    term_right1 = prod([646 * h / 720, 0], z)  # 646h*z/720
    coef_right1 = [coef_right1[0] + term_right1[0], coef_right1[1] + term_right1[1]]

    coef_right2 = prod([-264 * h / 720, 0], z)  # coeficiente de U^{n+2}: -264h*z/720
    coef_right3 = prod([106 * h / 720, 0], z)  # coeficiente de U^{n+1}: 106h*z/720
    coef_right4 = prod([-19 * h / 720, 0], z)  # coeficiente de U^n: -19h*z/720

    # U^{n+4} = [coef_right1/coef_left]U^{n+3} + [coef_right2/coef_left]U^{n+2} + [coef_right3/coef_left]U^{n+1} + [coef_right4/coef_left]U^n
    term1 = div(coef_right1, coef_left)
    term2 = div(coef_right2, coef_left)
    term3 = div(coef_right3, coef_left)
    term4 = div(coef_right4, coef_left)

    Un4 = prod(term1, Un3)
    Un4 = [Un4[0] + prod(term2, Un2)[0], Un4[1] + prod(term2, Un2)[1]]
    Un4 = [Un4[0] + prod(term3, Un1)[0], Un4[1] + prod(term3, Un1)[1]]
    Un4 = [Un4[0] + prod(term4, Un)[0], Un4[1] + prod(term4, Un)[1]]

    return Un4