from utils_tools import*
from numba import njit


@njit(cache=True)
def euler_explicit(Un, z):
  Un1 = Un + prod(z,Un)
  return Un1


@njit(cache=True)
def euler_implict(Un, z):
  den = [1,0] - prod([1,0],z)
  Un1 = prod(div([1,0],den),Un)
  return Un1


@njit
def euler_implict_numba(Un, z):
    one = np.array([1.0, 0.0])
    denominator = complex_sub(one, z)
    factor = complex_div(one, denominator)   # 1 / (1 - z)
    Un1 = complex_prod(Un, factor)
    return Un1         # Un * (1 / (1 - z))



# @njit
# def euler_implict_numba(Un, z):
#     one_zero = np.array([1.0, 0.0])
#     den = complex_sub(one_zero, z)
#     Un1 = complex_div(Un, den)
#     return Un1



def trapezio(Un, z):
  num = [1,0] + z/2
  den = [1,0] - z/2
  Un1 = prod(div(num,den),Un)
  return Un1


# @njit
# def AB1(Un, z):
#   Un1 = Un + complex_prod(z,Un)
#   return  np.array(Un1)

@njit
def AB1(Un, z):
    result = Un + complex_prod(z, Un)
    return np.array([result[0], result[1]])  # Retorna sempre array


# @njit
# def AB2(Un1, Un, z):
#     den = [2, 0]
#     p = complex_prod([3,0], z)
#     num1 = [2 + p[0], 0 + p[1]]
#     parcela1 = complex_div(num1, den)
#     parcela2 = complex_div(z, den)
#     Un2 = complex_prod(parcela1, Un1)
#     Un2 = [Un2[0] - complex_prod(parcela2, Un)[0], Un2[1] - complex_prod(parcela2, Un)[1]]
#     return np.array(Un2)

@njit
def AB2(Un1, Un, z):
    den = np.array([2.0, 0.0])
    p = complex_prod(np.array([3.0, 0.0]), z)
    num1 = np.array([2.0 + p[0], 0.0 + p[1]])
    parcela1 = complex_div(num1, den)
    parcela2 = complex_div(z, den)
    Un2 = complex_prod(parcela1, Un1)
    result = np.array([Un2[0] - complex_prod(parcela2, Un)[0],
                       Un2[1] - complex_prod(parcela2, Un)[1]])
    return result



@njit
def AB3(Un2, Un1, Un, z):
    """
    Adams-Bashforth de 3ª ordem
    Fórmula: U^{n+3} = U^{n+2} + h/12 [23f(U^{n+2}) - 16f(U^{n+1}) + 5f(U^n)]
    Para f(U) = z*U
    """
    h = 1.0  # passo fixo para análise de estabilidade
    den = np.array([12.0, 0.0])  # denominador 12

    # Termos do numerador
    term1 = complex_prod(np.array([23.0 * h, 0.0]), z)  # 23h*z
    term2 = complex_prod(np.array([-16.0 * h, 0.0]), z)  # -16h*z
    term3 = complex_prod(np.array([5.0 * h, 0.0]), z)  # 5h*z

    # Coeficientes para cada U
    coef_Un2 = np.array([1.0, 0.0])  # coeficiente de U^{n+2}
    coef_Un2 = coef_Un2 + complex_div(term1, den)

    coef_Un1 = complex_div(term2, den)  # coeficiente de U^{n+1}
    coef_Un = complex_div(term3, den)  # coeficiente de U^n

    # Calcular U^{n+3}
    Un3 = complex_prod(coef_Un2, Un2)
    Un3 = Un3 + complex_prod(coef_Un1, Un1)
    Un3 = Un3 + complex_prod(coef_Un, Un)

    return Un3


@njit
def AB4(Un3, Un2, Un1, Un, z):
    """
    Adams-Bashforth de 4ª ordem
    U^{n+4} = U^{n+3} + h/24 [55f(U^{n+3}) - 59f(U^{n+2}) + 37f(U^{n+1}) - 9f(U^n)]
    Para f(U) = z*U
    """
    h = 1.0
    denom = 24.0

    # Coeficientes
    c1 = 55.0 * h / denom
    c2 = -59.0 * h / denom
    c3 = 37.0 * h / denom
    c4 = -9.0 * h / denom

    # Calcular termos
    term1 = complex_prod(np.array([1.0 + c1, 0.0]), z)
    term2 = complex_prod(np.array([c2, 0.0]), z)
    term3 = complex_prod(np.array([c3, 0.0]), z)
    term4 = complex_prod(np.array([c4, 0.0]), z)

    # Combinar termos
    Un4 = complex_prod(term1, Un3)
    Un4 = Un4 + complex_prod(term2, Un2)
    Un4 = Un4 + complex_prod(term3, Un1)
    Un4 = Un4 + complex_prod(term4, Un)

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


def AM2_correct(Un1, Un, z):
    """
    Adams-Moulton 2-step CORRETO
    U^{n+2} = U^{n+1} + h/12 [-z*U^n + 8z*U^{n+1} + 5z*U^{n+2}]
    """
    h = 1.0

    # Coeficiente do lado esquerdo: [1 - (5h*z)/12]
    den = [1, 0]
    term_den = prod([-5 * h / 12, 0], z)
    den = [den[0] + term_den[0], den[1] + term_den[1]]

    # Termos do lado direito
    # Coeficiente de U^{n+1}: [1 + (8h*z)/12]
    num1 = [1, 0]
    term_num1 = prod([8 * h / 12, 0], z)
    num1 = [num1[0] + term_num1[0], num1[1] + term_num1[1]]

    # Coeficiente de U^n: [(-h*z)/12]
    num2 = prod([-h / 12, 0], z)

    # Calcular U^{n+2}
    term_part1 = div(num1, den)
    term_part2 = div(num2, den)

    Un2 = prod(term_part1, Un1)
    Un2 = [Un2[0] + prod(term_part2, Un)[0], Un2[1] + prod(term_part2, Un)[1]]

    return Un2

def AM2(Un1, Un, z):
    den = [1, 0] - div(prod([5,0], z), [12, 0])
    num1 = [1, 0] + div(prod([8,0], z), [12, 0])
    num2 = div(z, [12, 0])

    Un2 = prod(div(num1, den), Un1) - prod(div(num2, den), Un)
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
@njit(cache=True)
def AM4(Un3, Un2, Un1, Un, z):
    """
    Adams-Moulton 4-step - Implementação CORRETA
    U^{n+4} = U^{n+3} + k/720 (-19f(U^n) + 106f(U^{n+1}) - 264f(U^{n+2}) + 646f(U^{n+3}) + 251f(U^{n+4}))
    Para f(U) = z*U
    """
    h = 1.0

    # Coeficiente do lado esquerdo: [1 - (251*h*z)/720]
    coef_left = np.array([1.0, 0.0])  # 1
    term_left = complex_prod(np.array([-251.0 * h / 720.0, 0.0]), z)  # -251*h*z/720
    coef_left = coef_left + term_left

    # Termos do lado direito:
    # Termo 1: U^{n+3}
    term1 = np.array([1.0, 0.0])  # coeficiente 1 para U^{n+3}

    # Termo 2: (646*h*z/720)U^{n+3}
    term2 = complex_prod(np.array([646.0 * h / 720.0, 0.0]), z)  # 646*h*z/720

    # Termo 3: (-264*h*z/720)U^{n+2}
    term3 = complex_prod(np.array([-264.0 * h / 720.0, 0.0]), z)  # -264*h*z/720

    # Termo 4: (106*h*z/720)U^{n+1}
    term4 = complex_prod(np.array([106.0 * h / 720.0, 0.0]), z)  # 106*h*z/720

    # Termo 5: (-19*h*z/720)U^n
    term5 = complex_prod(np.array([-19.0 * h / 720.0, 0.0]), z)  # -19*h*z/720

    # Combinar termos
    coef_Un3 = term1 + term2
    coef_Un3 = complex_div(coef_Un3, coef_left)

    coef_Un2 = complex_div(term3, coef_left)
    coef_Un1 = complex_div(term4, coef_left)
    coef_Un = complex_div(term5, coef_left)

    # Calcular U^{n+4}
    Un4 = complex_prod(coef_Un3, Un3)
    Un4 = Un4 + complex_prod(coef_Un2, Un2)
    Un4 = Un4 + complex_prod(coef_Un1, Un1)
    Un4 = Un4 + complex_prod(coef_Un, Un)

    return Un4


def AM5(Un4, Un3, Un2, Un1, Un, z):
    """
    Adams-Moulton 5-step - Versão compacta
    """
    h = 1.0

    # Denominador: [1 - (475*h*z)/1440]
    den = [1, 0]
    term_den = prod([-475 * h / 1440, 0], z)
    den = [den[0] + term_den[0], den[1] + term_den[1]]

    # Numerador para U^{n+4}: [1 + (1427*h*z)/1440]
    num_Un4 = [1, 0]
    term_num4 = prod([1427 * h / 1440, 0], z)
    num_Un4 = [num_Un4[0] + term_num4[0], num_Un4[1] + term_num4[1]]

    # Numerador para U^{n+3}: [(-798*h*z)/1440]
    num_Un3 = prod([-798 * h / 1440, 0], z)

    # Numerador para U^{n+2}: [(482*h*z)/1440]
    num_Un2 = prod([482 * h / 1440, 0], z)

    # Numerador para U^{n+1}: [(-173*h*z)/1440]
    num_Un1 = prod([-173 * h / 1440, 0], z)

    # Numerador para U^n: [(19*h*z)/1440]
    num_Un = prod([19 * h / 1440, 0], z)

    # Calcular coeficientes finais
    coef_Un4 = div(num_Un4, den)
    coef_Un3 = div(num_Un3, den)
    coef_Un2 = div(num_Un2, den)
    coef_Un1 = div(num_Un1, den)
    coef_Un = div(num_Un, den)

    # Calcular U^{n+5}
    Un5 = prod(coef_Un4, Un4)
    Un5 = [Un5[0] + prod(coef_Un3, Un3)[0], Un5[1] + prod(coef_Un3, Un3)[1]]
    Un5 = [Un5[0] + prod(coef_Un2, Un2)[0], Un5[1] + prod(coef_Un2, Un2)[1]]
    Un5 = [Un5[0] + prod(coef_Un1, Un1)[0], Un5[1] + prod(coef_Un1, Un1)[1]]
    Un5 = [Un5[0] + prod(coef_Un, Un)[0], Un5[1] + prod(coef_Un, Un)[1]]

    return Un5