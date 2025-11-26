from utils_tools import*


def runge_kutta2(Un, z):
    Ustar = Un + prod([0.5, 0],prod(Un, z))
    Un1 = Un + prod(Ustar, z)
    return Un1

def f(x, y):
    return x*y


def RK3(Un, z):
    """
    Método RK3 (Euler Melhorado) no formato similar ao BDF
    Para a equação teste: y' = z*y

    Parameters:
    Un: estado atual [real, imag]
    z: parâmetro complexo [real, imag]

    Returns:
    Un1: próximo estado [real, imag]
    """
    h = 1.0  # passo fixo para análise de estabilidade

    # Coeficientes do RK3 (Euler Melhorado)
    # A função de estabilidade é: G(z) = 1 + z + z²/2 + z³/4

    # Calcular o fator de amplificação
    z_h = prod(z, [h, 0])  # z*h

    z2 = prod(z_h, z_h)  # (z*h)²
    z3 = prod(z2, z_h)  # (z*h)³

    # Coeficientes do RK3
    term1 = [1, 0]
    term2 = div(z_h, [1, 0])
    term3 = div(z2, [2, 0])
    term4 = div(z3, [4, 0])

    # Somar todos os termos
    G = term1
    G = [G[0] + term2[0], G[1] + term2[1]]
    G = [G[0] + term3[0], G[1] + term3[1]]
    G = [G[0] + term4[0], G[1] + term4[1]]

    # Aplicar o fator de amplificação
    Un1 = prod(G, Un)

    return Un1



def RK4(Un, z):
    """
    Método de Runge-Kutta 4 no formato similar ao BDF
    Para a equação teste: y' = z*y

    Parameters:
    Un: estado atual [real, imag]
    z: parâmetro complexo [real, imag]

    Returns:
    Un1: próximo estado [real, imag]
    """
    h = 1.0  # passo fixo para análise de estabilidade

    # Coeficientes do RK4 para a equação y' = z*y
    # A função de estabilidade é: G(z) = 1 + z + z²/2 + z³/6 + z⁴/24

    # Calcular o fator de amplificação
    z_h = prod(z, [h, 0])  # z*h

    z2 = prod(z_h, z_h)  # (z*h)²
    z3 = prod(z2, z_h)  # (z*h)³
    z4 = prod(z3, z_h)  # (z*h)⁴

    # Coeficientes do RK4
    term1 = [1, 0]
    term2 = div(z_h, [1, 0])
    term3 = div(z2, [2, 0])
    term4 = div(z3, [6, 0])
    term5 = div(z4, [24, 0])

    # Somar todos os termos
    G = term1
    G = [G[0] + term2[0], G[1] + term2[1]]
    G = [G[0] + term3[0], G[1] + term3[1]]
    G = [G[0] + term4[0], G[1] + term4[1]]
    G = [G[0] + term5[0], G[1] + term5[1]]

    # Aplicar o fator de amplificação
    Un1 = prod(G, Un)

    return Un1