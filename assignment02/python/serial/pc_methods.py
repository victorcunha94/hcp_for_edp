import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*


def preditor_corrector_AB_AM(Un1, Un, z, preditor_order=2, corretor_order=2, n_correcoes=1):
    """
    Método Preditor-Corretor com Adams-Bashforth como preditor e Adams-Moulton como corretor
    Formato compatível com BDF2: recebe Un1, Un e retorna Un2

    Parameters:
    Un1: estado U^{n+1} [real, imag]
    Un: estado U^n [real, imag]
    z: parâmetro complexo [real, imag]
    preditor_order: ordem do método Adams-Bashforth (2, 3, 4)
    corretor_order: ordem do método Adams-Moulton (1, 2, 3, 4)
    n_correcoes: número de iterações do corretor

    Returns:
    Un2: próximo estado U^{n+2} [real, imag]
    """
    # Criar histórico com os dois estados disponíveis
    history = [Un.copy(), Un1.copy()]

    # PREDITOR: Adams-Bashforth
    if preditor_order == 2:
        Un_pred = AB2(history[-1], history[-2], z)
    elif preditor_order == 3:
        # Para ordem 3, precisamos de mais um estado anterior
        # Usar Euler implícito para estimar U^{n-1}
        Un_prev = euler_implict(history[-2], z)
        history.insert(0, Un_prev)
        Un_pred = AB3(history[-1], history[-2], history[-3], z)
    elif preditor_order == 4:
        # Para ordem 4, precisamos de mais dois estados anteriores
        Un_prev1 = euler_implict(history[-2], z)
        Un_prev2 = euler_implict(Un_prev1, z)
        history.insert(0, Un_prev2)
        history.insert(0, Un_prev1)
        Un_pred = AB4(history[-1], history[-2], history[-3], history[-4], z)
    else:
        raise ValueError("Ordem do preditor deve ser 2, 3 ou 4")

    # CORRETOR: Adams-Moulton (aplicado iterativamente)
    Un_corr = Un_pred

    for _ in range(n_correcoes):
        # Preparar histórico para o corretor
        corr_history = history.copy()
        corr_history.append(Un_corr)

        if corretor_order == 1:
            Un_corr = AM1(corr_history[-1], z)
        elif corretor_order == 2:
            Un_corr = AM2(corr_history[-1], corr_history[-2], z)
        elif corretor_order == 3:
            Un_corr = AM3(corr_history[-1], corr_history[-2], corr_history[-3], z)
        elif corretor_order == 4:
            Un_corr = AM4(corr_history[-1], corr_history[-2], corr_history[-3], corr_history[-4], z)
        else:
            raise ValueError("Ordem do corretor deve ser entre 1 e 4")

    return Un_corr