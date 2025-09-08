import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from utils_tools import*
from bdf_methods import*
from rk_methods import*
from adams_bashforth_moulton_methods import*




def preditor_corrector_AB_AM(Un_history, z, preditor_order, corretor_order, n_correcoes=1):
    """
    Método Preditor-Corretor com Adams-Bashforth como preditor e Adams-Moulton como corretor

    Parameters:
    Un_history: lista dos últimos estados [U^{n}, U^{n+1}, ..., U^{n+k-1}]
    z: parâmetro complexo [real, imag]
    preditor_order: ordem do método Adams-Bashforth (1, 2, 3, 4)
    corretor_order: ordem do método Adams-Moulton (1, 2, 3, 4)
    n_correcoes: número de iterações do corretor

    Returns:
    Un_next: próximo estado [real, imag]
    """
    # PREDITOR: Adams-Bashforth
    if preditor_order == 1:
        # Euler Explícito
        Un_pred = euler_explicit(Un_history[-1], z)
    elif preditor_order == 2:
        Un_pred = AB2(Un_history[-1], Un_history[-2], z)
    elif preditor_order == 3:
        Un_pred = AB3(Un_history[-1], Un_history[-2], Un_history[-3], z)
    elif preditor_order == 4:
        Un_pred = AB4(Un_history[-1], Un_history[-2], Un_history[-3], Un_history[-4], z)
    else:
        raise ValueError("Ordem do preditor deve ser entre 1 e 4")

    # CORRETOR: Adams-Moulton (aplicado iterativamente)
    Un_corr = Un_pred

    for _ in range(n_correcoes):
        # Preparar histórico para o corretor (últimos estados + predição atual)
        corr_history = Un_history.copy()
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