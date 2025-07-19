from typing import Callable

import numpy as np
import scipy.constants as C
from scipy.integrate import quad


def wigner_quasiprobability(
    psi: Callable[[float, float], complex], x: float, p: float, t: float = 0
) -> float:
    """Computes the Wigner quasiprobability distribution from the wavefunction.

    Args:
        psi (Callable[[float, float], complex]): The wavefunciton psi(x, t).
        x (float): The position to evaluate W at.
        p (float): The momentum to evaluate W at.
        t (float, optional): The time to evaluate t at. Defaults to 0.

    Returns:
        float: The value of W(x, p; t).
    """

    def integrand(y: float, t: float) -> float:
        left, right = np.conjugate(psi(x + y, t)), psi(x - y, t)
        exponential = np.exp(2j * p * y / C.hbar) / (np.pi * C.hbar)
        return left * right * exponential

    return quad(integrand, -np.inf, np.inf)
