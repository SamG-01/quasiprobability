from typing import Callable

import numpy as np
from scipy.integrate import quad_vec

def wigner_quasiprobability(
    psi: Callable[[float, float], complex], x: float, p: float, t: float = 0
) -> float:
    """Computes the Wigner quasiprobability distribution from the wavefunction.

    Args:
        psi (Callable[[float, float], complex]): The wavefunciton psi(x, t).
        x (float): The position to evaluate W at.
        p (float): The momentum to evaluate W at.
        t (float): The time to evaluate W at.

    Returns:
        float: The value of W(x, p).
    """

    def integrand(y: float) -> float:
        left, right = np.conjugate(psi(x + y, t)), psi(x - y, t)
        exponential = np.exp(2j * p * y) / np.pi
        return left * right * exponential

    return np.real(quad_vec(integrand, -np.inf, np.inf)[0])
