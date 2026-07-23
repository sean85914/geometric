import numpy as np


def solve_trig_linear_equation(A, B, C):
    # Asin(theta)+Bcos(theta)=C
    K = np.sqrt(A**2 + B**2)
    if np.isclose(K, 0, atol=1e-5):
        raise ValueError('A and B are both zero')
    if abs(C / K) > 1 + 1e-5:
        raise ValueError('No real solution for theta')
    v = np.clip(C / K, -1.0, 1.0)
    phi = np.arctan2(B, A)
    asin = np.arcsin(v)
    return np.array([
        asin - phi,
        np.pi - asin - phi
    ])
