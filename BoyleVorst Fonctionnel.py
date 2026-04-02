import numpy as np
from scipy.stats import norm


def boyle_vorst_long_call(S0, K, T, r_eff, sigma, n, k):

    h     = T / n
    R     = (1 + r_eff) ** h      
    u     = np.exp(sigma * np.sqrt(h))
    d     = 1 / u
    u_bar = u * (1 + k)            
    d_bar = d * (1 - k)            

    # Prix du sous-jacent aux nœuds terminaux
    S_T = np.array([S0 * u**j * d**(n - j) for j in range(n + 1)])

    #  Δ = 1 action, B = -K en cash
    # Δ = 0,        B = 0
    Delta = np.where(S_T > K, 1.0, 0.0)
    B     = np.where(S_T > K, -K,  0.0)

    # on résout :
    #   Δ·S·ū + B·R = Δ₁·S·ū + B₁   =>  V_up
    #   Δ·S·d̄ + B·R = Δ₂·S·d̄ + B₂   =>  V_down
    # Soustraction :
    #   Δ = (V_up - V_down) / (S·(ū - d̄))
    #   B = (V_down - Δ·S·d̄) / R
    for i in range(n - 1, -1, -1):
        S_i = np.array([S0 * u**j * d**(i - j) for j in range(i + 1)])

        V_up   = Delta[1:] * S_i * u_bar + B[1:]
        V_down = Delta[:-1] * S_i * d_bar + B[:-1]

        Delta = (V_up - V_down) / (S_i * (u_bar - d_bar))
        B     = (V_down - Delta * S_i * d_bar) / R

    return Delta[0] * S0 + B[0]

print(boyle_vorst_long_call(100,80,1,0.1,0.2,52,0.02))



def boyle_vorst_short_call(S0, K, T, r_eff, sigma, n, k):
    h     = T / n
    R     = (1 + r_eff) ** h
    u     = np.exp(sigma * np.sqrt(h))
    d     = 1.0 / u
    u_bar = u * (1 - k)
    d_bar = d * (1 + k)

    # Condition de validité (eq. 20-21 du papier)
    if u * (1 - k) < R * (1 + k) or d * (1 + k) > R * (1 - k):
        return float('nan')

    S_T   = np.array([S0 * u**j * d**(n-j) for j in range(n+1)])

    # Portfolio à maturité : short call = -1 action + K cash si ITM
    Delta = np.where(S_T > K, -1.0, 0.0)
    B     = np.where(S_T > K,  K,   0.0)

    # Backward induction
    for i in range(n-1, -1, -1):
        S_i    = np.array([S0 * u**j * d**(i-j) for j in range(i+1)])
        V_up   = Delta[1:]  * S_i * u_bar + B[1:]
        V_down = Delta[:-1] * S_i * d_bar + B[:-1]
        Delta  = (V_up - V_down) / (S_i * (u_bar - d_bar))
        B      = (V_down - Delta * S_i * d_bar) / R

    return -(Delta[0] * S0 + B[0])

print(boyle_vorst_short_call(100,80,1,0.1,0.2,52,0.005))
