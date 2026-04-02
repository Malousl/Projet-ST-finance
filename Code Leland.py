import numpy as np
from scipy.stats import norm
 
 
def leland_number(k, sigma, dt):
    return k * np.sqrt(2.0 / (np.pi * dt)) / sigma
 
 
def adjusted_volatility(sigma, Le, seller=True):
    sigma_hat_sq = sigma**2 * (1 + Le) if seller else sigma**2 * (1 - Le)
    if sigma_hat_sq <= 0:
        raise ValueError(f"Volatilité ajustée négative (Le={Le:.4f}). Réduire k ou augmenter dt.")
    return np.sqrt(sigma_hat_sq)
 
 
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
 
 
def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1
 
 
def leland_price(S, K, T, r, sigma, k, dt, option_type="call"):
    Le = leland_number(k, sigma, dt)
 
    sigma_ask = adjusted_volatility(sigma, Le, seller=True)
    sigma_bid = adjusted_volatility(sigma, Le, seller=False)
 
    return {
        "prix_ask":           round(black_scholes(S, K, T, r, sigma_ask, option_type), 6),
        "prix_bid":           round(black_scholes(S, K, T, r, sigma_bid, option_type), 6),
        "prix_black_scholes": round(black_scholes(S, K, T, r, sigma,     option_type), 6),
        "spread_bid_ask":     round(black_scholes(S, K, T, r, sigma_ask, option_type)
                                  - black_scholes(S, K, T, r, sigma_bid, option_type), 6),
        "Le":                 round(Le, 6),
        "sigma_ask":          round(sigma_ask, 6),
        "sigma_bid":          round(sigma_bid, 6),
    }
 
 
def simulate_leland_replication(S0, K, T, r, sigma, k, N, seller=True, option_type="call", seed=42):
    np.random.seed(seed)
    dt = T / N
    Le = leland_number(k, sigma, dt)
 
    sigma_hat = adjusted_volatility(sigma, Le, seller=seller)
 
    Z = np.random.standard_normal(N)
    S = np.zeros(N + 1)
    S[0] = S0
    for i in range(N):
        S[i + 1] = S[i] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[i])
 
    delta_prev = black_scholes_delta(S[0], K, T, r, sigma_hat, option_type)
    portfolio_value = black_scholes(S[0], K, T, r, sigma_hat, option_type)
    tc_total = abs(delta_prev) * S[0] * k
 
    for i in range(1, N):
        tau = T - i * dt
        if tau <= 0:
            break
        delta_new = black_scholes_delta(S[i], K, tau, r, sigma_hat, option_type)
        tc = abs(delta_new - delta_prev) * S[i] * k
        portfolio_value = portfolio_value * np.exp(r * dt) + (delta_new - delta_prev) * S[i] - tc
        tc_total += tc
        delta_prev = delta_new
 
    payoff = max(S[-1] - K, 0) if option_type == "call" else max(K - S[-1], 0)
 
    return {
        "S_terminal":               round(S[-1], 4),
        "payoff":                   round(payoff, 4),
        "valeur_portefeuille":      round(portfolio_value, 4),
        "erreur_réplication":       round(portfolio_value - payoff, 4),
        "coûts_transaction_totaux": round(tc_total, 4),
    }