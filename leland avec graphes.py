import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── paramètres de base ──────────────────────────────────────────────
S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

def leland_number(k, sigma, dt):
    return k * np.sqrt(2.0 / (np.pi * dt)) / sigma

def black_scholes(S, K, T, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def bs_delta(S, K, T, r, sig):
    d1 = (np.log(S/K) + (r + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    return norm.cdf(d1)

def sigma_hat(sigma, Le, seller=True):
    val = sigma**2 * (1 + Le) if seller else sigma**2 * (1 - Le)
    return np.sqrt(val) if val > 0 else np.nan

def simulate_replication(S0, K, T, r, sigma, k, N, seed=42):
    np.random.seed(seed)
    dt = T / N
    Le = leland_number(k, sigma, dt)
    sh = sigma_hat(sigma, Le, seller=True)
    if np.isnan(sh):
        return np.nan

    Z = np.random.standard_normal(N)
    S = np.zeros(N+1); S[0] = S0
    for i in range(N):
        S[i+1] = S[i] * np.exp((r - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*Z[i])

    delta_prev = bs_delta(S[0], K, T, r, sh)
    pf = black_scholes(S[0], K, T, r, sh)
    for i in range(1, N):
        tau = T - i*dt
        if tau <= 0: break
        d_new = bs_delta(S[i], K, tau, r, sh)
        tc = abs(d_new - delta_prev) * S[i] * k
        pf = pf*np.exp(r*dt) + (d_new - delta_prev)*S[i] - tc
        delta_prev = d_new

    payoff = max(S[-1] - K, 0)
    return pf - payoff

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Modèle de Leland — analyse des coûts de transaction", fontsize=14, fontweight='bold')

# ── Graphe 1 : prix ask / bid / BS en fonction de k ─────────────────
ax = axes[0, 0]
k_vals = np.linspace(0.001, 0.08, 100)
dt_fixed = 1/52  # révision hebdomadaire

ask_prices, bid_prices, bs_price = [], [], []
bs_ref = black_scholes(S, K, T, r, sigma)
for k in k_vals:
    Le = leland_number(k, sigma, dt_fixed)
    sh_ask = sigma_hat(sigma, Le, seller=True)
    sh_bid_val = sigma**2 * (1 - Le)
    ask_prices.append(black_scholes(S, K, T, r, sh_ask))
    bid_prices.append(black_scholes(S, K, T, r, np.sqrt(sh_bid_val)) if sh_bid_val > 0 else np.nan)

ax.plot(k_vals*100, ask_prices, label='Prix ask (vendeur)', color='tab:red')
ax.plot(k_vals*100, bid_prices, label='Prix bid (acheteur)', color='tab:blue')
ax.axhline(bs_ref, linestyle='--', color='gray', label='Black-Scholes (k=0)')
ax.fill_between(k_vals*100, bid_prices, ask_prices, alpha=0.12, color='purple', label='Spread bid-ask')
ax.set_xlabel('Taux de coût k (%)')
ax.set_ylabel('Prix de l\'option')
ax.set_title('Prix en fonction du taux de coût k\n(révision hebdomadaire, $\\Delta t = 1/52$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Graphe 2 : surcoût Z en fonction de Δt, pour plusieurs k ────────
ax = axes[0, 1]
dt_vals = np.linspace(1/252, 1/4, 300)  # de quotidien à trimestriel
bs_ref = black_scholes(S, K, T, r, sigma)

for k in [0.005, 0.01, 0.02, 0.04]:
    Z_vals = []
    for dt in dt_vals:
        Le = leland_number(k, sigma, dt)
        sh = sigma_hat(sigma, Le, seller=True)
        Z_vals.append(black_scholes(S, K, T, r, sh) - bs_ref)
    ax.plot(dt_vals * 252, Z_vals, label=f'k = {int(k*100)}%')

ax.set_xlabel('Fréquence de révision (nombre de jours entre chaque révision)')
ax.set_ylabel('Surcoût Z = Prix Leland − Prix BS')
ax.set_title('Surcoût Z en fonction de la fréquence\n(Proposition I : $Z \\propto 1/\\sqrt{\\Delta t}$)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
# repère pour montrer la divergence
ax.axvline(52, linestyle=':', color='gray', alpha=0.6, label='Hebdo')

# ── Graphe 3 : volatilité ajustée en fonction de Δt ─────────────────
ax = axes[1, 0]
dt_vals2 = np.linspace(1/252, 1/2, 400)

for k in [0.005, 0.01, 0.02, 0.04]:
    sh_vals = []
    for dt in dt_vals2:
        Le = leland_number(k, sigma, dt)
        sh_vals.append(sigma_hat(sigma, Le, seller=True))
    ax.plot(dt_vals2 * 252, sh_vals, label=f'k = {int(k*100)}%')

ax.axhline(sigma, linestyle='--', color='gray', label='$\\sigma$ (sans coûts)')
ax.set_xlabel('Fréquence de révision (nombre de pas par an)')
ax.set_ylabel('Volatilité ajustée $\\hat{\\sigma}$')
ax.set_title('Divergence de $\\hat{\\sigma}$ quand $\\Delta t \\to 0$\n(invalidation de Leland par Kabanov)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Graphe 4 : erreur de réplication en fonction de N ───────────────
ax = axes[1, 1]
N_vals = [10, 20, 30, 52, 75, 100, 150, 200, 260]

for k in [0.005, 0.01, 0.02]:
    errors = []
    for N in N_vals:
        err = simulate_replication(S, K, T, r, sigma, k, N, seed=0)
        errors.append(err if err is not None else np.nan)
    ax.plot(N_vals, errors, marker='o', markersize=4, label=f'k = {int(k*100)}%')

ax.axhline(0, linestyle='--', color='gray', alpha=0.6)
ax.set_xlabel('Nombre de pas de révision N')
ax.set_ylabel('Erreur de réplication (valeur portefeuille − payoff)')
ax.set_title('Erreur de réplication vs. fréquence\n(l\'erreur ne converge pas vers 0)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('leland_analyse.png', dpi=150, bbox_inches='tight')
plt.show()
