import numpy as np


def palmer_short_call(S0, K, T, r_eff, sigma, n, k):

    h = T / n
    R = (1 + r_eff) ** h
    u = np.exp(sigma * np.sqrt(h))
    d = 1.0 / u

    # Prix aux nœuds terminaux
    S_T = np.array([S0 * u**j * d**(n - j) for j in range(n + 1)])

    # Conditions terminales : short call
    Delta = np.where(S_T > K, -1.0, 0.0)
    B     = np.where(S_T > K,  K,   0.0)

    # Backward induction
    for i in range(n - 1, -1, -1):
        S_i = np.array([S0 * u**j * d**(i - j) for j in range(i + 1)])

        du = Delta[1:]   # Δ dans l'état up
        dd = Delta[:-1]  # Δ dans l'état down
        Bu = B[1:]
        Bd = B[:-1]

        # Constante C_eq (éq. 2.3 de Palmer, partie sans valeurs absolues)
        C_eq = (du * u - dd * d) / (u - d) + (Bu - Bd) / (S_i * (u - d))

        # Évaluation de f aux points charnières pour déterminer l'intervalle
        # f(Δ) = Δ - C_eq - k*(|Δu-Δ|*u - |Δd-Δ|*d) / (u-d)
        f_u = du - C_eq - k * ((0)         * u - np.abs(dd - du) * d) / (u - d)
        f_d = dd - C_eq - k * (np.abs(du - dd) * u - (0)         * d) / (u - d)

        Delta_new = np.zeros(i + 1)

        for j in range(i + 1):
            ceq = C_eq[j]
            fu  = f_u[j]
            fd  = f_d[j]
            duj = du[j]
            ddj = dd[j]

            if duj < ddj:   # cas typique du short call
                if fu >= 0:
                    # Δ ≤ Δu  →  pente 1+k
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u - ddj * d)) / (1 + k)
                elif fd >= 0:
                    # Δu ≤ Δ ≤ Δd  →  pente 1 - k(u+d)/(u-d)
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u + ddj * d)) / (1 - k * (u + d) / (u - d))
                else:
                    # Δ ≥ Δd  →  pente 1-k
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u - ddj * d)) / (1 - k)

            else:            # Δu ≥ Δd (long call, put, etc.)
                if fd >= 0:
                    # Δ ≤ Δd  →  pente 1+k
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u - ddj * d)) / (1 + k)
                elif fu >= 0:
                    # Δd ≤ Δ ≤ Δu  →  pente 1 + k(u+d)/(u-d)
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u + ddj * d)) / (1 + k * (u + d) / (u - d))
                else:
                    # Δ ≥ Δu  →  pente 1-k
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u - ddj * d)) / (1 - k)

        # Calcul de B via la condition d'autofinancement (état up)
        Delta_arr = Delta_new
        B_new = (S_i * u * (du - Delta_arr + k * np.abs(du - Delta_arr)) + Bu) / R

        Delta = Delta_new
        B     = B_new

    return -(Delta[0] * S0 + B[0])


def palmer_long_call(S0, K, T, r_eff, sigma, n, k):

    h = T / n
    R = (1 + r_eff) ** h
    u = np.exp(sigma * np.sqrt(h))
    d = 1.0 / u

    S_T   = np.array([S0 * u**j * d**(n - j) for j in range(n + 1)])
    Delta = np.where(S_T > K,  1.0, 0.0)
    B     = np.where(S_T > K, -K,   0.0)

    for i in range(n - 1, -1, -1):
        S_i = np.array([S0 * u**j * d**(i - j) for j in range(i + 1)])

        du = Delta[1:]
        dd = Delta[:-1]
        Bu = B[1:]
        Bd = B[:-1]

        C_eq = (du * u - dd * d) / (u - d) + (Bu - Bd) / (S_i * (u - d))

        f_u = du - C_eq - k * ((0)              * u - np.abs(dd - du) * d) / (u - d)
        f_d = dd - C_eq - k * (np.abs(du - dd)  * u - (0)             * d) / (u - d)

        Delta_new = np.zeros(i + 1)

        for j in range(i + 1):
            ceq = C_eq[j]
            fu  = f_u[j]
            fd  = f_d[j]
            duj = du[j]
            ddj = dd[j]

            if duj >= ddj:   # cas typique du long call
                if fd >= 0:
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u - ddj * d)) / (1 + k)
                elif fu >= 0:
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u + ddj * d)) / (1 + k * (u + d) / (u - d))
                else:
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u - ddj * d)) / (1 - k)
            else:
                if fu >= 0:
                    Delta_new[j] = (ceq + k / (u - d) * (duj * u - ddj * d)) / (1 + k)
                elif fd >= 0:
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u + ddj * d)) / (1 - k * (u + d) / (u - d))
                else:
                    Delta_new[j] = (ceq - k / (u - d) * (duj * u - ddj * d)) / (1 - k)

        B_new = (S_i * u * (du - Delta_new + k * np.abs(du - Delta_new)) + Bu) / R

        Delta = Delta_new
        B     = B_new

    return Delta[0] * S0 + B[0]


# ── Vérification sur les valeurs du tableau 2.1 de Palmer ──────────────────

print("Short call (Palmer 2001, Table 2.1) :")
print(f"{'X':>6} {'n':>5} {'k':>7}  {'C (Palmer)':>12}")
for X in [80, 90, 100, 110, 120]:
    for n in [6, 13, 52, 250]:
        for k in [0, 0.00125, 0.005, 0.02]:
            C = palmer_short_call(100, X, 1, 0.10, 0.20, n, k)
            print(f"{X:>6} {n:>5} {k:>7.5f}  {C:>12.3f}")
