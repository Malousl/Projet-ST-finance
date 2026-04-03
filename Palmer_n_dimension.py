import numpy as np
from scipy.optimize import linprog
from itertools import product
import pandas as pd


def prixnd(S0, K, T, r_eff, sigma, n, k, d):

    h = T / n
    R = (1 + r_eff) ** h

    S0    = np.asarray(S0,    dtype=float).ravel()
    sigma = np.asarray(sigma, dtype=float).ravel()
    k_arr = np.asarray(k,     dtype=float).ravel()

    if S0.size    == 1: S0    = np.full(d, S0[0])
    if sigma.size == 1: sigma = np.full(d, sigma[0])
    if k_arr.size == 1: k_arr = np.full(d, k_arr[0])

    u  = np.exp(sigma * np.sqrt(h))
    dv = 1.0 / u


    Xi   = {}  # xi_i  par nœud terminal
    Xi0  = {}  # xi0   par nœud terminal

    for inds in product(*[range(n + 1)] * d):
        inds_arr = np.array(inds)
        S_T = S0 * u**inds_arr * dv**(n - inds_arr)
        key = (n,) + inds
        if np.sum(S_T) > K:
            Xi[key]  = np.full(d, 1.0)
            Xi0[key] = -K
        else:
            Xi[key]  = np.zeros(d)
            Xi0[key] = 0.0

    # ----------------------------------------------------------------
    # Induction backward
    # ----------------------------------------------------------------
    for step in range(n - 1, -1, -1):
        for inds in product(*[range(step + 1)] * d):
            inds_arr = np.array(inds, dtype=float)
            S_now = S0 * u**inds_arr * dv**(step - inds_arr)

            moves = list(product([0, 1], repeat=d))  # 2^d scénarios
            M = len(moves)

            # Pour chaque scénario : prix futurs et valeurs futures du portefeuille
            S_next_list  = []
            xi_next_list = []
            xi0_next_list= []

            for mv in moves:
                mv_arr = np.array(mv)
                inds_fut = tuple(inds[i] + mv_arr[i] for i in range(d))  # INDICES FUTURS
                S_next = S0 * u**np.array(inds_fut) * dv**(step+1 - np.array(inds_fut))  # PRIX ABSOLUS
                key_fut = (step + 1,) + inds_fut
                S_next_list.append(S_next)
                xi_next_list.append(Xi[key_fut])
                xi0_next_list.append(Xi0[key_fut])
            # ----------------------------------------------------------------
            # Variables LP : [xi_0, ..., xi_{d-1}, xi0,
            #                  z_{0,0}, ..., z_{0,d-1},   <- scénario 0
            #                  z_{1,0}, ..., z_{1,d-1},   <- scénario 1
            #                  ...
            #                  z_{M-1,0}, ..., z_{M-1,d-1}]
            # Taille : d + 1 + M*d
            # ----------------------------------------------------------------
            n_xi  = d
            n_xi0 = 1
            n_z   = M * d
            n_var = n_xi + n_xi0 + n_z

            idx_xi  = np.arange(d)               # indices de xi
            idx_xi0 = d                           # indice de xi0
            def idx_z(m, i): return d + 1 + m*d + i

            # Objectif : min  sum_i xi_i * S_i_now + xi0
            c_obj = np.zeros(n_var)
            c_obj[idx_xi]  = S_now
            c_obj[idx_xi0] = 1.0

            A_ub = []
            b_ub = []

            for m, mv in enumerate(moves):
                S_next   = S_next_list[m]
                xi_fut   = xi_next_list[m]
                xi0_fut  = xi0_next_list[m]

                # Valeur future du portefeuille optimal (côté droit)
                V_fut = np.dot(xi_fut, S_next) + xi0_fut

                # --- Contrainte 1 : réplication ---
                # -sum_i xi_i * S_next_i - xi0 * R + k * sum_i z_{m,i} * S_next_i <= -V_fut
                row = np.zeros(n_var)
                row[idx_xi]  = -S_next
                row[idx_xi0] = -R
                for i in range(d):
                    row[idx_z(m, i)] = k_arr[i] * S_next[i]
                A_ub.append(row)
                b_ub.append(-V_fut)

                # --- Contraintes 2 & 3 : linéarisation |xi_i - xi_fut_i| <= z_{m,i} ---
                for i in range(d):
                    #  xi_i - z_{m,i} <= xi_fut_i
                    row2 = np.zeros(n_var)
                    row2[idx_xi[i]]  =  1.0
                    row2[idx_z(m,i)] = -1.0
                    A_ub.append(row2)
                    b_ub.append(xi_fut[i])

                    # -xi_i - z_{m,i} <= -xi_fut_i
                    row3 = np.zeros(n_var)
                    row3[idx_xi[i]]  = -1.0
                    row3[idx_z(m,i)] = -1.0
                    A_ub.append(row3)
                    b_ub.append(-xi_fut[i])

            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)

            # xi libre, xi0 libre, z >= 0
            bounds = [(-1e9, 1e9)] * (d + 1) + [(0, 1e9)] * (M * d)

            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if not res.success:
                raise RuntimeError(f"LP failed step={step}, inds={inds}: {res.message}")

            key = (step,) + inds
            Xi[key]  = res.x[idx_xi]
            Xi0[key] = res.x[idx_xi0]

    # Prix = -(xi^0 · S0 + xi0^0)  (short call -> opposé)
    key0 = (0,) + (0,) * d
    price = (np.dot(Xi[key0], S0) + Xi0[key0])
    return price



def boyle_vorst_long_call_1d(S0, K, T, r_eff, sigma, n, k):
    h = T / n
    R = (1 + r_eff) ** h
    u = np.exp(sigma * np.sqrt(h))
    d = 1 / u
    u_bar = u * (1 + k)
    d_bar = d * (1 - k)

    S_T   = np.array([S0 * u**j * d**(n-j) for j in range(n+1)])
    Delta = np.where(S_T > K, 1.0,  0.0)
    B     = np.where(S_T > K, -K,   0.0)

    for i in range(n - 1, -1, -1):
        S_i    = np.array([S0 * u**j * d**(i-j) for j in range(i+1)])
        V_up   = Delta[1:]  * S_i * u_bar + B[1:]
        V_down = Delta[:-1] * S_i * d_bar + B[:-1]
        Delta  = (V_up - V_down) / (S_i * (u_bar - d_bar))
        B      = (V_down - Delta * S_i * d_bar) / R

    return Delta[0] * S0 + B[0]


# def build_comparison_table(
#     S0=100., T=1., r_eff=0.10, sigma=0.2,
#     k_values=(0.0, 0.00125, 0.005, 0.02),
#     n_values=(6, 13, 52, 250),
#     K_values=(80., 90., 100., 110., 120.),
# ):
#     records = {}
#     for k in k_values:
#         print(f"  k = {k*100:g}%...")
#         for n in n_values:
#             for K in K_values:
#                 lc = boyle_vorst_long_call_1d(S0, K, T, r_eff, sigma, n, k)
#                 sc = prixnd(S0, K, T, r_eff, sigma, n, k, d=1)
#                 records[(k, K, n)] = (round(lc, 3), round(sc, 3), round(abs(lc - sc), 4))

#     rows = []
#     for k in k_values:
#         for K in K_values:
#             row = {"k": k, "K": int(K)}
#             for n in n_values:
#                 lc, sc, ec = records[(k, K, n)]
#                 row[f"BV_{n}"]    = lc
#                 row[f"nD_{n}"]    = sc
#                 row[f"ecart_{n}"] = ec
#             rows.append(row)

#     return pd.DataFrame(rows).set_index(["k", "K"])


# def print_comparison_table(df, n_values=(6, 13, 52, 250)):
#     """
#     Structure :
#                     n=6              n=13             n=52            n=250
#     K        BV     nD   |Δ|   BV     nD   |Δ|   BV     nD   |Δ|   BV     nD   |Δ|
#     """
#     cw = 8   # largeur colonne valeur
#     ew = 7   # largeur colonne écart

#     # Ligne 1 : groupes n
#     header1 = f"{'K':>6}"
#     for n in n_values:
#         header1 += f"{'n = ' + str(n):^{2*cw+ew+2}}"
#     print(header1)

#     # Ligne 2 : BV / nD / |Δ| par groupe
#     header2 = f"{'':>6}"
#     for _ in n_values:
#         header2 += f"{'BV':>{cw}}{'nD':>{cw}}{'|Δ|':>{ew}}  "
#     print(header2)

#     sep = "-" * len(header2)
#     print(sep)

#     for k, grp in df.groupby(level="k"):
#         label = "k = 0%" if k == 0 else f"k = {k*100:g}%"
#         pad   = (len(header2) - len(label)) // 2
#         print(f"\n{' ' * pad}{label}")

#         for (_, K), row in grp.iterrows():
#             line = f"{K:>6}"
#             for n in n_values:
#                 bv = row[f"BV_{n}"]
#                 nd = row[f"nD_{n}"]
#                 ec = row[f"ecart_{n}"]
#                 line += f"{bv:>{cw}.3f}{nd:>{cw}.3f}{ec:>{ew}.4f}  "
#             print(line)

#     print(sep)


# # --- Lancement ---
# print("Calcul en cours...")
# df_cmp = build_comparison_table(r_eff=0.10)
# print_comparison_table(df_cmp)

# # Export CSV
# df_cmp.to_csv("comparison_BV_vs_nD.csv")
# print("\nSauvegardé : comparison_BV_vs_nD.csv")



def build_dimension_tables(
    dims=(2, 3),                    # dimensions à tester
    S0_values=(90., 100., 110.),    # S0 pour actif 1 et 2
    k_values=(0.0, 0.005, 0.01, 0.02),
    K_values=(90., 100., 110.),     # strikes
    n_values=(6, 13, 26),           # pas de temps (d=3 trop lent sinon)
    T=1.0, r_eff=0.10, sigma=0.2
):
    """Génère tableaux pour d=2 et d=3"""
    
    results = {}
    
    for d in dims:
        print(f"\n{'='*60}")
        print(f"TABLEAUX DIMENSION {3}")
        print(f"{'='*60}")
        
        # Paramètres S0 pour d dimensions
        S0_grid = np.full(d, S0_values[0])
        
        records = []
        for S0_1 in S0_values:
            S0_grid[0] = S0_1  # Premier actif varie
            
            print(f"\nS0_1 = {S0_1}, k varie...")
            
            for k in k_values:
                k_arr = np.full(d, k)
                
                for K in K_values:
                    for n in n_values:
                        try:
                            price = prixnd(
                                S0=S0_grid, K=K, T=T, 
                                r_eff=r_eff, sigma=sigma, 
                                n=n, k=k_arr, d=d
                            )
                            records.append({
                                'd': d,
                                'S0_1': S0_1,
                                'S0_2': S0_grid[1] if d >= 2 else np.nan,
                                'k': k,
                                'K': K,
                                'n': n,
                                'price': round(price, 4)
                            })
                            print(f"  d={d}, S0=[{S0_1:.0f}", end="")
                            if d >= 2: print(f",{S0_grid[1]:.0f}", end="")
                            print(f"], k={k:.3f}, K={K}, n={n} → {price:.4f}")
                            
                        except Exception as e:
                            print(f"  ERREUR d={d}, S0_1={S0_1}, k={k}, K={K}, n={n}: {e}")
                            records.append({
                                'd': d, 'S0_1': S0_1, 'S0_2': S0_grid[1] if d >= 2 else np.nan,
                                'k': k, 'K': K, 'n': n, 'price': np.nan
                            })
        
        results[d] = pd.DataFrame(records)
    
    return results

def print_dimension_tables(results):
    """Affiche les tableaux formatés"""
    
    for d, df in results.items():
        print(f"\n{'='*80}")
        print(f"    RÉSULTATS SUPER-RÉPLICATION SHORT CALL - DIMENSION {d}")
        print(f"{'='*80}")
        
        # Pivot table par k et K
        for k in sorted(df['k'].unique()):
            print(f"\n  k = {k*100:.2f}%")
            sub_df = df[df['k'] == k].copy()
            
            pivot = sub_df.pivot_table(
                values='price', 
                index='K', 
                columns=['S0_1', 'n'],
                aggfunc='first'
            ).round(4)
            
            print(pivot.to_string(float_format='%.4f'))
        
        # Stats globales
        print(f"\n  📊 Stats d={d}:")
        print(f"     Prix min: {df['price'].min():.4f}")
        print(f"     Prix max: {df['price'].max():.4f}")
        print(f"     Prix moyen: {df['price'].mean():.4f}")
        print(f"     Temps total: {df['price'].count()} calculs OK")

# === LANCEMENT ===
if __name__ == "__main__":
    print("🚀 Calcul des tableaux super-réplication d=2 et d=3...")
    
    tables = build_dimension_tables(
        dims=(2, 3),
        S0_values=(90., 100., 110.),
        k_values=(0.0, 0.005, 0.01, 0.02),
        K_values=(90., 100., 110.),
        n_values=(6, 13, 26)  # d=3 trop lent avec n=52
    )
    
    # Affichage formaté
    print_dimension_tables(tables)
    
    # Export CSV
    for d, df in tables.items():
        df.to_csv(f"super_replication_d{d}_results.csv", index=False)
        print(f"\n💾 Exporté: super_replication_d{d}_results.csv")