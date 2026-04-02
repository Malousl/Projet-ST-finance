"""
Pricer d'options avec coûts de transaction proportionnels
Implémentation du modèle de Bensaid, Lesne, Pagès & Scheinkman (1992),
"Derivative Asset Pricing with Transaction Costs", Mathematical Finance.
"""

import numpy as np




class TransactionCostPricer:
    """
    Calcule P1(x), le "manufacturing cost" (borne supérieure du prix) d'un call
    européen avec cash settlement, dans le modèle binomial à coûts de transaction
    proportionnels de Bensaid et al. (1992).

    Paramètres
    ----------
    S0          : Prix initial du sous-jacent
    K           : Strike de l'option
    u           : Facteur de hausse  (u > 1)
    d           : Facteur de baisse  (0 < d < 1)
    k           : Taux de coût de transaction proportionnel (ex. 0.20 pour 20%)
    T           : Nombre de périodes
    delta_steps : Taille de la grille pour Delta dans [0, 1]
    """

    def __init__(self, S0, K, u, d, k, T, delta_steps=1000):
        self.S0 = S0
        self.K  = K
        self.u  = u
        self.d  = d
        self.k  = k
        self.T  = T
        self.deltas = np.linspace(0.0, 1.0, delta_steps)

    def _phi(self, y):
        """
        Fonction de cout de transaction phi(y) (papier, p. 67) :
          phi(y) = (1+k)*y   si y >= 0  (achat)
                 = y/(1+k)   si y < 0   (vente)
        y = Delta_new - Delta_old.  S*phi(y) = cash sorti du compte.
        """
        return np.where(y >= 0, y * (1.0 + self.k), y / (1.0 + self.k))

    def _terminal_Q(self):
        """
        Condition aux limites à t = T — Cash settlement (section 4.2).

        Le vendeur liquide sa position à T (Delta_T = 0) et verse le payoff cash.
        Formule du papier :
          Q_T(Delta_{T-1}, omega) = B_T(omega) + S_T(omega) * phi(-Delta_{T-1})
                                  = (S_T - K)^+ - S_T * Delta_{T-1} / (1+k)

        Retourne shape (T+1, n_deltas).
        Indice j = nœud "j hausses, T-j baisses".
        """
        n = len(self.deltas)
        Q = np.zeros((self.T + 1, n))
        for j in range(self.T + 1):
            ST = self.S0 * (self.u ** j) * (self.d ** (self.T - j))
            BT = max(ST - self.K, 0.0)
            Q[j, :] = BT + ST * self._phi(-self.deltas)
        return Q

    def price(self):
        """
        Résout le modèle par récurrence arrière et retourne P1(x).

        Récurrence (Theorem 3.1) :
          R_{t+1}(Delta_t) = max(Q_{t+1}(Delta_t, up), Q_{t+1}(Delta_t, down))
          Q_t(Delta_{t-1}) = min_{Delta_t} [R_{t+1}(Delta_t) + S_t*phi(Delta_t - Delta_{t-1})]

        Programme t=0 (CORRECTION du code original) :
          P1 = min_{Delta_0} [Delta_0 * S0 + max(Q1(Delta_0, up), Q1(Delta_0, down))]
          sans coûts de transaction à t=0 ("no transaction costs at origin", papier p.71)
        """
        deltas = self.deltas
        n      = len(deltas)

        Q = self._terminal_Q()   # shape (T+1, n)

        # --- Récurrence backward de t = T-1 jusqu'à t = 1 ---
        for t in range(self.T - 1, 0, -1):
            new_Q = np.zeros((t + 1, n))

            for j in range(t + 1):
                St = self.S0 * (self.u ** j) * (self.d ** (t - j))

                # Nœud j à t  →  fils up   = nœud j+1 à t+1
                #                 fils down = nœud j   à t+1
                R_next = np.maximum(Q[j + 1], Q[j])   # shape (n,)

                # min_{Delta_t} [R_next(Delta_t) + St * phi(Delta_t - Delta_{t-1})]
                # On itère sur Delta_t (indice l) et garde le min pour chaque Delta_{t-1}
                best = np.full(n, np.inf)
                for l in range(n):
                    c = R_next[l] + St * self._phi(deltas[l] - deltas)
                    np.minimum(best, c, out=best)

                new_Q[j, :] = best

            Q = new_Q   # shape (t+1, n)

        # --- Programme D_0 à t = 0 : PAS de coûts de transaction ---
        # Q a maintenant shape (2, n) = Q_1 pour nœuds down (j=0) et up (j=1)
        #
        # CORRECTION (BUG 1) : on minimise Delta_0*S0 + R1(Delta_0)
        # sans appliquer phi(Delta_0 - 0), contrairement au code original.
        R1         = np.maximum(Q[1], Q[0])       # R1(Delta_0) = max(Q1_up, Q1_down)
        total_cost = deltas * self.S0 + R1

        idx_opt        = np.argmin(total_cost)
        optimal_cost   = total_cost[idx_opt]
        optimal_delta0 = deltas[idx_opt]
        optimal_B0     = R1[idx_opt]

        print(f"  Delta_0 optimal : {optimal_delta0:.4f}")
        print(f"  B_0     optimal : {optimal_B0:.4f}")

        return optimal_cost


