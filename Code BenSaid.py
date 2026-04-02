import numpy as np

class TransactionCostPricer:
    def __init__(self, S0, K, u, d, k, T, delta_steps=200):
        """
        Initialise les paramètres du modèle binomial avec coûts de transaction.
        
        S0: Prix initial du sous-jacent
        K: Prix d'exercice de l'option (Strike)
        u: Facteur de hausse
        d: Facteur de baisse
        k: Coût de transaction proportionnel
        T: Nombre de périodes
        delta_steps: Précision de la grille pour les quantités d'actions
        """
        self.S0 = S0
        self.K = K
        self.u = u
        self.d = d
        self.k = k
        self.T = T
        
        self.deltas = np.linspace(0.0, 1.0, delta_steps)
        
    def phi(self, y):
        """
        Fonction de coût de transaction phi(y)
        y: Quantité d'actions échangée (Delta_t - Delta_{t-1})
        """
        return np.where(y >= 0, y * (1 + self.k), y / (1 + self.k))

    def terminal_payoff(self, ST):
        """
        Payoff d'un Call Européen (règlement en espèces).
        """
        payoff = np.maximum(ST - self.K, 0)
        print(f"Payoff à la maturité pour S{self.T} = {ST}: {payoff}")
        return payoff

    def price(self):
        """
        Résout le modèle par récurrence arrière (Backward Induction).
        Retourne le coût de fabrication optimal à t=0.
        """
        # Générer l'arbre des prix finaux à t=T
        # states_S[j] correspond à j hausses et (T-j) baisses
        states_S = np.array([self.S0 * (self.u**j) * (self.d**(self.T - j)) for j in range(self.T + 1)])
        
        # Initialisation de la fonction de valeur à maturité Q_T(Delta_{T-1}, node)
        # Dimensions : (nombre de noeuds à t, nombre de deltas possibles)
        Q = np.zeros((self.T + 1, len(self.deltas)))
        
        # Remplissage à T : 
        # B_T = Payoff. Puis on ajoute le coût de liquidation si on exige une quantité finale cible
        # Pour un cash settlement explicite, on suppose que l'investisseur liquide son portefeuille
        for j in range(self.T + 1):
            ST = states_S[j]
            BT = self.terminal_payoff(ST)
            # On cherche à dominer la position (cash reçu >= Payoff)
            for i, delta_prev in enumerate(self.deltas):
                # Le coût final si on liquide toutes les actions héritées pour payer BT
                # L'équation du papier pour Q_T
                Q[j, i] = BT + ST * self.phi(-delta_prev)
                
        # Remontée dans l'arbre (Backward Induction)
        for t in range(self.T - 1, -1, -1):
            # Noeuds de prix à l'étape t
            current_S = np.array([self.S0 * (self.u**j) * (self.d**(t - j)) for j in range(t + 1)])
            new_Q = np.zeros((t + 1, len(self.deltas)))
            
            for j in range(t + 1):
                St = current_S[j]
                
                # Q_{t+1} pour les états up et down
                Q_up = Q[j + 1, :]
                Q_down = Q[j, :]
                
                # R_{t+1}(Delta_t) = max(Q_up, Q_down)
                R_t1 = np.maximum(Q_up, Q_down)
                
                # Résolution du programme Q_t(Delta_{t-1})
                for i, delta_prev in enumerate(self.deltas):
                    # Coût de transaction pour passer de delta_prev à chaque delta_new
                    delta_diffs = self.deltas - delta_prev
                    transaction_costs = St * self.phi(delta_diffs)
                    
                    # Fonction objectif à minimiser : R_{t+1}(Delta_t) + S_t * phi(Delta_t - Delta_{t-1})
                    objective = R_t1 + transaction_costs
                    
                    # Trouver le minimum
                    new_Q[j, i] = np.min(objective)
                    
            Q = new_Q
            
        # À t=0, l'investisseur commence avec Delta_{-1} = 0 action
        # Le prix est Q_0(0, S0)
        initial_delta_index = np.argmin(np.abs(self.deltas - 0.0))
        optimal_cost = Q[0, initial_delta_index]
        
        return optimal_cost

# --- Exemple d'utilisation basé sur les paramètres du papier ---
if __name__ == "__main__":
    # Paramètres de l'introduction du papier
    S0 = 100
    K = 100
    u = 1.3
    d = 0.9
    k = 0.20 # 20% de coûts de transaction
    T = 2    # Modèle à 2 périodes (t=0, 1, 2)
    
    pricer = TransactionCostPricer(S0, K, u, d, k, T, delta_steps=500)
    prix_dominant = pricer.price()
    
    print(f"Le coût de fabrication (borne supérieure) de l'option est estimé à : {prix_dominant:.2f}")
