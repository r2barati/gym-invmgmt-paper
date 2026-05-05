import numpy as np

class BaseHeuristicAgent:
    def __init__(self, env, is_blind=False):
        self.env = env
        self.is_blind = is_blind

        # Identify nodes that can hold inventory and place orders
        self.main_nodes = sorted(
            [n for n in env.graph.nodes()
             if n not in env.network.market and n not in env.network.rawmat]
        )
        
        # Determine blind prior mu from topology (avoid a fixed global demand prior)
        max_lt_overall = 0
        total_initial_inv = 0
        for e in env.network.reorder_links:
            max_lt_overall = max(max_lt_overall, env.graph.edges[e].get('L', 0))
        for n in env.network.retail:
            total_initial_inv += env.graph.nodes[n].get('I0', 100)
            
        retail_count = len(env.network.retail) if len(env.network.retail) > 0 else 1
        self.topology_prior_mu = (total_initial_inv / retail_count) / (max_lt_overall + 1)
        if self.topology_prior_mu <= 0:
            self.topology_prior_mu = 20.0 # Conservative fallback

    def estimate_demand_stats(self, current_period):
        """
        Estimate mean (mu) and standard deviation (sigma).
        Informed: Query demand_engine exactly.
        Blind: 5-period moving average of D, using topology prior at t=0.
        """
        if not self.is_blind:
            mu = self.env.demand_engine.get_current_mu(current_period)
            # Fetch true sigma based on engine's noise scale
            noise_scale = getattr(self.env.demand_engine, 'noise_scale', 1.0)
            sigma = np.sqrt(mu) * noise_scale if mu > 0 else 1.0
            return mu, max(sigma, 1.0)
        else:
            if current_period == 0:
                return self.topology_prior_mu, max(np.sqrt(self.topology_prior_mu), 1.0)
            
            lookback = min(5, current_period)
            D = self.env.D
            
            if hasattr(D, 'iloc'):
                recent = D.iloc[current_period - lookback:current_period].values
            else:
                recent = D[current_period - lookback:current_period]
                
            mu = np.mean(recent)
            if recent.shape[0] >= 3:
                # Use mean standard deviation across retailers
                sigma = np.mean(np.std(recent, axis=0, ddof=1))
            else:
                sigma = np.sqrt(mu)
                
            return mu, max(sigma, 1.0)

    def estimate_lead_time_demand(self, current_period, L):
        """
        Estimates total lead-time demand parameters (mu_L, sigma_L) for a horizon of length L+1.
        Informed: Sums true expected demand and variances along the expected lead time path.
        Blind: Multiplies the current point estimate by L+1.
        """
        if not self.is_blind and hasattr(self.env.demand_engine, 'get_current_mu'):
            mu_L = 0.0
            var_L = 0.0
            noise_scale = getattr(self.env.demand_engine, 'noise_scale', 1.0)
            for k in range(L + 1):
                t_forecast = current_period + k
                mu_t = self.env.demand_engine.get_current_mu(t_forecast)
                mu_L += mu_t
                sigma_t = np.sqrt(mu_t) * noise_scale if mu_t > 0 else 1.0
                var_L += sigma_t**2
            return mu_L, np.sqrt(var_L)
        else:
            mu, sigma = self.estimate_demand_stats(current_period)
            return mu * (L + 1), sigma * np.sqrt(L + 1)

    def inventory_position(self, node, current_period):
        """
        Calculates the true pipeline inventory position: OnHand + InTransit - Backlog.
        Uses the current pipeline stock Y[t] rather than future rows.
        """
        node_idx = self.env.network.node_map[node]

        if hasattr(self.env.X, 'loc'):
            on_hand = self.env.X.loc[current_period, node_idx]
        else:
            on_hand = self.env.X[current_period, node_idx]

        in_transit = 0
        incoming_edges = []
        for i, edge in enumerate(self.env.network.reorder_links):
            if edge[1] == node:
                # Y[t, i] represents ALL pipeline stock arriving at t+1 or later.
                if hasattr(self.env.Y, 'loc'):
                    in_transit += self.env.Y.loc[current_period, i]
                else:
                    in_transit += self.env.Y[current_period, i]
                incoming_edges.append((i, edge))

        backlog = 0
        if getattr(self.env, 'backlog', True) and node in self.env.network.retail:
            for k in self.env.graph.successors(node):
                if (node, k) in self.env.network.retail_map:
                    r_idx = self.env.network.retail_map[(node, k)]
                    if current_period > 0:
                        if hasattr(self.env.U, 'loc'):
                            backlog += self.env.U.loc[current_period - 1, r_idx]
                        else:
                            backlog += self.env.U[current_period - 1, r_idx]

        inv_pos = on_hand + in_transit - backlog
        return inv_pos, incoming_edges

    def allocate_order(self, order_needed, incoming_edges, actions):
        """
        Allocates orders across multiple suppliers using a balanced
        policy based on unit cost and lead time, avoiding naive equal splits.
        """
        if order_needed <= 0 or not incoming_edges:
            return
            
        if len(incoming_edges) == 1:
            actions[incoming_edges[0][0]] = order_needed
            return
            
        scores = []
        for idx, edge in incoming_edges:
            p = self.env.graph.edges[edge].get('p', 0.0)
            L = self.env.graph.edges[edge].get('L', 0)
            # Lower cost and lower lead time -> higher score
            score = 1.0 / (1.0 + p + 0.1 * L)
            scores.append(score)
            
        total_score = sum(scores)
        for (idx, edge), score in zip(incoming_edges, scores):
            actions[idx] += order_needed * (score / total_score)

    def clip_action(self, actions):
        """
        Clips actions to the environment bounds.
        """
        return np.clip(actions, self.env.action_space.low, self.env.action_space.high)
