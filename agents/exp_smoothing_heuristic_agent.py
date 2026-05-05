import numpy as np
from scipy.stats import norm
from agents.heuristic_utils import BaseHeuristicAgent

class ExpSmoothingHeuristicAgent(BaseHeuristicAgent):
    """
    Holt Linear Smoothing / Exponential Smoothing Agent.

    Forecasts mean demand using double exponential smoothing (level + trend),
    and computes safety stock using either realized forecast error or true variance.
    """

    def __init__(self, env, is_blind=False, alpha=0.3, beta=0.1, service_level=0.95):
        super().__init__(env, is_blind)
        self.alpha = alpha
        self.beta = beta
        self.z_alpha = norm.ppf(service_level)

        self.num_retail = len(self.env.network.retail_links)
        self.level = [None] * self.num_retail
        self.trend = [0.0] * self.num_retail
        self.mse = [0.0] * self.num_retail
        self.periods_observed = [0] * self.num_retail

        self.node_info = {}
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)

            if node in env.network.retail:
                b = 0
                for k in env.graph.successors(node):
                    if 'b' in env.graph.edges[(node, k)]:
                        b = max(b, env.graph.edges[(node, k)]['b'])
            else:
                downstream_prices = [env.graph.edges.get((node, k), {}).get('p', 0.0) for k in env.graph.successors(node)]
                incoming_prices = [env.graph.edges.get(e, {}).get('p', 0.0) for e in env.network.reorder_links if e[1] == node]
                        
                max_sell = max(downstream_prices) if downstream_prices else 0.0
                min_buy  = min(incoming_prices)   if incoming_prices else 0.0
                b = max(max_sell - min_buy, 0.5)

            cr = b / (b + h) if (b + h) > 0 else 0.5
            
            incoming_edges = [e for e in env.network.reorder_links if e[1] == node]
            max_L = max([env.graph.edges[e]['L'] for e in incoming_edges]) if incoming_edges else 0

            self.node_info[node] = {
                'cr': cr,
                'max_L': max_L
            }

    def _update_forecast(self, current_period):
        if current_period == 0:
            return

        D = self.env.D
        for idx, (r, m) in enumerate(self.env.network.retail_links):
            if hasattr(D, 'iloc'):
                recent_demand = D.iloc[current_period - 1, idx]
            else:
                recent_demand = D[current_period - 1, idx]

            if self.level[idx] is None:
                self.level[idx] = recent_demand
                self.trend[idx] = 0.0
                self.mse[idx] = recent_demand
                self.periods_observed[idx] = 1
            else:
                forecast = self.level[idx] + self.trend[idx]
                error = recent_demand - forecast

                new_level = self.alpha * recent_demand + (1 - self.alpha) * (self.level[idx] + self.trend[idx])
                new_trend = self.beta * (new_level - self.level[idx]) + (1 - self.beta) * self.trend[idx]

                self.level[idx] = new_level
                self.trend[idx] = new_trend

                # Update MSE for blind mode
                self.mse[idx] = self.mse[idx] + 0.1 * (error**2 - self.mse[idx])
                self.periods_observed[idx] += 1

    def _compute_targets(self, current_period):
        self._update_forecast(current_period)

        targets = {}
        for node in self.main_nodes:
            info = self.node_info[node]
            L = info['max_L']
            cr = info['cr']

            # Find all retail edges downstream of this node
            downstream_edges = []
            for idx, (r, m) in enumerate(self.env.network.retail_links):
                if r == node:
                    downstream_edges.append(idx)
            # If not a retail node, approximate by taking all retail edges. 
            # (In a true DP we'd trace exactly, but for global mean approximation this is standard).
            if not downstream_edges:
                downstream_edges = list(range(self.num_retail))

            node_mu_L = 0.0
            node_var_L = 0.0

            if self.level[0] is None:
                # Initialization phase
                node_mu, node_sigma = self.estimate_demand_stats(current_period)
                node_mu_L = node_mu * (L + 1)
                node_var_L = (node_sigma * np.sqrt(L + 1))**2
            else:
                for idx in downstream_edges:
                    link_mu = max(0, self.level[idx] + self.trend[idx])
                    if not self.is_blind:
                        # Informed: keep Holt's mean forecast, but use the
                        # simulator variance scale. Independent retail streams
                        # aggregate by summing variances, not by dividing std.
                        _, informed_sigma_L = self.estimate_lead_time_demand(current_period, L)
                        node_mu_L += link_mu * (L + 1)
                        node_var_L += informed_sigma_L**2
                    else:
                        node_mu_L += link_mu * (L + 1)
                        node_var_L += max(1.0, self.mse[idx]) * (L + 1)

            sigma_L = np.sqrt(node_var_L)
            
            if node_mu_L > 0:
                optimal_S = max(0, node_mu_L + norm.ppf(cr) * sigma_L)
            else:
                optimal_S = 0

            targets[node] = optimal_S

        return targets

    def get_action(self, obs, current_period):
        if current_period >= self.env.num_periods:
            return np.zeros(len(self.env.network.reorder_links))

        targets = self._compute_targets(current_period)
        actions = np.zeros(len(self.env.network.reorder_links))

        for node in self.main_nodes:
            inv_position, incoming_edges = self.inventory_position(node, current_period)
            target = targets.get(node, 0)

            order_needed = max(0, target - inv_position)
            self.allocate_order(order_needed, incoming_edges, actions)

        return self.clip_action(actions)
