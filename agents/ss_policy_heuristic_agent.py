import numpy as np
from scipy.stats import norm
from agents.heuristic_utils import BaseHeuristicAgent

class SSPolicyHeuristicAgent(BaseHeuristicAgent):
    """
    (s, S) Reorder-Point / Order-Up-To Policy Agent.

    Calculation:
      s = μ_L + z_{α} × σ_L      (safety stock reorder point)
      S = s + EOQ                  (order-up-to level)
      EOQ = sqrt(2 × μ × K / h)   (Economic Order Quantity)
    """

    def __init__(self, env, is_blind=False, service_level=0.95):
        super().__init__(env, is_blind)
        self.service_level = service_level
        self.z_alpha = norm.ppf(service_level)

        self.node_info = {}
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)

            incoming_edges = [e for e in env.network.reorder_links if e[1] == node]
            max_L = max([env.graph.edges[e]['L'] for e in incoming_edges]) if incoming_edges else 0
            
            # Extract fixed ordering cost from incoming edges
            K = 0.0
            if incoming_edges:
                # Average fixed cost across incoming supply edges
                K = np.mean([env.graph.edges[e].get('K', 0.0) for e in incoming_edges])
            
            if K == 0.0:
                # If K is completely 0, (s,S) should ideally collapse to a base-stock policy.
                # But to preserve the EOQ logic for testing, we can use a small calibrated virtual cost
                # ONLY if it's explicitly 0 everywhere.
                if node in env.network.factory:
                    K = env.graph.nodes[node].get('o', 0.01) * 10
                else:
                    K = h * 10

            self.node_info[node] = {
                'h': h,
                'K': K,
                'max_L': max_L
            }

    def _compute_ss_levels(self, current_period):
        levels = {}

        for node in self.main_nodes:
            info = self.node_info[node]
            L = info['max_L']
            h = info['h']
            K = info['K']

            # Lead-time demand statistics
            mu_L, sigma_L = self.estimate_lead_time_demand(current_period, L)
            # Per-period demand for EOQ formula
            mu, _ = self.estimate_demand_stats(current_period)

            s = mu_L + self.z_alpha * sigma_L

            if h > 0 and mu > 0:
                eoq = np.sqrt(2 * mu * K / h)
            else:
                eoq = mu_L

            S = s + eoq
            levels[node] = {'s': s, 'S': S}

        return levels

    def get_action(self, obs, current_period):
        if current_period >= self.env.num_periods:
            return np.zeros(len(self.env.network.reorder_links))

        levels = self._compute_ss_levels(current_period)
        actions = np.zeros(len(self.env.network.reorder_links))

        for node in self.main_nodes:
            inv_position, incoming_edges = self.inventory_position(node, current_period)

            s = levels[node]['s']
            S = levels[node]['S']

            if inv_position < s:
                order_needed = max(0, S - inv_position)
            else:
                order_needed = 0

            self.allocate_order(order_needed, incoming_edges, actions)

        return self.clip_action(actions)
