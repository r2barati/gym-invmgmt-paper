import numpy as np
import networkx as nx
from scipy.stats import norm
from agents.heuristic_utils import BaseHeuristicAgent

class EchelonApproxAgent(BaseHeuristicAgent):
    """
    Clark-Scarf-inspired Echelon Base-Stock Heuristic Agent.

    This is an approximation of the Clark-Scarf dynamic program.
    It computes an approximate echelon base-stock target from cumulative lead time
    and a critical ratio, rather than solving exact echelon holding costs via DP.

    Key Difference from Newsvendor:
      - Newsvendor sets target using local lead time.
      - EchelonApprox sets target using cumulative lead time to the end consumer.
    """

    def __init__(self, env, is_blind=False):
        super().__init__(env, is_blind)

        # Pre-compute echelon logic
        self.node_info = {}
        self.echelon_nodes = {}
        
        for node in self.main_nodes:
            h = env.graph.nodes[node].get('h', 0.1)

            # Cumulative downstream lead time
            cum_L = self._compute_cumulative_lead_time(node)
            
            # Local max incoming lead time
            incoming_edges = [e for e in env.network.reorder_links if e[1] == node]
            local_L = max([env.graph.edges[e]['L'] for e in incoming_edges]) if incoming_edges else 0

            # Echelon Holding Cost approximation
            # Approximate echelon holding-cost adjustment for divergent networks.
            downstream_h = 0
            for k in env.graph.successors(node):
                downstream_h += env.graph.nodes[k].get('h', 0.0)
                
            echelon_h = max(0.01, h - downstream_h)

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

            cr = b / (b + echelon_h) if (b + echelon_h) > 0 else 0.5

            self.node_info[node] = {
                'cr': cr,
                'cum_L': cum_L,
                'local_L': local_L
            }

            descendants = nx.descendants(env.graph, node)
            self.echelon_nodes[node] = [
                n for n in self.main_nodes
                if n == node or n in descendants
            ]

    def _compute_cumulative_lead_time(self, node):
        """Recursively compute maximum lead time to the market."""
        if node in self.env.network.retail:
            return 0
            
        max_downstream = 0
        for k in self.env.graph.successors(node):
            if (node, k) in self.env.network.reorder_links:
                L = self.env.graph.edges[(node, k)].get('L', 0)
                max_downstream = max(max_downstream, L + self._compute_cumulative_lead_time(k))
        return max_downstream

    def _compute_targets(self, current_period):
        targets = {}

        for node in self.main_nodes:
            info = self.node_info[node]
            # Echelon uses cumulative lead time + local lead time
            total_L = info['cum_L'] + info['local_L']
            
            mu_L, sigma_L = self.estimate_lead_time_demand(current_period, total_L)

            if mu_L > 0:
                optimal_S = max(0, mu_L + norm.ppf(info['cr']) * sigma_L)
            else:
                optimal_S = 0

            targets[node] = optimal_S

        return targets

    def get_action(self, obs, current_period):
        if current_period >= self.env.num_periods:
            return np.zeros(len(self.env.network.reorder_links))

        targets = self._compute_targets(current_period)
        actions = np.zeros(len(self.env.network.reorder_links))
        
        local_inv = {}
        incoming_by_node = {}
        for node in self.main_nodes:
            local_inv[node], incoming_by_node[node] = self.inventory_position(
                node, current_period)

        for node in self.main_nodes:
            echelon_inv = sum(local_inv[n] for n in self.echelon_nodes[node])
            target = targets.get(node, 0)
            order_needed = max(0, target - echelon_inv)
            
            self.allocate_order(order_needed, incoming_by_node[node], actions)

        return self.clip_action(actions)
