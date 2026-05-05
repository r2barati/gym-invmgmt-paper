import numpy as np
from scipy.stats import norm
from agents.heuristic_utils import BaseHeuristicAgent

class NewsvendorHeuristicAgent(BaseHeuristicAgent):
    """
    Newsvendor / Critical-Ratio Base-Stock Heuristic Agent.

    For each inventory-holding node, the agent calculates an optimal
    base-stock target (S) using the Newsvendor critical ratio:
        CR = b / (b + h)
    where b = backlog/stockout penalty, h = holding cost.
    The target S is then set to satisfy lead-time demand with
    probability >= CR, using the Normal distribution approximation.
    """

    def __init__(self, env, is_blind=False):
        super().__init__(env, is_blind)

        # Pre-compute node costs
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

    def _compute_targets(self, current_period):
        targets = {}

        for node in self.main_nodes:
            info = self.node_info[node]
            mu_L, sigma_L = self.estimate_lead_time_demand(current_period, info['max_L'])

            if mu_L > 0:
                # Normal approximation for generalizability across topologies
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

        for node in self.main_nodes:
            inv_position, incoming_edges = self.inventory_position(node, current_period)
            target = targets.get(node, 0)
            
            order_needed = max(0, target - inv_position)
            self.allocate_order(order_needed, incoming_edges, actions)

        return self.clip_action(actions)

# Backward-compat alias
HeuristicAgent = NewsvendorHeuristicAgent
