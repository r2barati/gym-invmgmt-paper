"""
example_custom_agent.py — A minimal blind agent to copy when plugging into evaluate_custom.py.

Order-up-to policy: forecast each retail link's demand as a moving average of
realized demand, push that forecast upstream along the reorder links, and order
the gap between a base-stock target and each node's inventory position
(on hand + in transit - backlog). It uses only what the blind tier exposes.

  python benchmarks/evaluate_custom.py \\
      --policy benchmarks/example_custom_agent.py:MovingAverageBaseStock --tier blind
"""

import networkx as nx
import numpy as np


class MovingAverageBaseStock:
    def __init__(self, view, seed, window=5, z=1.0):
        self.view = view
        self.window = window
        self.z = z
        net = view.network
        g = view.graph
        self.n_retail = len(net.retail_links)

        # Reorder links feeding each stocking node: (action index, link)
        self.incoming = {}
        for idx, link in enumerate(net.reorder_links):
            self.incoming.setdefault(link[1], []).append((idx, link))

        # share[node] @ retail_demand = demand rate flowing through node.
        # Each node passes its demand to its suppliers in equal parts.
        share = {n: np.zeros(self.n_retail) for n in net.main_nodes}
        for i, (retailer, _) in enumerate(net.retail_links):
            share[retailer][i] = 1.0
        for node in reversed(list(nx.topological_sort(g))):
            if node not in share or node not in self.incoming:
                continue
            suppliers = [src for _, (src, _) in self.incoming[node] if src in share]
            for src in suppliers:
                share[src] += share[node] / len(suppliers)
        self.share = share

        self.lead_time = {n: max(g.edges[link]['L'] for _, link in links)
                          for n, links in self.incoming.items()}
        self.backlog_idx = {n: [i for i, (r, _) in enumerate(net.retail_links) if r == n]
                            for n in net.main_nodes}

        # Prior before any demand is observed (same idea as the blind heuristics):
        # initial retail inventory spread over the longest lead time.
        retail_i0 = np.mean([g.nodes[r].get('I0', 0) for r in net.retail]) if net.retail else 0
        prior = retail_i0 / (max(self.lead_time.values(), default=0) + 1)
        self.prior = prior if prior > 0 else 20.0

    def get_action(self, obs, t):
        view = self.view
        history = view.demand_history()
        if len(history):
            mu = history[-self.window:].mean(axis=0)
        else:
            mu = np.full(self.n_retail, self.prior)

        s = view.obs_slices
        on_hand = obs[s['inventory']]
        backlog = obs[s['backlog']]
        actions = np.zeros(view.action_space.shape)
        for pos, node in enumerate(view.network.main_nodes):
            links = self.incoming.get(node)
            if not links:
                continue
            horizon = self.lead_time[node] + 1
            demand = float(self.share[node] @ mu) * horizon
            target = demand + self.z * np.sqrt(max(demand, 0.0))
            in_transit = sum(obs[view.pipeline_slices[link]].sum()
                             for _, link in links if link in view.pipeline_slices)
            position = on_hand[pos] + in_transit - backlog[self.backlog_idx[node]].sum()
            order = max(0.0, target - position)
            for idx, _ in links:
                actions[idx] = order / len(links)
        return np.clip(actions, view.action_space.low, view.action_space.high)
