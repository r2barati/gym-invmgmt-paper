"""
per_link_wrapper.py — Per-Link Feature Extraction Wrapper

Restructures the environment's observation into per-link features
for the Shared MLP Residual RL architecture.

Each reorder link gets 14 features (topology-invariant), enabling
the same MLP weights to work across different network topologies.
"""

import numpy as np
import gymnasium as gym

from agents.heuristic_agent import HeuristicAgent


class PerLinkFeatureWrapper(gym.ObservationWrapper):
    """
    Observation wrapper that extracts per-link features for each reorder link.

    Output observation: flat vector of shape (n_links × FEATURES_PER_LINK,)

    Per-link features (14 total):
      0: purchaser_on_hand    — inventory at purchaser node
      1: link_in_transit      — pipeline inventory for this link
      2: lead_time            — lead time L for this link
      3: inv_position         — purchaser inventory position
      4: lt_target            — Newsvendor target: μ × (L+1)
      5: gap                  — lt_target - inv_position
      6: heuristic_action     — heuristic's proposed order for this link
      7: backlog              — purchaser backlog (retail only)
      8: demand_velocity      — recent demand estimate (global)
      9: norm_time            — t / T (global)
     10: sentiment            — goodwill sentiment (global)
     11: is_retail            — binary: purchaser is retailer
     12: is_factory           — binary: supplier is factory
     13: holding_cost         — holding cost at purchaser
    """

    FEATURES_PER_LINK = 14

    def __init__(self, env, heuristic_agent=None, is_blind=False):
        super().__init__(env)
        self.is_blind = is_blind

        net = self.unwrapped.network
        self.n_links = len(net.reorder_links)

        # Pre-compute static per-link properties
        self._link_info = []
        for i, (supplier, purchaser) in enumerate(net.reorder_links):
            L = net.graph.edges[(supplier, purchaser)].get('L', 0)
            h = net.graph.nodes[purchaser].get('h', 0.0)
            is_retail = 1.0 if purchaser in net.retail else 0.0
            is_factory = 1.0 if supplier in net.factory else 0.0

            # Retail link index for backlog lookup
            retail_idx = None
            if purchaser in net.retail:
                for k in net.graph.successors(purchaser):
                    if (purchaser, k) in net.retail_map:
                        retail_idx = net.retail_map[(purchaser, k)]
                        break

            # All incoming edges to purchaser (for total in-transit)
            incoming_indices = [
                j for j, e in enumerate(net.reorder_links) if e[1] == purchaser
            ]

            self._link_info.append({
                'supplier': supplier,
                'purchaser': purchaser,
                'purchaser_idx': net.node_map[purchaser],
                'link_idx': i,
                'lead_time': float(L),
                'holding_cost': float(h),
                'is_retail': is_retail,
                'is_factory': is_factory,
                'retail_idx': retail_idx,
                'incoming_indices': incoming_indices,
            })

        # Build heuristic agent if not provided
        if heuristic_agent is not None:
            self.heuristic_agent = heuristic_agent
        else:
            self.heuristic_agent = HeuristicAgent(self.env.unwrapped, is_blind=is_blind)

        # Redefine observation space
        obs_dim = self.n_links * self.FEATURES_PER_LINK
        self.observation_space = gym.spaces.Box(
            low=np.full(obs_dim, -np.inf, dtype=np.float64),
            high=np.full(obs_dim, np.inf, dtype=np.float64),
            dtype=np.float64,
        )

    def _estimate_mu(self, t):
        """Estimate current demand mean."""
        core = self.unwrapped
        if not self.is_blind:
            return core.demand_engine.get_current_mu(t)
        else:
            if t == 0:
                return 20.0
            lookback = min(5, t)
            recent = core.D[t - lookback:t]
            return float(np.mean(recent)) if recent.size > 0 else 20.0

    def observation(self, obs):
        """Extract per-link features from current environment state."""
        # Use unwrapped to safely access core env attributes through wrapper chains
        env = self.unwrapped
        t = env.period
        net = env.network

        # --- Global features (shared across all links) ---
        mu = self._estimate_mu(t)

        # Demand velocity
        if t > 0:
            lookback = min(3, t)
            demand_vel = float(np.mean(env.D[t - lookback:t]))
        else:
            demand_vel = mu

        norm_time = t / max(env.num_periods, 1)
        sentiment = getattr(env.demand_engine, 'sentiment', 1.0)

        # --- Heuristic base actions ---
        env._update_state()
        heuristic_actions = self.heuristic_agent.get_action(env.state, t)

        # Convert dict to array if needed
        if isinstance(heuristic_actions, dict):
            heuristic_arr = np.zeros(self.n_links)
            for edge, val in heuristic_actions.items():
                if edge in net.reorder_map:
                    heuristic_arr[net.reorder_map[edge]] = val
        else:
            heuristic_arr = np.array(heuristic_actions)

        # --- Per-link feature extraction ---
        features = np.zeros(self.n_links * self.FEATURES_PER_LINK, dtype=np.float64)

        for i, info in enumerate(self._link_info):
            offset = i * self.FEATURES_PER_LINK

            # Purchaser on-hand inventory
            on_hand = env.X[t, info['purchaser_idx']]

            # In-transit for this specific link
            link_in_transit = env.Y[t, info['link_idx']] if t < env.num_periods else 0.0

            # Total in-transit to purchaser (for inventory position)
            total_in_transit = 0.0
            for j in info['incoming_indices']:
                total_in_transit += env.Y[t, j] if t < env.num_periods else 0.0

            # Backlog at purchaser. In lost-sales mode, U records prior lost
            # demand for KPI/cost accounting, not standing backlog owed today.
            backlog = 0.0
            if getattr(env, 'backlog', True) and info['retail_idx'] is not None and t > 0:
                backlog = env.U[t - 1, info['retail_idx']]

            # Inventory position
            inv_position = on_hand + total_in_transit - backlog

            # Lead-time target
            L = info['lead_time']
            lt_target = mu * (L + 1)

            # Gap
            gap = lt_target - inv_position

            # Heuristic base action for this link
            h_action = heuristic_arr[i]

            # Pack features
            features[offset + 0] = on_hand
            features[offset + 1] = link_in_transit
            features[offset + 2] = L
            features[offset + 3] = inv_position
            features[offset + 4] = lt_target
            features[offset + 5] = gap
            features[offset + 6] = h_action
            features[offset + 7] = backlog
            features[offset + 8] = demand_vel
            features[offset + 9] = norm_time
            features[offset + 10] = sentiment
            features[offset + 11] = info['is_retail']
            features[offset + 12] = info['is_factory']
            features[offset + 13] = info['holding_cost']

        return features
