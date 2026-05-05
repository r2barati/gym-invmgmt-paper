"""
llm_invagent_central_agent.py — LLM-InvAgent-C: Staged Centralized LLM Agent

Implements a centralized InvAgent-inspired staged-prompt baseline.
Each period, the agent makes separate LLM calls per supply chain stage
(retail → distributor → factory), passing downstream order signals
upstream — mimicking InvAgent's multi-agent decision flow within a
single centralized controller.

This is a richer alternative to the zero-shot ``LLMZeroShotAgent``.
Neither agent implements the full decentralized multi-agent InvAgent
protocol (AutoGen-based, one conversational agent per node).

Key differences from LLM-ZS:
  - Per-stage "card" with full InvAgent-style state (costs, lead times,
    capacity, arrival schedule, upstream backlog)
  - Staged downstream→upstream loop: retail orders become demand signals
    for distributor decisions, etc.
  - Multiple LLM calls per period (one per stage tier)

References:
  - InvAgent: A Tool for Inventory Management with LLM-Based Multi-Agent
    Systems (Liu, Zefang et al., arXiv:2407.11384)
"""

import json
import logging
import numpy as np

from agents.llm_utils import (
    query_chat,
    parse_orders_json,
    format_link_desc,
    get_node_type,
    get_node_costs,
    get_lead_times_for_node,
    get_arriving_deliveries,
)

logger = logging.getLogger(__name__)


class CentralInvAgent:
    """
    Centralized InvAgent-inspired staged-prompt LLM agent.

    Each period, iterates downstream-to-upstream:
      1. Retail nodes: decide reorder quantities based on demand/backlog
      2. Distributor nodes: decide based on retail orders as demand signal
      3. Factory nodes: decide based on distributor orders as demand signal

    Parameters
    ----------
    env : CoreEnv
        The raw environment instance.
    backend : str
        'local' for local GGUF, 'openai' for API, 'mock' for deterministic tests.
    model_name : str or None
        For 'openai': model name (e.g. 'gpt-4').
    api_key : str or None
        API key for 'openai' backend.
    verbose : bool
        Print prompts and responses.
    """

    def __init__(self, env, backend: str = 'local', model_name: str = None,
                 api_key: str = None, verbose: bool = False):
        self.env = env
        self.backend = backend
        self.verbose = verbose
        self.api_key = api_key
        self.model_name = model_name

        self.reorder_links = env.network.reorder_links
        self.network = env.network

        # Organize nodes by tier (downstream first)
        self._build_tier_map()

        # Tracking
        self.parse_failures = 0
        self.total_queries = 0

    def _build_tier_map(self):
        """Group nodes into tiers: retail → distributor → factory."""
        self.tiers = []  # List of (tier_name, [nodes], [link_indices])

        # Tier 1: Retail nodes
        retail_nodes = sorted(self.network.retail)
        retail_link_idxs = []
        for i, (src, dst) in enumerate(self.reorder_links):
            if dst in retail_nodes:
                retail_link_idxs.append(i)
        if retail_nodes:
            self.tiers.append(("Retailer", retail_nodes, retail_link_idxs))

        # Tier 2: Distributor nodes
        distrib_nodes = sorted(self.network.distrib)
        distrib_link_idxs = []
        for i, (src, dst) in enumerate(self.reorder_links):
            if dst in distrib_nodes:
                distrib_link_idxs.append(i)
        if distrib_nodes:
            self.tiers.append(("Distributor", distrib_nodes, distrib_link_idxs))

        # Tier 3: Factory nodes
        factory_nodes = sorted(self.network.factory)
        factory_link_idxs = []
        for i, (src, dst) in enumerate(self.reorder_links):
            if dst in factory_nodes:
                factory_link_idxs.append(i)
        if factory_nodes:
            self.tiers.append(("Factory", factory_nodes, factory_link_idxs))

    def _build_stage_card(self, node, period: int, downstream_orders: dict) -> str:
        """
        Build an InvAgent-style stage card for a single node.

        Contains all decision-critical information:
          - Identity and type
          - On-hand inventory
          - Current backlog (U[t-1])
          - Pipeline stock per incoming link (Y[t, i])
          - Lead times per link
          - Arriving deliveries by delay slot
          - Holding cost, backlog penalty, sale price
          - Capacity and conversion (factories)
          - Recent demand history (retail)
          - Downstream order signal (from previous tier)
        """
        idx = self.network.node_map[node]
        node_type = get_node_type(node, self.network)
        costs = get_node_costs(node, self.env)
        lead_times = get_lead_times_for_node(node, self.env)
        arrivals = get_arriving_deliveries(node, self.env, period)

        card = {
            "Stage": f"{node_type} {node}",
            "On_Hand_Inventory": round(float(self.env.X[period, idx]), 1),
        }

        # Lead times
        if lead_times:
            card["Lead_Times"] = lead_times

        # Pipeline stock per incoming link
        pipeline = {}
        for i, (src, dst) in enumerate(self.reorder_links):
            if dst == node:
                pipeline[f"{src}→{dst}"] = round(float(self.env.Y[period, i]), 1)
        if pipeline:
            card["Pipeline_Stock"] = pipeline

        # Arriving deliveries
        if arrivals:
            card["Arriving_Deliveries"] = arrivals

        # Backlog (retail only, read from t-1)
        if node in self.network.retail:
            for k in self.env.graph.successors(node):
                if (node, k) in self.network.retail_map:
                    retail_idx = self.network.retail_map[(node, k)]
                    if period >= 1:
                        card["Current_Backlog"] = round(
                            float(self.env.U[period - 1, retail_idx]), 1
                        )
                    # Recent demand
                    start = max(0, period - 5)
                    demand_hist = [
                        round(float(d), 1)
                        for d in self.env.D[start:period, retail_idx]
                    ]
                    if demand_hist:
                        card["Recent_Demand_History"] = demand_hist
                    break

        # Costs
        if costs:
            card["Costs"] = costs

        # Downstream order signal (from previous tier decisions)
        if node in downstream_orders:
            card["Downstream_Order_Signal"] = round(downstream_orders[node], 1)

        return json.dumps(card, indent=2)

    def _build_tier_prompt(
        self,
        tier_name: str,
        nodes: list,
        link_indices: list[int],
        period: int,
        downstream_orders: dict,
    ) -> list[dict]:
        """Build chat messages for a tier-level decision."""
        # Build stage cards for all nodes in this tier
        cards = []
        for node in nodes:
            cards.append(self._build_stage_card(node, period, downstream_orders))

        # Describe the links this tier controls
        tier_links = [self.reorder_links[i] for i in link_indices]
        link_desc = format_link_desc(tier_links)
        n_links = len(tier_links)

        system_msg = (
            f"Supply Chain Manager for {tier_name} tier. "
            f"Maximize profit: balance holding costs vs stockout penalties.\n\n"
            f"Links: {link_desc} ({n_links} links)\n\n"
            f"Return ONLY a JSON object. No explanation. No markdown.\n"
            f'Format: {{"orders": [q0, ..., q{n_links - 1}]}}\n'
            f"Each value: non-negative number.\n\n"
            f'Example: {{"orders": [{", ".join(["15.0"] * n_links)}]}}'
        )

        user_msg = (
            f"Period {period}/{self.env.num_periods}\n\n"
            f"Stage Cards:\n" + "\n---\n".join(cards)
        )

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        """
        Generate an action via staged LLM inference (retail → distrib → factory).

        Returns an action array in [0, action_space.high], ready for env.step().
        """
        action = np.zeros(len(self.reorder_links))
        downstream_orders = {}  # node → total order quantity placed TO this node

        for tier_name, nodes, link_indices in self.tiers:
            if not link_indices:
                continue

            messages = self._build_tier_prompt(
                tier_name, nodes, link_indices, current_period, downstream_orders
            )

            if self.verbose:
                total_chars = sum(len(m['content']) for m in messages)
                print(f"\n[LLM-InvAgent-C] Period {current_period}, "
                      f"Tier {tier_name} ({total_chars} chars, {len(link_indices)} links)")

            response = query_chat(
                messages,
                backend=self.backend,
                api_key=self.api_key,
                model_name=self.model_name,
            )

            if self.verbose:
                print(f"[LLM-InvAgent-C] Response: {response[:200]}...")

            self.total_queries += 1
            tier_orders, success = parse_orders_json(response, len(link_indices))
            if not success:
                self.parse_failures += 1

            # Place orders for this tier's links
            for j, link_idx in enumerate(link_indices):
                action[link_idx] = tier_orders[j]

                # Record as downstream demand signal for the supplier node
                src, dst = self.reorder_links[link_idx]
                downstream_orders[src] = downstream_orders.get(src, 0.0) + tier_orders[j]

        # Clip to action space
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    @property
    def parse_failure_rate(self) -> float:
        """Fraction of queries that failed to parse."""
        if self.total_queries == 0:
            return 0.0
        return self.parse_failures / self.total_queries
