"""
llm_agent.py — LLM-ZS: Zero-shot LLM diagnostic agent for CoreEnv

A centralized zero-shot JSON-prompt LLM baseline. Issues one prompt
per period containing the full multi-echelon network state as a compact
JSON object and asks for all reorder quantities in a single response.

This is the lightweight LLM baseline. For the richer InvAgent-inspired
staged variant, see ``llm_invagent_central_agent.py``.

References:
  - InvAgent (Quan & Liu, arXiv:2407.11384) inspired the staged inventory prompt format used here.
"""

import json
import logging
import numpy as np

from agents.llm_utils import (
    query_chat,
    parse_orders_json,
    format_link_desc,
    get_node_type,
)

logger = logging.getLogger(__name__)


class LLMZeroShotAgent:
    """
    Zero-shot centralized LLM agent for supply chain inventory management.

    One LLM call per period. Encodes the full network state as a compact
    JSON dictionary and asks for all reorder link quantities in one response.

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

        # Identify main supply chain nodes (exclude market sinks and raw material sources)
        self.nodes = sorted(
            n for n in env.graph.nodes()
            if n not in env.network.market and n not in env.network.rawmat
        )
        self.reorder_links = env.network.reorder_links

        # Parse failure tracking
        self.parse_failures = 0
        self.total_queries = 0

    def format_state_prompt(self, period: int) -> str:
        """
        Encode the current environment state into a structured JSON dictionary.

        State parsing details:
          - Backlog reads U[period-1] (last completed period), not U[period]
          - Pipeline reads Y[period, i] (stock variable), not sum(Y[period:, i])
        """
        state = {"Period": period, "Total_Periods": self.env.num_periods, "Nodes": []}

        for node in self.nodes:
            idx = self.env.network.node_map[node]
            on_hand = float(self.env.X[period, idx])

            # Pipeline: read the stock variable at current time only (not sum of future)
            in_transit = 0.0
            incoming_links = [(i, e) for i, e in enumerate(self.reorder_links) if e[1] == node]
            for i, _ in incoming_links:
                in_transit += float(self.env.Y[period, i])

            # Demand history and backlog (for retail nodes)
            demand_history = []
            backlog = 0.0
            if node in self.env.network.retail:
                retail_idx = -1
                for k in self.env.graph.successors(node):
                    if (node, k) in self.env.network.retail_map:
                        retail_idx = self.env.network.retail_map[(node, k)]
                        break

                if retail_idx != -1:
                    start = max(0, period - 5)
                    demand_history = [round(float(d), 1) for d in self.env.D[start:period, retail_idx]]
                    # Fix: read backlog from PREVIOUS period (U is end-of-period)
                    if period >= 1:
                        backlog = float(self.env.U[period - 1, retail_idx])

            node_type = get_node_type(node, self.env.network)

            node_state = {
                "Node_ID": node,
                "Type": node_type,
                "On_Hand_Inventory": round(on_hand, 1),
                "In_Transit_Pipeline": round(in_transit, 1),
            }
            if demand_history:
                node_state["Recent_Demand_History"] = demand_history
            if backlog > 0:
                node_state["Current_Backlog"] = round(backlog, 1)

            state["Nodes"].append(node_state)

        return json.dumps(state, indent=2)

    def _build_system_prompt(self) -> str:
        """System prompt for zero-shot JSON-format ordering."""
        n_links = len(self.reorder_links)
        link_table = "\n".join(
            f"  link_{i}: {src}->{dst}"
            for i, (src, dst) in enumerate(self.reorder_links)
        )
        list_example = ", ".join(["10.0"] * n_links)
        dict_example = ", ".join(f'"link_{i}": 10.0' for i in range(n_links))
        return (
            f"You are a Supply Chain Operations Manager controlling a multi-echelon "
            f"inventory network. Maximize total profit by balancing holding costs "
            f"against backlog penalties.\n\n"
            f"NETWORK: {len(self.nodes)} nodes and exactly {n_links} reorder links.\n"
            f"Action index mapping:\n{link_table}\n\n"
            f"DECISION: Output one order quantity for every listed link. "
            f"The response is invalid unless it contains exactly {n_links} values.\n\n"
            f"Return ONLY a JSON object. No explanation. No markdown. No reasoning.\n"
            f'Preferred format: {{"orders_by_link": {{{dict_example}}}}}\n'
            f'Also accepted: {{"orders": [{list_example}]}}\n'
            f"Each value must be a non-negative number.\n\n"
            f"Do not omit any link and do not aggregate by node."
        )

    def get_action(self, obs: np.ndarray, current_period: int) -> np.ndarray:
        """
        Generate an action via LLM inference.

        Returns an action array in [0, action_space.high], ready for env.step().
        """
        state_text = self.format_state_prompt(current_period)
        system = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Current State:\n{state_text}"},
        ]

        if self.verbose:
            print(f"\n[LLM-ZS] Period {current_period} prompt ({len(system) + len(state_text)} chars)")

        response = query_chat(
            messages,
            backend=self.backend,
            api_key=self.api_key,
            model_name=self.model_name,
        )

        if self.verbose:
            print(f"[LLM-ZS] Response: {response[:200]}...")

        self.total_queries += 1
        n_links = len(self.reorder_links)
        action, success = parse_orders_json(response, n_links)
        if not success:
            repair_messages = messages + [
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": (
                        f"Invalid shape. Return ONLY JSON with exactly {n_links} "
                        f"numeric order quantities, one for each link_0 through "
                        f"link_{n_links - 1}. Use the orders_by_link format."
                    ),
                },
            ]
            repaired = query_chat(
                repair_messages,
                backend=self.backend,
                api_key=self.api_key,
                model_name=self.model_name,
            )
            self.total_queries += 1
            action, success = parse_orders_json(repaired, n_links)
        if not success:
            self.parse_failures += 1

        # Clip to action space
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    @property
    def parse_failure_rate(self) -> float:
        """Fraction of queries that failed to parse."""
        if self.total_queries == 0:
            return 0.0
        return self.parse_failures / self.total_queries


# Backward compatibility alias
LLMAgent = LLMZeroShotAgent
