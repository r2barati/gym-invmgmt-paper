import json
import logging
import numpy as np

from agents.heuristic_utils import BaseHeuristicAgent
from agents.llm_utils import query_chat, get_node_type

logger = logging.getLogger(__name__)

class LLMPolicyAgent(BaseHeuristicAgent):
    """
    LLM-Policy-C Agent (LLM-as-Strategist).
    Queries the LLM once per episode to get strategy parameters
    (multipliers), then applies a deterministic bounded base-stock
    policy using those parameters for all periods.
    """
    def __init__(self, env, backend='local', model_name=None, api_key=None, verbose=False, is_blind=True):
        super().__init__(env, is_blind=is_blind)
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.verbose = verbose
        
        self.total_queries = 0
        self.parse_failures = 0
        
        self.policy_params = self._query_policy_parameters()

    def _build_system_prompt(self):
        nodes = len(self.main_nodes)
        links = len(self.env.network.reorder_links)
        
        return (
            f"Supply Chain Manager. Network has {nodes} nodes and {links} links.\n"
            f"Choose conservative inventory policy parameters for this supply-chain episode.\n\n"
            f"Return ONLY a JSON object with these exact keys:\n"
            f"demand_multiplier: number in [0.5, 2.0]\n"
            f"safety_z: number in [0.0, 3.0]\n"
            f"order_cap_fraction: number in [0.05, 1.0]\n"
            f"retail_multiplier: number in [0.2, 2.0]\n"
            f"distribution_multiplier: number in [0.2, 2.0]\n"
            f"factory_multiplier: number in [0.2, 2.0]\n\n"
            f"No explanation. No markdown."
        )

    def _query_policy_parameters(self):
        system_msg = self._build_system_prompt()
        messages = [{"role": "system", "content": system_msg}]
        
        if self.verbose:
            print(f"\n[LLM-Policy-C] Prompt: {system_msg}")
            
        response = query_chat(
            messages,
            backend=self.backend,
            api_key=self.api_key,
            model_name=self.model_name
        )
        
        if self.verbose:
            print(f"[LLM-Policy-C] Response: {response}")
            
        self.total_queries += 1
        
        default_params = {
            "demand_multiplier": 1.0,
            "safety_z": 1.0,
            "order_cap_fraction": 0.35,
            "retail_multiplier": 1.0,
            "distribution_multiplier": 0.8,
            "factory_multiplier": 0.6,
        }
        
        try:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # Update defaults with valid parsed keys
                parsed_any = False
                for k in default_params:
                    if k in data and isinstance(data[k], (int, float)):
                        default_params[k] = float(data[k])
                        parsed_any = True
                
                if parsed_any:
                    return default_params
        except Exception as e:
            logger.warning(f"LLM-Policy-C parse error: {e}")
            
        self.parse_failures += 1
        return default_params

    @property
    def parse_failure_rate(self) -> float:
        if self.total_queries == 0:
            return 0.0
        return self.parse_failures / self.total_queries

    def get_action(self, obs, current_period):
        if current_period >= self.env.num_periods:
            return np.zeros(len(self.env.network.reorder_links))

        actions = np.zeros(len(self.env.network.reorder_links))
        
        for node in self.main_nodes:
            node_type = get_node_type(node, self.env.network)
            if node_type == "Retailer":
                tier_mult = self.policy_params["retail_multiplier"]
            elif node_type == "Distributor":
                tier_mult = self.policy_params["distribution_multiplier"]
            else:
                tier_mult = self.policy_params["factory_multiplier"]
                
            inv_position, inc_edges = self.inventory_position(node, current_period)
            
            if not inc_edges:
                continue
                
            max_L = max([self.env.graph.edges[e]['L'] for _, e in inc_edges])
            
            mu_L, sigma_L = self.estimate_lead_time_demand(current_period, max_L)
            
            forecast = mu_L * self.policy_params["demand_multiplier"]
            safety = self.policy_params["safety_z"] * sigma_L
            target = forecast + safety
            
            order = max(target - inv_position, 0)
            order *= tier_mult
            
            # Cap the order using order_cap_fraction based on the environment's action bounds
            idx_list = [i for i, _ in inc_edges]
            max_high = max([self.env.action_space.high[i] for i in idx_list])
            order = np.clip(order, 0, max_high * self.policy_params["order_cap_fraction"])
                
            self.allocate_order(order, inc_edges, actions)
            
        return self.clip_action(actions)
