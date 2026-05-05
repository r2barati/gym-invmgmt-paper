import pulp
import numpy as np
import logging

class StandaloneOracleOptimizer:
    def __init__(self, env, known_demand_scenario, planning_horizon=30, is_continuous=False):
        """
        Standalone Oracle: Solves MILP/LP using PERFECT future knowledge.
        
        Args:
            env: The instantiated environment object (CoreEnv).
            known_demand_scenario: Dictionary of {(src, dst): np.array_of_demands}
            planning_horizon: The length of the episode (default 30).
            is_continuous: If True, uses Continuous LP variables instead of Integer (MILP).
            
        Note:
            In scenarios with endogenous goodwill, the Oracle computes an upper bound
            based on a fixed demand trace and bypasses dynamic sentiment feedback.
            This represents a "Perfect Information + Perfect Service" relaxation.
        """
        self.env = env
        self.H = planning_horizon
        self.known_demand = known_demand_scenario
        self.is_continuous = is_continuous
        self.var_cat = 'Continuous' if is_continuous else 'Integer'

        # --- Extract Static Parameters ---
        self.lead_times = {}
        self.prices = {}
        self.pipe_costs = {}
        self.backlog_pen = {}
        self.holding_costs = {}
        self.op_costs = {}
        self.yields = {}
        self.capacities = {}

        # Edge Parameters
        for e in self.env.graph.edges():
            data = self.env.graph.edges[e]
            self.lead_times[e] = int(data.get('L', 0))
            self.prices[e] = data.get('p', 0.0)
            self.pipe_costs[e] = data.get('g', 0.0)
            self.backlog_pen[e] = data.get('b', 0.0)

        # Node Parameters
        for n in self.env.graph.nodes():
            data = self.env.graph.nodes[n]
            self.holding_costs[n] = data.get('h', 0.0)
            self.op_costs[n] = data.get('o', 0.0)
            self.yields[n] = data.get('v', 1.0)
            self.capacities[n] = data.get('C', 1e9)

        # Identify Main Nodes for logic filtering (exclude market=0, specific raw matterials if needed)
        self.main_nodes = sorted([n for n in self.env.graph.nodes() if n not in self.env.network.market and n not in self.env.network.rawmat])

    def solve_full_horizon(self):
        """
        Solves the ENTIRE horizon in one shot at t=0.
        This ensures the global optimum is found without re-planning noise.
        """
        # Create LP Problem
        prob = pulp.LpProblem("Global_Oracle_Optimization", pulp.LpMaximize)
        T_range = range(self.H)

        # --- Variables ---
        flow_vars = pulp.LpVariable.dicts("Flow",
                                          ((t, u, v) for t in T_range for (u,v) in self.env.network.reorder_links),
                                          lowBound=0, cat=self.var_cat)

        sales_vars = pulp.LpVariable.dicts("Sales",
                                           ((t, u, v) for t in T_range for (u,v) in self.env.network.retail_links),
                                           lowBound=0, cat=self.var_cat)

        inv_vars = pulp.LpVariable.dicts("Inv",
                                         ((t, n) for t in T_range for n in self.main_nodes),
                                         lowBound=0, cat='Continuous')

        backlog_vars = pulp.LpVariable.dicts("Backlog",
                                             ((t, u, v) for t in T_range for (u,v) in self.env.network.retail_links),
                                             lowBound=0, cat='Continuous')

        has_fixed_costs = any(self.env.graph.edges[e].get('K', 0.0) > 0 for e in self.env.network.reorder_links)
        if has_fixed_costs:
            setup_vars = pulp.LpVariable.dicts("Setup",
                                               ((t, u, v) for t in T_range for (u,v) in self.env.network.reorder_links),
                                               cat='Binary')

        # --- Constraints ---
        for t in T_range:
            if has_fixed_costs:
                for (u, v) in self.env.network.reorder_links:
                    cap = self.env.graph.nodes.get(u, {}).get('C', 1e6)
                    prob += flow_vars[(t, u, v)] <= cap * setup_vars[(t, u, v)]


            # A. Inventory Balance
            for n in self.main_nodes:
                # 1. Incoming (Arrivals)
                incoming_qty = 0
                for k in self.env.graph.predecessors(n):
                    L = self.lead_times.get((k, n), 0)
                    if t - L >= 0:
                        incoming_qty += flow_vars[(t - L, k, n)]

                # 2. Outgoing (Shipments)
                outgoing_qty = 0
                for k in self.env.graph.successors(n):
                    if (n, k) in self.env.network.retail_links:
                        outgoing_qty += sales_vars[(t, n, k)]
                    elif (n, k) in self.env.network.reorder_links:
                        outgoing_qty += flow_vars[(t, n, k)]

                # 3. Balance Equation
                if t == 0:
                    prev_inv = self.env.graph.nodes[n].get('I0', 0)
                else:
                    prev_inv = inv_vars[(t-1, n)]

                nu = self.yields.get(n, 1.0)

                if n in self.env.network.retail:
                     prob += inv_vars[(t, n)] == prev_inv + incoming_qty - outgoing_qty
                else:
                     prob += inv_vars[(t, n)] == prev_inv + incoming_qty - (1.0/nu * outgoing_qty)

                # 4. Supply Constraint (Simulator Time-Step Alignment)
                if n in self.env.network.retail:
                    # Retailers satisfy market demand AFTER arrivals are processed
                    # (Matches CoreEnv._STEP Phase 3: Demand Realization)
                    prob += outgoing_qty <= prev_inv + incoming_qty
                else:
                    # Manufacturers and Distributors ship downstream BEFORE arrivals are processed
                    # (Matches CoreEnv._STEP Phase 1: min(request, available_cap, v * available_inv))
                    # Note: available_inv = self.X[t], evaluated before incoming_total is added.
                    prob += (1.0/nu * outgoing_qty) <= prev_inv

                # 5. Capacity Constraints
                if n in self.env.network.factory:
                     out_reqs = pulp.lpSum([flow_vars[(t, n, k)] for k in self.env.graph.successors(n) if (n,k) in self.env.network.reorder_links])
                     prob += out_reqs <= self.capacities.get(n, 1e9)

            # B. Demand Logic
            for (r, m) in self.env.network.retail_links:
                true_demand = self.known_demand[(r,m)][t]

                if self.env.backlog:
                    # Backlog mode: unfulfilled demand carries forward
                    if t == 0:
                        prev_backlog = 0 
                    else:
                        prev_backlog = backlog_vars[(t-1, r, m)]

                    prob += sales_vars[(t, r, m)] <= true_demand + prev_backlog
                    prob += backlog_vars[(t, r, m)] == prev_backlog + true_demand - sales_vars[(t, r, m)]
                else:
                    # Lost-sales mode: unfulfilled demand is lost (not carried forward)
                    # Sales bounded by current-period demand only
                    prob += sales_vars[(t, r, m)] <= true_demand
                    # Track unfulfilled demand for penalty calculation (U[t] = D - sold)
                    prob += backlog_vars[(t, r, m)] == true_demand - sales_vars[(t, r, m)]

        # --- Objective Function ---
        total_profit = 0
        discount_factor = self.env.alpha

        for t in T_range:
            period_profit = 0
            
            # Revenue & Backlog
            for (r, m) in self.env.network.retail_links:
                p = self.prices.get((r, m), 0)
                b = self.backlog_pen.get((r, m), 0)
                period_profit += p * sales_vars[(t, r, m)] - b * backlog_vars[(t, r, m)]

            # Purchasing, Op Cost, Pipeline Cost 
            for (src, dst) in self.env.network.reorder_links:
                p = self.prices.get((src, dst), 0)
                g = self.pipe_costs.get((src, dst), 0)
                L = self.lead_times.get((src, dst), 0)
                K = self.env.graph.edges[(src, dst)].get('K', 0.0)

                flow = flow_vars[(t, src, dst)]

                is_src_main = src in self.main_nodes
                is_dst_main = dst in self.main_nodes

                if is_src_main: period_profit += p * flow
                if is_dst_main: period_profit -= p * flow
                if has_fixed_costs and K > 0:
                    period_profit -= K * setup_vars[(t, src, dst)]

                # Op Cost (Factory)
                if src in self.env.network.factory:
                    period_profit -= (self.op_costs.get(src, 0)/self.yields.get(src, 1.0)) * flow

            # Holding Costs (Inventory)
            for n in self.main_nodes:
                period_profit -= self.holding_costs.get(n, 0) * inv_vars[(t, n)]

            total_profit += (discount_factor ** t) * period_profit

        # --- Pipeline Holding Costs (g × Y) ---
        for t in T_range:
            for (src, dst) in self.env.network.reorder_links:
                g = self.pipe_costs.get((src, dst), 0)
                L = self.lead_times.get((src, dst), 0)
                is_dst_main = dst in self.main_nodes
                if is_dst_main and L > 0 and g > 0:
                    flow = flow_vars[(t, src, dst)]
                    for k in range(L):
                        abs_hold_t = t + k
                        if abs_hold_t < self.H:
                            total_profit -= (discount_factor ** abs_hold_t) * g * flow

        prob += total_profit

        # --- Solve ---
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != 1:
            logging.error("Solver Failed to find optimal solution.")
            return None

        # Extract All Actions
        all_actions = []
        for t in T_range:
            action_t = np.zeros(len(self.env.network.reorder_links))
            for i, edge in enumerate(self.env.network.reorder_links):
                val = pulp.value(flow_vars[(t, edge[0], edge[1])])
                action_t[i] = max(0, val)
            all_actions.append(action_t)

        return all_actions


# Import used by OracleAgent plan evaluation.
from gym_invmgmt.core_env import CoreEnv

class OracleAgent:
    """Wrapper that caches full Oracle offline-plan and yields actions sequentially for env benchmark loop."""
    def __init__(self, env, seed=42):
        self.env = env
        self._actions = self._generate_plan(self.env, seed)

    def _generate_plan(self, current_env, seed):
        # Use the exact kwargs the environment was constructed with to preserve shape/config consistency
        import copy
        env_kwargs = copy.deepcopy(current_env.env_kwargs)
        
        # If demand_config was None, env uses default dict, which we need to fetch
        demand_config = env_kwargs.get('demand_config')
        if demand_config is None:
            demand_config = current_env.demand_engine.config
            env_kwargs['demand_config'] = demand_config

        use_goodwill = demand_config.get('use_goodwill', False)
        
        if use_goodwill:
            # BidirectionalOracle for endogenous goodwill scenarios.
            # BidirectionalOracle explores both pessimistic (zero-action) and optimistic
            # no-shortage starting points for endogenous goodwill scenarios.
            from agents.baselines import BidirectionalOracle
            bidir = BidirectionalOracle(env_kwargs, planning_horizon=current_env.num_periods)
            actions, _ = bidir.solve(seed=seed)
        else:
            # For exogenous demand, StandaloneOracleOptimizer is exact and faster
            # than rolling-horizon re-planning because the problem is linear and fully known.
            from gym_invmgmt.core_env import CoreEnv
            probe_env = CoreEnv(**env_kwargs)
            probe_env.reset(seed=seed)
            for _ in range(current_env.num_periods):
                probe_env.step(np.zeros(probe_env.action_space.shape))
                
            full_demand_trace = {}
            for i, edge in enumerate(probe_env.network.retail_links):
                full_demand_trace[edge] = probe_env.D[:, i].copy()
                
            oracle = StandaloneOracleOptimizer(
                probe_env, 
                known_demand_scenario=full_demand_trace, 
                planning_horizon=current_env.num_periods,
                is_continuous=True
            )
            actions = oracle.solve_full_horizon()
            
            if actions is None:
                actions = [np.zeros(probe_env.action_space.shape)] * current_env.num_periods
                
        return actions
    def _evaluate_plan(self, env_kwargs, actions, seed):
        """Execute a plan and return total profit."""
        env = CoreEnv(**env_kwargs)
        env.reset(seed=seed)
        for t, a in enumerate(actions):
            env.step(a)
        return float(np.sum(env.P))

    def get_action(self, obs, t):
        if self._actions is None:
            raise RuntimeError("OracleAgent was instantiated incorrectly.")
            
        if t < len(self._actions):
            return self._actions[t]
        return np.zeros(len(self._actions[0])) if self._actions else np.zeros(1)
