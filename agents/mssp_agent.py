import pulp
import numpy as np
import logging
import time
from agents.or_utils import (
    add_fixed_costs,
    get_demand_prior,
    get_pipeline_arrivals,
    pipeline_holding_charge,
)

class RollingHorizonMSSPAgent:
    """
    Rolling-horizon MSSP with SAA demand scenarios and explicit non-anticipativity constraints.
    """
    def __init__(self, env, planning_horizon=30, branching_depth=3, is_continuous=True, is_blind=False, seed=None):
        self.env = env
        self.H = planning_horizon
        self.depth = branching_depth
        self.is_continuous = is_continuous
        self.is_blind = is_blind
        self.rng = np.random.default_rng(seed)
        self.var_cat = 'Continuous' if is_continuous else 'Integer'
        
        # SAA scenarios count
        self.num_scenarios = 10

        # Solver diagnostics
        self.last_solve_time = 0.0
        self.last_solve_status = 0

        # --- Extract Static Parameters ---
        self.lead_times = {}
        self.prices = {}
        self.pipe_costs = {}
        self.backlog_pen = {}
        self.holding_costs = {}
        self.op_costs = {}
        self.yields = {}
        self.capacities = {}
        self.fixed_costs = {}

        for e in self.env.graph.edges():
            data = self.env.graph.edges[e]
            self.lead_times[e] = int(data.get('L', 0))
            self.prices[e] = data.get('p', 0.0)
            self.pipe_costs[e] = data.get('g', 0.0)
            self.backlog_pen[e] = data.get('b', 0.0)
            self.fixed_costs[e] = data.get('K', 0.0)

        for n in self.env.graph.nodes():
            data = self.env.graph.nodes[n]
            self.holding_costs[n] = data.get('h', 0.0)
            self.op_costs[n] = data.get('o', 0.0)
            self.yields[n] = data.get('v', 1.0)
            self.capacities[n] = data.get('C', 1e9)

        self.main_nodes = sorted([n for n in self.env.graph.nodes() if n not in self.env.network.market and n not in self.env.network.rawmat])

    def _generate_scenario_tree(self, current_period, effective_horizon):
        """
        Sample Average Approximation (SAA) for multi-retailer networks.
        Uses independent stochastic paths while preserving non-anticipativity
        at t=0 and scaling to any topology.
        """
        final_demands = []
        final_probs = [1.0 / self.num_scenarios] * self.num_scenarios

        for s in range(self.num_scenarios):
            scenario_path = {idx: [] for idx in range(len(self.env.network.retail_links))}
            for t in range(effective_horizon):
                for idx, (r, m) in enumerate(self.env.network.retail_links):
                    query_period = current_period if self.is_blind else current_period + t
                    mu = get_demand_prior(self.env, self.is_blind, query_period, idx, (r, m))
                    
                    if t < self.depth:
                        # Stochastic branching via reproducible agent RNG
                        demand_sample = self.rng.poisson(mu)
                    else:
                        # Deterministic tail
                        demand_sample = int(np.round(mu))
                        
                    scenario_path[idx].append(demand_sample)
            final_demands.append(scenario_path)

        return final_demands, final_probs

    def get_action(self, current_period):
        t_start = time.perf_counter()
        
        # Cap horizon to remaining episode length
        effective_horizon = min(self.H, self.env.num_periods - current_period)
        
        max_lt = 0
        for e in self.env.graph.edges():
            max_lt = max(max_lt, int(self.env.graph.edges[e].get('L', 0)))
        effective_horizon = max(effective_horizon, min(self.env.num_periods - current_period, max_lt + 2))
        
        if effective_horizon <= 0:
            self.last_solve_time = time.perf_counter() - t_start
            self.last_solve_status = 1
            return np.zeros(len(self.env.network.reorder_links))

        scenarios, scenario_probs = self._generate_scenario_tree(current_period, effective_horizon)
        Scenarios = range(len(scenarios))
        T_range = range(effective_horizon)

        prob = pulp.LpProblem(f"MSSP_{current_period}", pulp.LpMaximize)

        # Variables
        flow_vars = pulp.LpVariable.dicts("Flow",
            ((s, t, u, v) for s in Scenarios for t in T_range for (u,v) in self.env.network.reorder_links),
            lowBound=0, cat=self.var_cat)
        sales_vars = pulp.LpVariable.dicts("Sales",
            ((s, t, u, v) for s in Scenarios for t in T_range for (u,v) in self.env.network.retail_links),
            lowBound=0, cat=self.var_cat)
        inv_vars = pulp.LpVariable.dicts("Inv",
            ((s, t, n) for s in Scenarios for t in T_range for n in self.main_nodes),
            lowBound=0, cat='Continuous')
        backlog_vars = pulp.LpVariable.dicts("Backlog",
            ((s, t, u, v) for s in Scenarios for t in T_range for (u,v) in self.env.network.retail_links),
            lowBound=0, cat='Continuous')

        # Add fixed-cost setup variables
        setup_vars = add_fixed_costs(prob, flow_vars, self.env, T_range, Scenarios, self.is_continuous)

        # 1. Non-Anticipativity Constraints (NAC)
        for t in T_range:
            history_groups = {}
            for s in Scenarios:
                # History incorporates all retailer links
                history = tuple(tuple(scenarios[s][idx][:t]) for idx in range(len(self.env.network.retail_links)))
                if history not in history_groups: history_groups[history] = []
                history_groups[history].append(s)

            for hist, group in history_groups.items():
                leader = group[0]
                for follower in group[1:]:
                    for (u,v) in self.env.network.reorder_links:
                        prob += flow_vars[(leader, t, u, v)] == flow_vars[(follower, t, u, v)]

        # 2. Physical Constraints
        for s in Scenarios:
            for t in T_range:
                # Inventory Balance
                for n in self.main_nodes:
                    incoming = 0
                    for k in self.env.graph.predecessors(n):
                        L = self.lead_times.get((k,n), 0)
                        if t - L >= 0:
                            incoming += flow_vars[(s, t-L, k, n)]
                        else:
                            incoming += get_pipeline_arrivals(self.env, current_period, t, L, k, n)

                    outgoing = 0
                    for k in self.env.graph.successors(n):
                        if (n,k) in self.env.network.retail_links: outgoing += sales_vars[(s, t, n, k)]
                        elif (n,k) in self.env.network.reorder_links: outgoing += flow_vars[(s, t, n, k)]

                    if t == 0:
                         n_idx = self.env.network.node_map[n]
                         prev_inv = self.env.X[current_period, n_idx]
                    else:
                         prev_inv = inv_vars[(s, t-1, n)]
                    nu = self.yields[n]

                    if n in self.env.network.retail:
                        prob += inv_vars[(s, t, n)] == prev_inv + incoming - outgoing
                    else:
                        prob += inv_vars[(s, t, n)] == prev_inv + incoming - (1/nu * outgoing)

                    # Time-Step Alignment
                    if n in self.env.network.retail:
                        prob += outgoing <= prev_inv + incoming
                    else:
                        prob += (1.0/nu * outgoing) <= prev_inv

                    if n in self.env.network.factory:
                        out_reqs = pulp.lpSum([flow_vars[(s, t, n, k)] for k in self.env.graph.successors(n) if (n,k) in self.env.network.reorder_links])
                        prob += out_reqs <= self.capacities[n]

                # Retailer Logic
                for idx, (r, m) in enumerate(self.env.network.retail_links):
                    current_demand_s = scenarios[s][idx][t]

                    if self.env.backlog:
                        if t == 0:
                            if current_period > 0:
                                 retail_idx = self.env.network.retail_map[(r, m)]
                                 prev_backlog = self.env.U[current_period-1, retail_idx]
                            else:
                                 prev_backlog = 0 
                        else:
                            prev_backlog = backlog_vars[(s, t-1, r, m)]

                        prob += sales_vars[(s, t, r, m)] <= current_demand_s + prev_backlog
                        prob += backlog_vars[(s, t, r, m)] == prev_backlog + current_demand_s - sales_vars[(s, t, r, m)]
                    else:
                        prob += sales_vars[(s, t, r, m)] <= current_demand_s
                        prob += backlog_vars[(s, t, r, m)] == current_demand_s - sales_vars[(s, t, r, m)]

        # 3. Expected Value Objective
        total_obj = 0
        discount_factor = self.env.alpha

        for s in Scenarios:
            scenario_profit = 0
            prob_s = scenario_probs[s]

            for t in T_range:
                period_profit = 0
                for (r, m) in self.env.network.retail_links:
                    p = self.prices.get((r,m),0); b = self.backlog_pen.get((r,m),0)
                    period_profit += p * sales_vars[(s, t, r, m)] - b * backlog_vars[(s, t, r, m)]

                for (src, dst) in self.env.network.reorder_links:
                    p = self.prices.get((src,dst),0); g = self.pipe_costs.get((src,dst),0); L = self.lead_times.get((src,dst),0)
                    K = self.fixed_costs.get((src,dst),0)
                    flow = flow_vars[(s, t, src, dst)]
                    if src in self.main_nodes: period_profit += p * flow
                    if dst in self.main_nodes: period_profit -= p * flow
                    if src in self.env.network.factory:
                        period_profit -= (self.op_costs.get(src,0)/self.yields.get(src,1)) * flow
                        
                    if K > 0 and setup_vars is not None:
                        period_profit -= K * setup_vars[(s, t, src, dst)]

                for n in self.main_nodes:
                    period_profit -= self.holding_costs.get(n,0) * inv_vars[(s, t, n)]

                scenario_profit += (discount_factor ** t) * period_profit

            # Pipeline holding costs follow CoreEnv's period-by-period Y[t+1] charge.
            for t in T_range:
                for (src, dst) in self.env.network.reorder_links:
                    if dst not in self.main_nodes:
                        continue
                    g = self.pipe_costs.get((src, dst), 0)
                    L = self.lead_times.get((src, dst), 0)
                    flow = flow_vars[(s, t, src, dst)]
                    scenario_profit -= pipeline_holding_charge(
                        flow, g, L, t, effective_horizon, discount_factor)

            total_obj += prob_s * scenario_profit

        prob += total_obj

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        self.last_solve_time = time.perf_counter() - t_start
        self.last_solve_status = prob.status

        if prob.status != 1:
            logging.error(f"Solver Failed at period {current_period} with status {prob.status}")
            return np.zeros(len(self.env.network.reorder_links))

        action_t0 = np.zeros(len(self.env.network.reorder_links))
        # NAC guarantees that Scenario 0 represents all scenarios at t=0
        for i, edge in enumerate(self.env.network.reorder_links):
            val = pulp.value(flow_vars[(0, 0, edge[0], edge[1])])
            if val is None:
                action_t0[i] = 0
            else:
                action_t0[i] = max(0, val)

        return action_t0
