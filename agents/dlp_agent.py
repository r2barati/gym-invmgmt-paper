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

class DLPAgent:
    def __init__(self, env, planning_horizon=10, is_continuous=True, is_blind=False):
        self.env = env
        self.H = planning_horizon
        self.is_continuous = is_continuous
        self.is_blind = is_blind
        self.var_cat = 'Continuous' if is_continuous else 'Integer'

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

        # Identify Main Nodes
        self.main_nodes = sorted([n for n in self.env.graph.nodes() if n not in self.env.network.market and n not in self.env.network.rawmat])

    def get_action(self, obs, current_period):
        t_start = time.perf_counter()
        
        # Cap horizon to the actual remaining episode length
        effective_horizon = min(self.H, self.env.num_periods - current_period)
        
        # Adaptive Horizon to prevent pipeline hiding exploit
        max_lt = 0
        for e in self.env.graph.edges():
            max_lt = max(max_lt, int(self.env.graph.edges[e].get('L', 0)))
        
        # We ensure horizon is at least enough to capture lead times, but cap at remaining
        effective_horizon = max(effective_horizon, min(self.env.num_periods - current_period, max_lt + 2))
        
        if effective_horizon <= 0:
            self.last_solve_time = time.perf_counter() - t_start
            self.last_solve_status = 1
            return np.zeros(len(self.env.network.reorder_links))

        prob = pulp.LpProblem(f"DLP_Optimization_t{current_period}", pulp.LpMaximize)
        T_range = range(effective_horizon)
        Scenarios = [0] # DLP is a single-scenario case

        flow_vars = pulp.LpVariable.dicts("Flow",
                                          ((0, t, u, v) for t in T_range for (u,v) in self.env.network.reorder_links),
                                          lowBound=0, cat=self.var_cat)
        sales_vars = pulp.LpVariable.dicts("Sales",
                                           ((0, t, u, v) for t in T_range for (u,v) in self.env.network.retail_links),
                                           lowBound=0, cat=self.var_cat)
        inv_vars = pulp.LpVariable.dicts("Inv",
                                         ((0, t, n) for t in T_range for n in self.main_nodes),
                                         lowBound=0, cat='Continuous')
        backlog_vars = pulp.LpVariable.dicts("Backlog",
                                             ((0, t, u, v) for t in T_range for (u,v) in self.env.network.retail_links),
                                             lowBound=0, cat='Continuous')

        # Add fixed-cost setup variables
        setup_vars = add_fixed_costs(prob, flow_vars, self.env, T_range, Scenarios, self.is_continuous)

        for t in T_range:
            # A. Inventory Balance
            for n in self.main_nodes:
                incoming_qty = 0
                for k in self.env.graph.predecessors(n):
                    L = self.lead_times.get((k, n), 0)
                    if t - L >= 0:
                        incoming_qty += flow_vars[(0, t - L, k, n)]
                    else:
                        # Use shared or_utils historical pipeline warm-start
                        incoming_qty += get_pipeline_arrivals(self.env, current_period, t, L, k, n)

                outgoing_qty = 0
                for k in self.env.graph.successors(n):
                    if (n, k) in self.env.network.retail_links:
                        outgoing_qty += sales_vars[(0, t, n, k)]
                    elif (n, k) in self.env.network.reorder_links:
                        outgoing_qty += flow_vars[(0, t, n, k)]

                if t == 0:
                    n_idx = self.env.network.node_map[n]
                    prev_inv = self.env.X[current_period, n_idx]
                else:
                    prev_inv = inv_vars[(0, t-1, n)]

                nu = self.yields.get(n, 1.0)

                if n in self.env.network.retail:
                     prob += inv_vars[(0, t, n)] == prev_inv + incoming_qty - outgoing_qty
                else:
                     prob += inv_vars[(0, t, n)] == prev_inv + incoming_qty - (1.0/nu * outgoing_qty)

                if n in self.env.network.retail:
                    prob += outgoing_qty <= prev_inv + incoming_qty
                else:
                    prob += (1.0/nu * outgoing_qty) <= prev_inv

                if n in self.env.network.factory:
                     out_reqs = pulp.lpSum([flow_vars[(0, t, n, k)] for k in self.env.graph.successors(n) if (n,k) in self.env.network.reorder_links])
                     prob += out_reqs <= self.capacities.get(n, 1e9)

            # B. Demand Logic
            for (r, m) in self.env.network.retail_links:
                retail_idx = self.env.network.retail_map[(r, m)]
                # Use shared or_utils demand estimator, capturing future drift ONLY if informed
                query_period = current_period if self.is_blind else current_period + t
                mu = get_demand_prior(self.env, self.is_blind, query_period, retail_idx, (r, m))

                if self.env.backlog:
                    if t == 0:
                         if current_period > 0:
                              prev_backlog = self.env.U[current_period-1, retail_idx]
                         else:
                              prev_backlog = 0 
                    else:
                         prev_backlog = backlog_vars[(0, t-1, r, m)]

                    prob += sales_vars[(0, t, r, m)] <= mu + prev_backlog
                    prob += backlog_vars[(0, t, r, m)] == prev_backlog + mu - sales_vars[(0, t, r, m)]
                else:
                    prob += sales_vars[(0, t, r, m)] <= mu
                    prob += backlog_vars[(0, t, r, m)] == mu - sales_vars[(0, t, r, m)]

        # Objective
        total_profit = 0
        discount_factor = self.env.alpha
        for t in T_range:
            period_profit = 0
            for (r, m) in self.env.network.retail_links:
                p = self.prices.get((r, m), 0)
                b = self.backlog_pen.get((r, m), 0)
                period_profit += p * sales_vars[(0, t, r, m)] - b * backlog_vars[(0, t, r, m)]

            for (src, dst) in self.env.network.reorder_links:
                p = self.prices.get((src, dst), 0)
                g = self.pipe_costs.get((src, dst), 0)
                K = self.fixed_costs.get((src, dst), 0)
                L = self.lead_times.get((src, dst), 0)
                flow = flow_vars[(0, t, src, dst)]

                if src in self.main_nodes: period_profit += p * flow
                if dst in self.main_nodes: period_profit -= p * flow

                if src in self.env.network.factory:
                    period_profit -= (self.op_costs.get(src, 0)/self.yields.get(src, 1.0)) * flow

                # Add Fixed ordering cost K if applicable
                if K > 0 and setup_vars is not None:
                    period_profit -= K * setup_vars[(0, t, src, dst)]

            for n in self.main_nodes:
                period_profit -= self.holding_costs.get(n, 0) * inv_vars[(0, t, n)]

            total_profit += (discount_factor ** t) * period_profit

        # Pipeline holding costs follow CoreEnv's period-by-period Y[t+1] charge.
        for t in T_range:
            for (src, dst) in self.env.network.reorder_links:
                if dst not in self.main_nodes:
                    continue
                g = self.pipe_costs.get((src, dst), 0)
                L = self.lead_times.get((src, dst), 0)
                flow = flow_vars[(0, t, src, dst)]
                total_profit -= pipeline_holding_charge(
                    flow, g, L, t, effective_horizon, discount_factor)

        prob += total_profit

        # Solve silently
        prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))
        
        self.last_solve_time = time.perf_counter() - t_start
        self.last_solve_status = prob.status

        if prob.status != 1:
            logging.error(f"Solver Failed at period {current_period} with status {prob.status}")
            return np.zeros(len(self.env.network.reorder_links))

        action_t0 = np.zeros(len(self.env.network.reorder_links))
        for i, edge in enumerate(self.env.network.reorder_links):
            val = pulp.value(flow_vars[(0, 0, edge[0], edge[1])])
            if val is None:
                action_t0[i] = 0
            else:
                action_t0[i] = max(0, val)

        return action_t0
