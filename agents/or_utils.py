import numpy as np
import pulp

def get_demand_prior(env, is_blind, current_period, retail_idx, edge):
    """
    Returns the estimated mean demand (mu) for a specific retail edge.
    If informed, uses the ground-truth demand engine parameters.
    If blind, uses a rolling average of past realized demand, or a default prior at t=0.
    """
    if is_blind:
        if current_period == 0:
            # Try to pull from config if available to avoid hard-coded 20.0
            config_mu = getattr(env.demand_engine, 'config', {}).get('base_mu', 20.0)
            return config_mu
        else:
            start_idx = max(0, current_period - 5)
            # Rolling mean of past realized demand
            return np.mean(env.D[start_idx:current_period, retail_idx])
    else:
        # Informed: peek at future distribution parameters
        if hasattr(env.demand_engine, 'get_edge_mu'):
            return env.demand_engine.get_edge_mu(edge, current_period)
        else:
            return env.demand_engine.get_current_mu(current_period)

def get_pipeline_arrivals(env, current_period, t_relative, L, k, n):
    """
    Calculates historical pipeline arrivals from previously executed orders.
    This provides the warm-start pipeline state for rolling horizon LP baselines.
    """
    past_time = current_period + t_relative - L
    if past_time >= 0:
        reorder_idx = env.network.reorder_map[(k, n)]
        return env.R[past_time, reorder_idx]
    return 0

def add_fixed_costs(prob, flow_vars, env, T_range, Scenarios, is_continuous):
    """
    Integrates fixed ordering costs (K) into the LP problem if present in the topology.
    This converts the LP into a MILP by adding binary setup variables.
    Returns the setup variables dictionary so it can be added to the objective.
    """
    setup_vars = None
    has_fixed_costs = any(env.graph.edges[e].get('K', 0.0) > 0 for e in env.network.reorder_links)
    
    if has_fixed_costs:
        # Always use Integer/Binary for setup variables, regardless of flow continuity
        setup_vars = pulp.LpVariable.dicts("Setup",
            ((s, t, u, v) for s in Scenarios for t in T_range for (u,v) in env.network.reorder_links),
            cat='Binary')
            
        for s in Scenarios:
            for t in T_range:
                for (u, v) in env.network.reorder_links:
                    cap = env.graph.nodes.get(u, {}).get('C', 1e6)
                    # Flow can only be positive if setup is 1
                    prob += flow_vars[(s, t, u, v)] <= cap * setup_vars[(s, t, u, v)]
                    
    return setup_vars

def pipeline_holding_charge(flow, g, L, order_period, horizon_len, discount_factor):
    """
    Simulator-aligned pipeline holding cost for one replenishment order.

    CoreEnv charges ``g * Y[t+1, edge]`` each period. An order placed at
    ``order_period`` with lead time ``L`` is therefore charged in periods
    ``order_period`` through ``order_period + L - 1``, capped by the planning
    horizon. This avoids a lumped ``g * L`` approximation when
    discounting or end-of-horizon truncation matters.
    """
    if L <= 0 or g <= 0:
        return 0.0
    final_hold_period = min(order_period + int(L), horizon_len)
    return pulp.lpSum(
        (discount_factor ** hold_t) * g * flow
        for hold_t in range(order_period, final_hold_period)
    )
