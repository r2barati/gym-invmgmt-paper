import numpy as np

from gym_invmgmt.core_env import CoreEnv
from agents.oracle import StandaloneOracleOptimizer


class BidirectionalOracle:
    """
    Fixed-point simulation-optimization benchmark for endogenous goodwill scenarios.

    The procedure solves full-horizon LPs against two demand traces:
    a pessimistic trace induced by zero actions and an optimistic trace induced by
    no-shortage goodwill growth. Each candidate plan is replayed in the real
    endogenous environment, and the higher-profit feasible trajectory is retained.
    This is a clairvoyant benchmark, not a certified dynamic-programming upper
    bound.
    """

    def __init__(self, env_kwargs, planning_horizon=30, max_iterations=10, tolerance=1.0):
        self.env_kwargs = env_kwargs
        self.H = planning_horizon
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def _extract_demand_trace(self, env):
        demand_trace = {}
        for i, edge in enumerate(env.network.retail_links):
            demand_trace[edge] = env.D[:, i].copy()
        return demand_trace

    def _get_optimistic_trace(self, seed):
        """Return the demand trace under no-shortage goodwill growth."""
        sim_env = CoreEnv(**self.env_kwargs)
        sim_env.reset(seed=seed)
        for t in range(self.H):
            sim_env.step(np.zeros(sim_env.action_space.shape))
            sim_env.U[t, :] = 0.0
        return self._extract_demand_trace(sim_env)

    def _get_pessimistic_trace(self, seed):
        """Return the demand trace induced by zero replenishment actions."""
        sim_env = CoreEnv(**self.env_kwargs)
        sim_env.reset(seed=seed)
        for _ in range(self.H):
            sim_env.step(np.zeros(sim_env.action_space.shape))
        return self._extract_demand_trace(sim_env)

    def _iterate_from_trace(self, seed, initial_trace):
        """Run the fixed-point LP/replay loop from a given demand trace."""
        current_trace = initial_trace
        best_actions = None
        best_profit = -np.inf

        for _ in range(self.max_iterations):
            fixed_cfg = self.env_kwargs.copy()
            fixed_cfg['user_D'] = current_trace
            opt_env = CoreEnv(**fixed_cfg)
            opt_env.reset(seed=seed)

            oracle = StandaloneOracleOptimizer(
                opt_env,
                current_trace,
                self.H,
                is_continuous=True,
            )
            proposed = oracle.solve_full_horizon()
            if proposed is None:
                break

            sim_env = CoreEnv(**self.env_kwargs)
            sim_env.reset(seed=seed)
            for t in range(self.H):
                sim_env.step(proposed[t])

            realized_profit = float(np.sum(sim_env.P))
            new_trace = self._extract_demand_trace(sim_env)

            if realized_profit > best_profit:
                best_profit = realized_profit
                best_actions = proposed

            diff = sum(np.sum(np.abs(new_trace[e] - current_trace[e])) for e in new_trace)
            if diff <= self.tolerance:
                break

            current_trace = new_trace

        return best_actions, best_profit

    def solve(self, seed):
        """Return the best feasible plan and replayed environment state."""
        pessimistic_trace = self._get_pessimistic_trace(seed)
        pessimistic_actions, pessimistic_profit = self._iterate_from_trace(seed, pessimistic_trace)

        optimistic_trace = self._get_optimistic_trace(seed)
        optimistic_actions, optimistic_profit = self._iterate_from_trace(seed, optimistic_trace)

        if optimistic_profit >= pessimistic_profit and optimistic_actions is not None:
            best_actions = optimistic_actions
        elif pessimistic_actions is not None:
            best_actions = pessimistic_actions
        else:
            return None, None

        final_env = CoreEnv(**self.env_kwargs)
        final_env.reset(seed=seed)
        for t in range(self.H):
            final_env.step(best_actions[t])

        return best_actions, final_env
