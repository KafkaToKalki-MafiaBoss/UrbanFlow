"""
switch_penalty_wrapper.py

Adds a small reward penalty whenever the agent switches its green
phase without a clear queue-based justification. Targets the exact
failure seen in diagnose_rl_gridlock.py traces: the policy toggling
between action 0 (N/S) and action 1 (E/W) every decision even when the
current phase's queue is still larger than the alternate phase's.

This does NOT touch the environment's own reward_fn (diff-waiting-time,
untouched) -- it wraps on top, same layering pattern as
ClearanceCurriculumWrapper. Stack both wrappers together:

    env = SumoEnvironment(...)
    env = ClearanceCurriculumWrapper(env, ORIGINAL_ROUTE, CURRICULUM_ROUTE)
    env = SwitchPenaltyWrapper(env, penalty=0.5)
    env = Monitor(env)

Place in simulation/onelast/, next to curriculum_wrapper.py.
"""

import gymnasium as gym


class SwitchPenaltyWrapper(gym.Wrapper):
    """Penalizes phase switches that occur while no approach has a
    meaningful queue -- i.e. frivolous toggling in a near-empty network.
    Does NOT penalize switching toward a persistent/saturated queue, even
    if that queue's value isn't changing step-to-step (a jammed queue is
    exactly the case that justifies a switch, not the opposite)."""

    def __init__(self, env, penalty: float = 0.5, min_queue_threshold: float = 0.05):
        super().__init__(env)
        self.penalty = penalty
        self.min_queue_threshold = min_queue_threshold
        self._last_action = None
        self._last_obs = None

    def reset(self, **kwargs):
        self._last_action = None
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        act = int(action)
        if self._last_action is not None and act != self._last_action:
            # Justify the switch by whether ANY approach had a meaningful
            # queue right before this decision. Only penalize switching
            # when the network was essentially empty -- that's the only
            # case where a switch has no possible benefit.
            max_queue = max(self._last_obs) if self._last_obs is not None else 0.0
            if max_queue <= self.min_queue_threshold:
                reward -= self.penalty

        self._last_action = act
        self._last_obs = obs
        return obs, reward, terminated, truncated, info