"""
curriculum_wrapper.py (v4)

Adds a third episode mode on top of the original two (sustained flow /
general burst): "straggler" episodes, which start with only 1-2
vehicles total, on 1-2 approaches, and zero everywhere else.

Why: run_clearance_batch.py (post switch-penalty-boundary-fix) showed
the v3 seeds getting stuck indefinitely -- 220+ identical decisions --
whenever the network reduces to a small residual on a through-only
approach while the policy's chosen action is a protected-left phase
that structurally cannot serve that approach. The general burst mode
already samples near-empty states via MIN_COUNT=0, but apparently not
often or narrowly enough to teach the policy how to escape this exact
shape of dead-end. Straggler episodes directly oversample it.

This does NOT touch the observation function, action space, or
reward_fn -- fully comparable to v3 checkpoints/evals, just a change
to what states get visited during training.

Layering unchanged from v3:
    env = SumoEnvironment(...)
    env = ClearanceCurriculumWrapper(env, ORIGINAL_ROUTE, CURRICULUM_ROUTE)
    env = SwitchPenaltyWrapper(env, penalty=0.5)
    env = Monitor(env)

Place in simulation/onelast/, replacing the v3 curriculum_wrapper.py.
"""

import random
import shutil

import gymnasium as gym

from generate_scenario import generate_route_file

# Probability a given training episode is a burst-clearance episode
# (straggler or general) rather than the original sustained-flow episode.
BURST_EPISODE_PROB = 0.6

# Within a burst episode, probability it's specifically a "straggler"
# episode (near-cleared residual state) rather than a general random
# burst. Tunable: higher = more emphasis on the exact failure mode
# just diagnosed, at some risk of under-training general burst behavior
# if pushed too high.
STRAGGLER_EPISODE_PROB = 0.3

# General burst mode: random per-approach counts drawn uniformly in
# this range each burst episode.
MIN_COUNT = 0
MAX_COUNT = 13

# Straggler mode: 1-2 approaches get a small nonzero count, the rest
# are exactly zero. This directly targets the "one or two vehicles
# left on a through lane, policy stuck on a phase that can't serve
# them" dead-end.
STRAGGLER_MIN_VEHICLES = 1
STRAGGLER_MAX_VEHICLES = 2
STRAGGLER_MIN_APPROACHES = 1
STRAGGLER_MAX_APPROACHES = 2

APPROACHES = ("N", "S", "E", "W")


class ClearanceCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, original_route_path: str, curriculum_route_path: str):
        super().__init__(env)
        self.original_route_path = original_route_path
        self.curriculum_route_path = curriculum_route_path

    def _generate_general_burst_counts(self) -> dict:
        return {a: random.randint(MIN_COUNT, MAX_COUNT) for a in APPROACHES}

    def _generate_straggler_counts(self) -> dict:
        counts = {a: 0 for a in APPROACHES}
        n_active = random.randint(STRAGGLER_MIN_APPROACHES, STRAGGLER_MAX_APPROACHES)
        active_approaches = random.sample(APPROACHES, n_active)
        for a in active_approaches:
            counts[a] = random.randint(STRAGGLER_MIN_VEHICLES, STRAGGLER_MAX_VEHICLES)
        return counts

    def reset(self, **kwargs):
        if random.random() < BURST_EPISODE_PROB:
            if random.random() < STRAGGLER_EPISODE_PROB:
                counts = self._generate_straggler_counts()
            else:
                counts = self._generate_general_burst_counts()
            generate_route_file(counts, self.curriculum_route_path)
        else:
            shutil.copyfile(self.original_route_path, self.curriculum_route_path)

        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Patch: end the episode on true clearance, not just on the
        # num_seconds truncation. env.sumo is sumo-rl's own labeled
        # TraCI connection (per project's established connection lesson).
        if not terminated and self.env.sumo.simulation.getMinExpectedNumber() == 0:
            terminated = True

        return obs, reward, terminated, truncated, info