"""
curriculum_wrapper.py

Wraps SumoEnvironment so each episode is EITHER:
  (a) the original sustained uniform flow (onelast.rou.xml, unchanged) --
      keeps the agent learning sustained-congestion behavior, or
  (b) a randomized burst scenario (via generate_scenario.py) that must
      fully drain -- exposes the agent to near-empty / single-vehicle-
      residual states, which is the diagnosed root cause of the
      starvation failure seen in the original model.

Episode termination is also patched: normally SumoEnvironment only ends
an episode via truncation at num_seconds. This wrapper additionally
terminates the episode the moment the network actually clears
(getMinExpectedNumber() == 0), so burst episodes give the agent a real
reward signal tied to "did I finish draining," not just "did the clock
run out."

Route file is regenerated ON DISK before each reset() -- the ORIGINAL
onelast.rou.xml is only ever read from, never overwritten. A separate
onelast_curriculum.rou.xml is what the env actually points at.
"""

import random
import shutil

import gymnasium as gym

from generate_scenario import generate_route_file

# Probability a given training episode is a burst-clearance episode
# rather than the original sustained-flow episode. Tunable: higher =
# more emphasis on the failure mode being fixed, at some risk of
# eroding sustained-flow performance if pushed too high.
BURST_EPISODE_PROB = 0.6

# Random per-approach counts drawn uniformly in this range each burst
# episode. Upper bound matches generate_scenario.py's burst-window
# capacity (~13/approach). Lower bound of 0 deliberately included --
# empty approaches are exactly the understudied region.
MIN_COUNT = 0
MAX_COUNT = 13


class ClearanceCurriculumWrapper(gym.Wrapper):
    def __init__(self, env, original_route_path: str, curriculum_route_path: str):
        super().__init__(env)
        self.original_route_path = original_route_path
        self.curriculum_route_path = curriculum_route_path

    def reset(self, **kwargs):
        if random.random() < BURST_EPISODE_PROB:
            counts = {
                a: random.randint(MIN_COUNT, MAX_COUNT)
                for a in ("N", "S", "E", "W")
            }
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