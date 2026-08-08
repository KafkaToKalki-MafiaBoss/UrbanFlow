"""
train_curriculum.py

Trains a FRESH PPO model with a curriculum mixing sustained-flow
episodes and randomized burst-clearance episodes (see
curriculum_wrapper.py for the reasoning).

NOTE: this trains from scratch, NOT a warm-start from v1. Confirmed via
`PPO.load(v1_path, env=env)` raising "Action spaces do not match:
Discrete(2) != Discrete(4)": v1.zip's policy network was built with a
2-action output head (trained back when onelast.net.xml's tlLogic
apparently resolved to only 2 green phases), while the CURRENT net
file has 4 green phases (2 through + 2 protected-left, confirmed via
check_action_space.py). A 2-output policy cannot be attached to a
4-action env -- there is no valid way to reuse v1's weights here, so
warm-starting is not an option. This also means every prior result
(evaluate.py's 81%, the 20-scenario batch, the starvation diagnosis)
was produced by a model that could only ever select 2 of the
intersection's 4 real phases -- worth stating explicitly in the writeup.

Saves to rl_agent/models/ppo_onelast_v2.zip -- v1 is left untouched.
v2 is not directly comparable to v1's old numbers (different action
space entirely) -- plan to re-run evaluate.py AND the 20-scenario
clearance batch against v2 once training finishes, for a clean,
apples-to-apples set of headline numbers.

Run from anywhere; paths are anchored to this script's location.
Place this file in rl_agent/, next to observations.py (bare-imported
below, same convention as train.py).
"""

import os
import sys
import shutil

if "SUMO_HOME" not in os.environ:
    print("ERROR: Please set the SUMO_HOME environment variable")
    sys.exit(1)

import sumo_rl
from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "simulation", "onelast"))
from simulation.onelast.curriculum_wrapper import ClearanceCurriculumWrapper

from observations import FourApproachQueueObservation


ORIGINAL_ROUTE = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.rou.xml")
CURRICULUM_ROUTE = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast_curriculum.rou.xml")
NET_FILE = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.net.xml")

# Seed the curriculum route file with a valid copy before first reset.
shutil.copyfile(ORIGINAL_ROUTE, CURRICULUM_ROUTE)

env = SumoEnvironment(
    net_file=NET_FILE,
    route_file=CURRICULUM_ROUTE,
    use_gui=False,
    num_seconds=3600,  # ceiling only -- burst episodes end early via terminated=True
    single_agent=True,
    observation_class=FourApproachQueueObservation,
    delta_time=8,
    yellow_time=2,
    min_green=5,
)
env = ClearanceCurriculumWrapper(env, ORIGINAL_ROUTE, CURRICULUM_ROUTE)
env = Monitor(env)

model_dir = os.path.join(PROJECT_ROOT, "rl_agent", "models")
checkpoint_dir = os.path.join(PROJECT_ROOT, "rl_agent", "checkpoints_curriculum")
log_dir = os.path.join(PROJECT_ROOT, "rl_agent", "logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

checkpoint_callback = CheckpointCallback(
    save_freq=25_000, save_path=checkpoint_dir, name_prefix="ppo_onelast_v2"
)

# Fresh model -- action_space is correctly Discrete(4) from the start,
# matching the environment's real 4 green phases. Same policy/hparams
# as train.py's original run for a fair comparison.
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Start at the same order of magnitude as v1's 300k. Burst episodes
# terminate early (often well under 100s of sim time vs. the full
# 3600s sustained-flow episodes), so wall-clock time per timestep may
# average out faster than v1's run -- watch TensorBoard ep_rew_mean and
# extend if it hasn't stabilized by 300k.
TOTAL_TIMESTEPS = 300_000
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    tb_log_name="ppo_onelast_v2",
)

model.save(os.path.join(model_dir, "ppo_onelast_v2"))
print(f"Saved model to {os.path.join(model_dir, 'ppo_onelast_v2.zip')}")

env.close()