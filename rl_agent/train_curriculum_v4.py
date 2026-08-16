"""
train_curriculum_v4.py

Same as train_curriculum_v3.py (SubprocVecEnv 8-way parallel,
sumo_seed="random", explicit gamma + annealed LR, fixed
SwitchPenaltyWrapper with the <= boundary correction) PLUS the new
straggler-episode curriculum (curriculum_wrapper.py v4) targeting the
stuck-on-protected-left-phase dead-end found in the post-v3 clearance
batch (see run_clearance_batch.py results: mild_ew_heavy, mod_ew_heavy,
severe_e_only, severe_w_only, etc. -- all froze on action=2 for 220+
decisions with a nonzero-but-unchanging residual queue).

Usage (run one seed at a time -- 8 envs already targets full machine):

    python rl_agent/train_curriculum_v4.py --seed 1 --timesteps 1500000
    python rl_agent/train_curriculum_v4.py --seed 2 --timesteps 1500000
    python rl_agent/train_curriculum_v4.py --seed 3 --timesteps 1500000

Place in rl_agent/, next to train_curriculum_v3.py (left untouched).
"""

import argparse
import os
import sys
import shutil

if "SUMO_HOME" not in os.environ:
    print("ERROR: Please set the SUMO_HOME environment variable")
    sys.exit(1)

from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import get_linear_fn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "simulation", "onelast"))
from simulation.onelast.curriculum_wrapper import ClearanceCurriculumWrapper
from simulation.onelast.switch_penalty_wrapper import SwitchPenaltyWrapper

from observations import FourApproachQueueObservation


ORIGINAL_ROUTE = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.rou.xml")
NET_FILE = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.net.xml")

N_ENVS = 8


def make_env(seed: int, env_index: int, switch_penalty: float):
    """Factory for one SUMO env instance. Each env_index gets its own
    curriculum route file on disk so parallel instances never collide."""

    def _init():
        curriculum_route = os.path.join(
            PROJECT_ROOT, "simulation", "onelast",
            f"onelast_curriculum_v4_seed{seed}_env{env_index}.rou.xml",
        )
        shutil.copyfile(ORIGINAL_ROUTE, curriculum_route)

        env = SumoEnvironment(
            net_file=NET_FILE,
            route_file=curriculum_route,
            use_gui=False,
            num_seconds=3600,
            single_agent=True,
            observation_class=FourApproachQueueObservation,
            delta_time=8,
            yellow_time=2,
            min_green=5,
            sumo_seed="random",
        )
        env = ClearanceCurriculumWrapper(env, ORIGINAL_ROUTE, curriculum_route)
        env = SwitchPenaltyWrapper(env, penalty=switch_penalty)
        env = Monitor(env)
        return env

    return _init


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True,
                    help="Random seed -- run this script multiple times with different seeds.")
    p.add_argument("--timesteps", type=int, default=1_500_000,
                    help="Total training timesteps across all parallel envs combined.")
    p.add_argument("--switch-penalty", type=float, default=0.5,
                    help="Reward penalty per unjustified phase switch. Default 0.5.")
    p.add_argument("--ent-coef", type=float, default=0.01,
                    help="PPO entropy coefficient. Default 0.01 (SB3 default is 0.0).")
    p.add_argument("--n-envs", type=int, default=N_ENVS,
                    help=f"Number of parallel SUMO instances. Default {N_ENVS}.")
    args = p.parse_args()

    env_fns = [
        make_env(args.seed, i, args.switch_penalty) for i in range(args.n_envs)
    ]
    env = SubprocVecEnv(env_fns)

    model_dir = os.path.join(PROJECT_ROOT, "rl_agent", "models")
    checkpoint_dir = os.path.join(PROJECT_ROOT, "rl_agent", "checkpoints_v4", f"seed{args.seed}")
    log_dir = os.path.join(PROJECT_ROOT, "rl_agent", "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // args.n_envs, 1),
        save_path=checkpoint_dir,
        name_prefix=f"ppo_onelast_v4_seed{args.seed}",
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        ent_coef=args.ent_coef,
        gamma=0.99,
        learning_rate=get_linear_fn(3e-4, 3e-5, 1.0),
        seed=args.seed,
    )

    print(f"[train_curriculum_v4] seed={args.seed} timesteps={args.timesteps} "
          f"switch_penalty={args.switch_penalty} ent_coef={args.ent_coef} "
          f"n_envs={args.n_envs}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        tb_log_name=f"ppo_onelast_v4_seed{args.seed}",
    )

    save_path = os.path.join(model_dir, f"ppo_onelast_v4_seed{args.seed}")
    model.save(save_path)
    print(f"Saved model to {save_path}.zip")

    env.close()

    for i in range(args.n_envs):
        curriculum_route = os.path.join(
            PROJECT_ROOT, "simulation", "onelast",
            f"onelast_curriculum_v4_seed{args.seed}_env{i}.rou.xml",
        )
        try:
            os.remove(curriculum_route)
        except OSError:
            pass


if __name__ == "__main__":
    main()