"""
diagnose_rl_gridlock.py

Logs the RL agent's per-decision behavior (observation, action, queue
state) for a single scenario, step by step, so gridlock cases can be
inspected directly instead of guessed at.

Usage (run from simulation/onelast/):
    python diagnose_rl_gridlock.py --net onelast.net.xml --n 2 --s 2 --e 10 --w 10
    (mod_ew_heavy — one of the scenarios that failed to clear)
"""

import argparse
import os
import sys

from generate_scenario import generate_route_file

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("ERROR: Please set SUMO_HOME environment variable")


def diagnose(net_path, rou_path, model_path, max_sim_seconds=1800.0,
             delta_time=8, yellow_time=2, min_green=5, max_decisions=60):
    from stable_baselines3 import PPO
    from sumo_rl import SumoEnvironment

    BASE_DIR=os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__),".."),".."))
    sys.path.append(BASE_DIR)
    from rl_agent.observations import FourApproachQueueObservation

    env = SumoEnvironment(
        net_file=net_path,
        route_file=rou_path,
        use_gui=False,
        num_seconds=int(max_sim_seconds),
        delta_time=delta_time,
        yellow_time=yellow_time,
        min_green=min_green,
        observation_class=FourApproachQueueObservation,
        single_agent=True,
    )

    model = PPO.load(model_path, env=env)  # type: ignore
    obs, info = env.reset()

    print(f"{'decision':>8} {'sim_t':>7} {'action':>6} {'obs (N,S,E,W)':>28} {'waiting':>8}")
    print("-" * 65)

    for decision in range(max_decisions):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        sim_t = env.sumo.simulation.getTime()
        n_left = env.sumo.simulation.getMinExpectedNumber()
        obs_str = ", ".join(f"{x:.2f}" for x in obs)

        print(f"{decision:>8} {sim_t:>7.1f} {int(action):>6} {obs_str:>28} {n_left:>8}")

        if n_left == 0:
            print(f"\nCleared at t={sim_t:.1f}s after {decision + 1} decisions.")
            env.close()
            return

        if terminated or truncated:
            print(f"\nEpisode ended (terminated={terminated}, truncated={truncated}) "
                  f"without clearing. {n_left} vehicles still expected.")
            env.close()
            return

    print(f"\nStopped after {max_decisions} decisions (diagnostic cap, not env cap). "
          f"{env.sumo.simulation.getMinExpectedNumber()} vehicles still expected.")
    env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="simulation/onelast/onelast.net.xml")
    p.add_argument("--model", default="rl_agent/models/ppo_onelast_v3_seed1.zip")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--s", type=int, required=True)
    p.add_argument("--e", type=int, required=True)
    p.add_argument("--w", type=int, required=True)
    p.add_argument("--max-decisions", type=int, default=60)
    args = p.parse_args()

    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    rou_path = "_diagnose.rou.xml"
    generate_route_file(counts, rou_path)

    print(f"Scenario: {counts}\n")
    diagnose(args.net, rou_path, args.model, max_decisions=args.max_decisions)

    try:
        os.remove(rou_path)
    except OSError:
        pass