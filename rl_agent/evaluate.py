import os
import sys
import sqlite3
import numpy as np

if "SUMO_HOME" not in os.environ:
    print("ERROR: Please set the SUMO_HOME environment variable")
    sys.exit(1)

from sumo_rl import SumoEnvironment
from stable_baselines3 import PPO
from observations import FourApproachQueueObservation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_fixed_timer_baseline():
    db_path = os.path.join(PROJECT_ROOT, "simulation", "onelast", "baseline_results.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT AVG(E3_queue), AVG(E2_queue), AVG(E1_queue), AVG(E0_queue)
        FROM (
            SELECT 
                E3_queue, E2_queue, E1_queue, E0_queue,
                step / 160 AS bucket
            FROM fixed_timer_baseline
        )
        GROUP BY bucket,
    """)
    row = cursor.fetchone()
    conn.close()

    import numpy as np
    rows = np.array(rows)  # columns now in order: [North, South, East, West]

    MAX_QUEUE = 20.0
    rows_normalized = np.minimum(rows / MAX_QUEUE, 1.0)  # same scale + same cap as RL obs

    return {
        "label": "fixed_timer",
        "avg_queue_per_approach": rows_normalized.mean(axis=0),  # [North, South, East, West]
        "avg_queue_overall": rows_normalized.mean(),
        "peak_queue": rows_normalized.max(),
    }


def run_rl_episode(model_path):
    env = SumoEnvironment(
        net_file=os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.net.xml"),
        route_file=os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.rou.xml"),
        use_gui=False,
        num_seconds=3600,
        single_agent=True,
        observation_class=FourApproachQueueObservation,
        delta_time=8,
        yellow_time=2,
        min_green=5,
    )
    model = PPO.load(model_path)

    obs, info = env.reset()
    queue_history = []
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        queue_history.append(obs.copy())

    env.close()

    queue_history = np.array(queue_history)  # shape: (steps, 4) -> [N,S,E,W]
    return {
        "label": "rl_agent",
        "avg_queue_per_approach": queue_history.mean(axis=0),
        "avg_queue_overall": queue_history.mean(),
        "peak_queue": queue_history.max(),
    }


def print_comparison(fixed, rl):
    print("\n" + "=" * 70)
    print("BASELINE vs RL AGENT — ONELAST NETWORK")
    print("=" * 70)

    print(f"\n{'Metric':<30}{'Fixed-Timer':>18}{'RL Agent':>18}")
    print("-" * 70)
    print(f"{'Avg queue (overall)':<30}{fixed['avg_queue_overall']:>18.2f}{rl['avg_queue_overall']:>18.2f}")
    print(f"{'Peak queue (any approach)':<30}{fixed['peak_queue']:>18.2f}{rl['peak_queue']:>18.2f}")

    improvement = (fixed["avg_queue_overall"] - rl["avg_queue_overall"]) / fixed["avg_queue_overall"] * 100
    print(f"\nRL improvement over fixed-timer: {improvement:.1f}% reduction in avg queue")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    fixed_result = load_fixed_timer_baseline()
    rl_result = run_rl_episode(
        os.path.join(PROJECT_ROOT, "rl_agent", "models", "ppo_onelast_v1.zip")
    )
    print_comparison(fixed_result, rl_result)