"""
clearance_metrics.py

Phase B — comparison metric: given ONE generated .rou.xml scenario
(from generate_scenario.py), run it once under the fixed-timer signal
program (already baked into onelast.net.xml's tlLogic — no manual
phase-switching needed, confirmed from Stage5_Onelast_Script.py, which
never calls traci.trafficlight.setPhase()) and once under the trained
RL agent, and report clearance time for each.

Clearance time = simulated seconds elapsed until
traci.simulation.getMinExpectedNumber() == 0, i.e. every vehicle that
was ever going to depart has departed AND arrived. Same termination
condition for both runs -> fair comparison per Phase B design.

ASSUMPTIONS TO CONFIRM (marked below) — these mirror what's already
established in train.py / observations.py per project notes, but I
don't have those files verbatim, so please check the marked lines:
  - SumoEnvironment constructor arg names (net_file, route_file, ...)
  - single-agent / traffic signal id (assumed "J1")
  - FourApproachQueueObservation import path (assumed rl_agent.observations)
  - delta_time=8, yellow_time=2, min_green=5, matches train.py

Usage:
    from clearance_metrics import run_fixed_timer_clearance, run_rl_clearance

    fixed_time = run_fixed_timer_clearance(net_path, rou_path)
    rl_time = run_rl_clearance(net_path, rou_path, model_path)
"""

import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("ERROR: Please set SUMO_HOME environment variable")

import traci

# Safety cap so a gridlocked/oversaturated scenario can't hang forever.
# Matches the MAX_STEPS discipline already used in Stage5 (72000 steps
# at 0.05s = 3600s). Generated scenarios are much smaller (burst window
# ~30s), so this is a generous ceiling, not a tight one.
DEFAULT_MAX_SIM_SECONDS = 1800.0  # 30 sim-minutes


def run_fixed_timer_clearance(
    net_path: str,
    rou_path: str,
    step_length: float = 0.05,
    max_sim_seconds: float = DEFAULT_MAX_SIM_SECONDS,
) -> dict:
    """
    Runs the scenario under SUMO's own tlLogic-driven fixed-timer program
    (no Python-side phase switching — matches Stage5's approach exactly).

    Returns:
        {
            "clearance_time": float seconds, or None if max_sim_seconds hit
            "steps": int simulation steps taken,
            "cleared": bool,
        }
    """
    max_steps = int(max_sim_seconds / step_length)

    sumo_cfg = [
        "sumo",
        "--net-file", net_path,
        "--route-files", rou_path,
        "--step-length", str(step_length),
        "--no-step-log", "true",
        "--no-warnings", "true",
        "--quit-on-end",
    ]

    traci.start(sumo_cfg)
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0 and step < max_steps:
            traci.simulationStep()
            step += 1

        cleared = traci.simulation.getMinExpectedNumber() == 0
        clearance_time = traci.simulation.getTime() if cleared else None
    finally:
        traci.close()

    return {"clearance_time": clearance_time, "steps": step, "cleared": cleared}


def run_rl_clearance(
    net_path: str,
    rou_path: str,
    model_path: str,
    delta_time: int = 8,
    yellow_time: int = 2,
    min_green: int = 5,
    max_sim_seconds: float = DEFAULT_MAX_SIM_SECONDS,
) -> dict:
    """
    Runs the same scenario under the trained PPO agent, using env.sumo
    (sumo-rl's own labeled TraCI connection) rather than the bare traci
    module — matches the project's established TraCI-connection lesson.

    Returns same shape as run_fixed_timer_clearance().
    """
    # Deferred imports: only needed for the RL path, and importing
    # stable_baselines3 / sumo_rl unconditionally would break the
    # fixed-timer-only use case if those packages aren't installed.
    from stable_baselines3 import PPO
    from sumo_rl import SumoEnvironment

    # ASSUMPTION: import path for the custom observation class.
    # Adjust if observations.py lives elsewhere / is named differently.
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

    model = PPO.load(model_path)

    obs, info = env.reset()
    steps = 0
    cleared = False
    clearance_time = None

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            # Check true clearance via env's own TraCI connection, not
            # bare traci (env.sumo is sumo-rl's labeled connection).
            if env.sumo.simulation.getMinExpectedNumber() == 0:
                cleared = True
                clearance_time = env.sumo.simulation.getTime()
                break

            if terminated or truncated:
                # Episode ended (e.g. num_seconds cap) without clearing.
                break
    finally:
        env.close()

    return {"clearance_time": clearance_time, "steps": steps, "cleared": cleared}


if __name__ == "__main__":
    import argparse
    from generate_scenario import generate_route_file

    p = argparse.ArgumentParser(description="Phase B manual test: compare clearance time on one generated scenario.")
    p.add_argument("--net", default="onelast.net.xml")
    p.add_argument("--model", default="rl_agent/models/ppo_onelast_v1.zip")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--e", type=int, default=5)
    p.add_argument("--w", type=int, default=5)
    args = p.parse_args()

    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    rou_path = "clearance_test.rou.xml"
    generate_route_file(counts, rou_path)

    print(f"Scenario: {counts}")

    fixed_result = run_fixed_timer_clearance(args.net, rou_path)
    print(f"Fixed-timer: {fixed_result}")

    rl_result = run_rl_clearance(args.net, rou_path, args.model)
    print(f"RL agent:    {rl_result}")

    if fixed_result["cleared"] and rl_result["cleared"]:
        diff = fixed_result["clearance_time"] - rl_result["clearance_time"]
        pct = (diff / fixed_result["clearance_time"]) * 100
        print(f"\nRL cleared {diff:.2f}s faster ({pct:.1f}% reduction)" if diff > 0
              else f"\nFixed-timer cleared {-diff:.2f}s faster")