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

--- Max-green override (added after diagnosing starvation gridlock) ---
diagnose_rl_gridlock.py showed the trained policy can freeze on a
single action indefinitely once one queue shrinks to a near-empty
residual (observed: 55 consecutive identical decisions, one vehicle
permanently denied green). This is a distribution-shift failure, not
something worth retraining the model to fix under project time
constraints. Instead, run_rl_clearance() now supports an optional
max_green_seconds override: a thin wrapper AROUND model.predict(), not
a change to the model itself. If the current phase has held for
>= max_green_seconds, the wrapper forces a switch to the other phase
regardless of what the policy would have chosen. With Discrete(2),
action == target green phase index (confirmed in project notes), so
"force a switch" is simply 1 - last_action.

min_green is intentionally left untouched at the trained value (5s) --
changing it would introduce a second, new distribution shift on top of
the one already being corrected, and it wasn't implicated in the
starvation failure (the issue was no upper bound, not the lower one).
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
DEFAULT_MAX_SIM_SECONDS = 1800.0  # 30 sim-minutes

# Default max-green override, anchored to observed fixed-timer clearance
# times across the 20-scenario batch (longest was ~89s) and close to the
# fixed-timer's own per-phase allocation (42s). Keeps RL structurally
# unable to be worse than fixed-timer due to starvation, while still
# leaving it free to act intelligently under this ceiling.
DEFAULT_MAX_GREEN_SECONDS = 44


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
    max_green_seconds: float = None,
) -> dict:
    """
    Runs the same scenario under the trained PPO agent, using env.sumo
    (sumo-rl's own labeled TraCI connection) rather than the bare traci
    module — matches the project's established TraCI-connection lesson.

    max_green_seconds: if set, forces a phase switch once the current
    phase has held for this long, overriding the policy's own choice.
    None (default) preserves the original unmodified-policy behavior.

    Returns same shape as run_fixed_timer_clearance(), plus:
        "forced_switches": int, how many times the override fired.
    """
    from stable_baselines3 import PPO
    from sumo_rl import SumoEnvironment
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.append(BASE_DIR)
    from rl_agent.observations import FourApproachQueueObservation

    env = SumoEnvironment(
        net_file=net_path,
        route_file=rou_path,
        use_gui=True,
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
    forced_switches = 0

    last_action = None
    time_in_phase = 0.0

    try:
        while True:
            if (
                max_green_seconds is not None
                and last_action is not None
                and time_in_phase >= max_green_seconds
            ):
                # Confirmed via check_action_space.py: action_space is
                # Discrete(4), NOT Discrete(2) (4 green phases including
                # 2 protected-left phases). "1 - last_action" was wrong
                # and could emit an invalid action once last_action was
                # 2 or 3. Cycle to the next phase in rotation instead --
                # correct for any action_space size, guarantees every
                # phase eventually gets served (fixes starvation
                # regardless of how many phases exist).
                n_actions = env.action_space.n
                action = (last_action + 1) % n_actions
                forced_switches += 1
            else:
                predicted, _ = model.predict(obs, deterministic=True)
                action = int(predicted)

            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if last_action is not None and action == last_action:
                time_in_phase += delta_time
            else:
                time_in_phase = 0.0
            last_action = action

            if env.sumo.simulation.getMinExpectedNumber() == 0:
                cleared = True
                clearance_time = env.sumo.simulation.getTime()
                break

            if terminated or truncated:
                break
    finally:
        env.close()

    return {
        "clearance_time": clearance_time,
        "steps": steps,
        "cleared": cleared,
        "forced_switches": forced_switches,
    }


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
    p.add_argument("--max-green", type=float, default=None,
                    help="Max seconds any phase can stay green before a forced switch (e.g. 44). Omit to disable override.")
    args = p.parse_args()

    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    rou_path = "clearance_test.rou.xml"
    generate_route_file(counts, rou_path)

    print(f"Scenario: {counts}")

    fixed_result = run_fixed_timer_clearance(args.net, rou_path)
    print(f"Fixed-timer: {fixed_result}")

    rl_result = run_rl_clearance(args.net, rou_path, args.model, max_green_seconds=args.max_green)
    print(f"RL agent:    {rl_result}")

    if fixed_result["cleared"] and rl_result["cleared"]:
        diff = fixed_result["clearance_time"] - rl_result["clearance_time"]
        pct = (diff / fixed_result["clearance_time"]) * 100
        print(f"\nRL cleared {diff:.2f}s faster ({pct:.1f}% reduction)" if diff > 0
              else f"\nFixed-timer cleared {-diff:.2f}s faster")