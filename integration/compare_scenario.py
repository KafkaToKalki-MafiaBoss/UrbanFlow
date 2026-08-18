"""
compare_scenario.py

Phase C — orchestration. Wraps clearance_metrics.py's two runner
functions into ONE function that takes a user's {N,S,E,W} input and
returns a single combined result: exactly what Phase D's
backend/dashboard needs to call.

Both controllers are run against the SAME generated route file (same
vehicles, same routes, same depart times) -- this is what makes the
comparison fair, per the Phase A/B design.

Uses ppo_onelast_v3_seed3.zip by default (the curriculum-retrained model,
correct Discrete(4) action space, validated 20/20 clearance on the
imbalanced-scenario batch -- see clearance_batch_results_v3.csv).

Includes the max-green override as a safety net for the live demo,
even though v3 needed zero forced switches across all 20 test
scenarios. Cheap insurance: if a user-entered combination outside the
tested range ever causes a stall, the demo still completes instead of
hanging in front of a mentor.

--- steps naming fix ---
fixed_timer's "steps" and rl_agent's "steps" were NOT the same unit:
fixed_timer.steps counts raw SUMO simulation steps (step_length=0.05s
each, so a 60s clearance is ~1200 steps), while rl_agent.steps counts
RL decision steps (one per delta_time=8s agent action, so the same 60s
clearance is ~7-8 steps). Displaying both under an identical "steps"
key invites a direct, meaningless comparison (e.g. "1173 vs 7" reads
like the RL agent is ~170x more efficient, which is not a real claim
this project makes -- it's purely a step-length artifact). Renamed to
sim_steps / decision_steps in the merged result so the units are
explicit and the two numbers are never implicitly compared. Neither
number is omitted -- both are still fully available under their own
controller's dict, just clearly labeled.

Run from anywhere; paths are anchored to this script's location.
Place in integration/.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "simulation", "onelast"))

from simulation.onelast.generate_scenario import generate_route_file
from simulation.onelast.clearance_metrics import (
    run_fixed_timer_clearance,
    run_rl_clearance,
    DEFAULT_MAX_GREEN_SECONDS,
)

DEFAULT_NET_PATH = os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.net.xml")
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "rl_agent", "models", "ppo_onelast_v3_seed3.zip")


def _relabel_steps(result: dict, key_name: str) -> dict:
    """
    Returns a shallow copy of a clearance_metrics.py result dict with
    "steps" renamed to key_name. Doesn't mutate the original dict, so
    callers that want the raw run_fixed_timer_clearance/run_rl_clearance
    shape untouched (e.g. clearance_metrics.py's own __main__ block)
    are unaffected -- this relabeling is local to compare_scenario.py's
    merged output only.
    """
    relabeled = dict(result)
    relabeled[key_name] = relabeled.pop("steps")
    return relabeled


def run_scenario_comparison(
    counts: dict,
    net_path: str = DEFAULT_NET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    max_green_seconds: float = DEFAULT_MAX_GREEN_SECONDS,
    show_rl_gui: bool = True
) -> dict:
    """
    counts: {"N": int, "S": int, "E": int, "W": int} -- user input
    Returns a single dict combining both controllers' results, ready to
    hand to a dashboard/API layer as-is.
    """
    rou_path = os.path.join(PROJECT_ROOT, "simulation", "onelast", "_demo_scenario.rou.xml")
    generate_route_file(counts, rou_path)

    try:
        fixed_result = run_fixed_timer_clearance(net_path, rou_path)
        rl_result = run_rl_clearance(
            net_path, rou_path, model_path, max_green_seconds=max_green_seconds, show_rl_gui=show_rl_gui
        )
    finally:
        try:
            os.remove(rou_path)
        except OSError:
            pass

    # Relabel each controller's "steps" field to make the unit explicit
    # -- fixed_timer's is raw sim steps (step_length=0.05s), rl_agent's
    # is decision steps (delta_time=8s). Both values are still present,
    # just under names that can't be mistaken for the same unit.
    fixed_result = _relabel_steps(fixed_result, "sim_steps")
    rl_result = _relabel_steps(rl_result, "decision_steps")

    result = {
        "input_counts": counts,
        "total_vehicles": sum(counts.values()),
        "fixed_timer": fixed_result,
        "rl_agent": rl_result,
        "clearance_time_diff_s": None,
        "pct_improvement": None,
    }

    if fixed_result["cleared"] and rl_result["cleared"]:
        diff = fixed_result["clearance_time"] - rl_result["clearance_time"]
        pct = (diff / fixed_result["clearance_time"]) * 100
        result["clearance_time_diff_s"] = round(diff, 2)
        result["pct_improvement"] = round(pct, 2)

    return result


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Phase C manual test: run one scenario through both controllers.")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--e", type=int, default=5)
    p.add_argument("--w", type=int, default=5)
    p.add_argument("--max-green", type=float, default=DEFAULT_MAX_GREEN_SECONDS,
                    help=f"Max green seconds before forced switch. Default {DEFAULT_MAX_GREEN_SECONDS}. Pass a large number to effectively disable.")
    args = p.parse_args()

    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    result = run_scenario_comparison(counts, max_green_seconds=args.max_green)

    print(json.dumps(result, indent=2))