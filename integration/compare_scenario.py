"""
compare_scenario.py

Phase C — orchestration. Wraps clearance_metrics.py's two runner
functions into ONE function that takes a user's {N,S,E,W} input and
returns a single combined result: exactly what Phase D's
backend/dashboard needs to call.

Both controllers are run against the SAME generated route file (same
vehicles, same routes, same depart times) -- this is what makes the
comparison fair, per the Phase A/B design.

Uses ppo_onelast_v2.zip by default (the curriculum-retrained model,
correct Discrete(4) action space, validated 20/20 clearance on the
imbalanced-scenario batch -- see clearance_batch_results_v2.csv).

Includes the max-green override as a safety net for the live demo,
even though v2 needed zero forced switches across all 20 test
scenarios. Cheap insurance: if a user-entered combination outside the
tested range ever causes a stall, the demo still completes instead of
hanging in front of a mentor.

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
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "rl_agent", "models", "ppo_onelast_v2.zip")


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