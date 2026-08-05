"""
test_generate_scenario.py

Phase A, step 3 -- loads a generated scenario into onelast.net.xml via
TraCI, runs it, and confirms exactly N vehicles departed per approach.

Usage (run from anywhere -- --net resolves relative to cwd first, then
falls back to this script's own directory):
    python test_generate_scenario.py --net onelast.net.xml --n 10 --s 10 --e 5 --w 5
    python simulation/onelast/test_generate_scenario.py --net onelast.net.xml --n 10 --s 10 --e 5 --w 5
"""

import argparse
import os
import sys
import tempfile

from generate_scenario import generate_route_file, INCOMING_EDGE, APPROACH_ORDER

try:
    import traci
    import sumolib
except ImportError:
    sys.exit("Needs SUMO's Python bindings. Set SUMO_HOME and add "
              "$SUMO_HOME/tools to PYTHONPATH.")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_net_path(net_arg: str) -> str:
    """
    Resolve --net against, in order:
      1. as given (absolute, or relative to current working directory)
      2. relative to this script's own directory (simulation/onelast/)
    Fails loudly with both attempted paths if neither exists, instead of
    letting SUMO retry-then-die with a cryptic TraCI connection error.
    """
    candidates = [
        os.path.abspath(net_arg),
        os.path.join(SCRIPT_DIR, net_arg),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    sys.exit(
        "Net file not found. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + f"\n(cwd={os.getcwd()}, script dir={SCRIPT_DIR})"
    )


def run_check(net_path: str, counts: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        rou_path = os.path.join(tmpdir, "onelast.rou.xml")
        generate_route_file(counts, rou_path)

        sumo_binary = sumolib.checkBinary("sumo")  # headless
        cfg = [
            sumo_binary,
            "--net-file", net_path,
            "--route-files", rou_path,
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]
        traci.start(cfg)

        departed_by_approach = {a: 0 for a in APPROACH_ORDER}
        edge_to_approach = {edge: a for a, edge in INCOMING_EDGE.items()}
        seen = set()

        try:
            while traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                for veh_id in traci.simulation.getDepartedIDList():
                    if veh_id in seen:
                        continue
                    seen.add(veh_id)
                    first_edge = traci.vehicle.getRoute(veh_id)[0]
                    approach = edge_to_approach.get(first_edge)
                    if approach is None:
                        raise AssertionError(
                            f"Vehicle {veh_id} departed on unrecognized edge {first_edge}"
                        )
                    departed_by_approach[approach] += 1
        finally:
            traci.close()

        print("Expected:", counts)
        print("Observed:", departed_by_approach)

        ok = all(departed_by_approach.get(a, 0) == counts.get(a, 0) for a in APPROACH_ORDER)
        if ok:
            print("PASS: exact vehicle counts confirmed for all approaches.")
        else:
            print("FAIL: counts did not match.")
            sys.exit(1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="onelast.net.xml", help="Path to onelast.net.xml")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--e", type=int, default=5)
    p.add_argument("--w", type=int, default=5)
    args = p.parse_args()

    resolved_net = resolve_net_path(args.net)
    run_check(resolved_net, {"N": args.n, "S": args.s, "E": args.e, "W": args.w})