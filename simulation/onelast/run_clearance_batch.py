"""
run_clearance_batch.py

Phase B extension — runs a batch of imbalanced/heavier scenarios through
both controllers (fixed-timer, RL agent) and logs results to CSV for
reporting. Designed to answer: "does RL's clearance-time advantage grow
as approach-load imbalance grows?" rather than relying on a single
balanced 10/10/5/5 data point.

Run from simulation/onelast/ (or adjust --net path):
    python run_clearance_batch.py --net onelast.net.xml --model ../../rl_agent/models/ppo_onelast_v2.zip

NOTE: v2 replaces v1 as the default model here. v1 was later discovered
to have been trained with an accidentally-restricted Discrete(2) action
space (see check_action_space.py / diagnose_rl_gridlock.py findings),
while the real environment is Discrete(4). v2 is a fresh model trained
correctly against Discrete(4), plus a clearance curriculum targeting
the starvation gridlock found in the v1 batch run. Results from this
run are NOT directly comparable to the original v1 CSV -- different
action space entirely -- but are directly comparable to each other
across scenarios, and are the correct numbers to report going forward.

Outputs:
    clearance_batch_results_v2.csv  -- one row per scenario, both controllers
"""

import argparse
import csv
import os
import sys

# Add the project root to the Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

from generate_scenario import generate_route_file
from clearance_metrics import run_fixed_timer_clearance, run_rl_clearance

# ---------------------------------------------------------------------------
# 20 scenarios: mix of balanced (control group) and increasingly imbalanced
# / heavier loads, up to the ~13/approach burst-window ceiling.
# Format: (label, N, S, E, W)
# ---------------------------------------------------------------------------
SCENARIOS = [
    # -- Balanced baseline (light + heavy) --
    ("balanced_light",      3,  3,  3,  3),
    ("balanced_medium",     7,  7,  7,  7),
    ("balanced_heavy",      10, 10, 10, 10),
    ("balanced_max",        13, 13, 13, 13),

    # -- Mild imbalance (2x ratio) --
    ("mild_ns_heavy",       8,  8,  4,  4),
    ("mild_ew_heavy",       4,  4,  8,  8),
    ("mild_n_heavy",        10, 5,  5,  5),
    ("mild_e_heavy",        5,  5,  10, 5),

    # -- Moderate imbalance (4-5x ratio) --
    ("mod_ns_heavy",        10, 10, 2,  2),
    ("mod_ew_heavy",        2,  2,  10, 10),
    ("mod_n_only_heavy",    12, 3,  3,  3),
    ("mod_w_only_heavy",    3,  3,  3,  12),

    # -- Severe imbalance (near max ceiling one side, near-empty other) --
    ("severe_ns_heavy",     13, 13, 1,  1),
    ("severe_ew_heavy",     1,  1,  13, 13),
    ("severe_n_only",       13, 1,  1,  1),
    ("severe_s_only",       1,  13, 1,  1),
    ("severe_e_only",       1,  1,  13, 1),
    ("severe_w_only",       1,  1,  1,  13),

    # -- Asymmetric mixed loads (no clean symmetry, realistic-ish) --
    ("mixed_1",              13, 6,  9,  2),
    ("mixed_2",              2,  9,  6,  13),
]


def run_batch(net_path: str, model_path: str, out_csv: str):
    rows = []

    for label, n, s, e, w in SCENARIOS:
        counts = {"N": n, "S": s, "E": e, "W": w}
        rou_path = f"_batch_scenario_{label}.rou.xml"
        generate_route_file(counts, rou_path)

        print(f"\n=== {label}  N={n} S={s} E={e} W={w} ===")

        try:
            fixed_result = run_fixed_timer_clearance(net_path, rou_path)
        except Exception as ex:
            print(f"  Fixed-timer FAILED: {ex}")
            fixed_result = {"clearance_time": None, "steps": None, "cleared": False}

        try:
            rl_result = run_rl_clearance(net_path, rou_path, model_path, max_green_seconds=30)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print(f"  RL agent FAILED: {ex}")
            rl_result = {"clearance_time": None, "steps": None, "cleared": False}

        print(f"  Fixed-timer: {fixed_result}")
        print(f"  RL agent:    {rl_result}")

        row = {
            "label": label,
            "N": n, "S": s, "E": e, "W": w,
            "total_vehicles": n + s + e + w,
            "max_min_ratio": max(n, s, e, w) / max(min(n, s, e, w), 1),
            "fixed_clearance_s": fixed_result["clearance_time"],
            "fixed_cleared": fixed_result["cleared"],
            "rl_clearance_s": rl_result["clearance_time"],
            "rl_cleared": rl_result["cleared"],
        }

        if fixed_result["cleared"] and rl_result["cleared"]:
            diff = fixed_result["clearance_time"] - rl_result["clearance_time"]
            pct = (diff / fixed_result["clearance_time"]) * 100
            row["diff_s"] = round(diff, 2)
            row["pct_improvement"] = round(pct, 2)
        else:
            row["diff_s"] = None
            row["pct_improvement"] = None

        rows.append(row)

        # Clean up the temp route file for this scenario.
        try:
            os.remove(rou_path)
        except OSError:
            pass

    fieldnames = [
        "label", "N", "S", "E", "W", "total_vehicles", "max_min_ratio",
        "fixed_clearance_s", "fixed_cleared",
        "rl_clearance_s", "rl_cleared",
        "diff_s", "pct_improvement",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n\nWrote {len(rows)} results to {out_csv}")

    # Quick summary printed to console.
    valid = [r for r in rows if r["pct_improvement"] is not None]
    if valid:
        avg_pct = sum(r["pct_improvement"] for r in valid) / len(valid)
        best = max(valid, key=lambda r: r["pct_improvement"])
        worst = min(valid, key=lambda r: r["pct_improvement"])
        print(f"\nSummary across {len(valid)} valid scenarios:")
        print(f"  Average improvement: {avg_pct:.2f}%")
        print(f"  Best:  {best['label']} ({best['pct_improvement']:.2f}%, ratio={best['max_min_ratio']:.1f}x)")
        print(f"  Worst: {worst['label']} ({worst['pct_improvement']:.2f}%, ratio={worst['max_min_ratio']:.1f}x)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--net", default="simulation/onelast/onelast.net.xml")
    p.add_argument("--model", default="../../rl_agent/models/ppo_onelast_v2.zip")
    p.add_argument("--out", default="clearance_batch_results_v2.csv")
    args = p.parse_args()
    run_batch(args.net, args.model, args.out)