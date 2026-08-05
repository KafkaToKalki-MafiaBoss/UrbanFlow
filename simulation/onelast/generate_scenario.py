"""
generate_scenario.py

Phase A, step 2 — deterministically generates a `.rou.xml` for
`onelast.net.xml` given exact vehicle counts per approach (N/S/E/W).

Design (per Option 2 decisions):
  - Deterministic: same 4 input counts -> byte-identical route file every
    time, so the fixed-timer run and the RL run see the exact same
    vehicles, on the exact same routes, departing at the exact same times.
    This is what makes the clearance-time comparison (Phase B) fair.
  - Exact counts: N cars requested from an approach = exactly N <vehicle>
    entries whose route starts on that approach's incoming edge.
  - Staggered burst arrival: all vehicles across all approaches depart
    within the first ~30 simulated seconds, spaced BURST_SPACING seconds
    apart *within* each approach's own stream. This avoids SUMO insertion
    conflicts (two vehicles trying to occupy the same lane position at the
    same instant) while still representing "here's a batch of cars
    waiting at each approach right now" rather than a hyper-realistic
    long flow window.

Network facts (from onelast.net.xml, junction J1), verified against the
<connection> elements so every route below is guaranteed routable:

    Approach | incoming edge | outgoing edge (exit that direction)
    ---------|----------------|-------------------------------------
    North    | -E3            | E3
    South    | E2             | -E2
    East     | -E1            | E1
    West     | E0             | -E0

Every incoming edge at J1 connects to all three other outgoing edges
(confirmed via the <connection from=... to=...> entries), so any
approach -> any other approach's exit edge is a valid two-edge route.

Usage:
    python generate_scenario.py --n 10 --s 10 --e 5 --w 5 --out onelast_custom.rou.xml

    or programmatically:
        from generate_scenario import generate_route_file
        generate_route_file({"N": 10, "S": 10, "E": 5, "W": 5}, "onelast_custom.rou.xml")
"""

import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ---------------------------------------------------------------------------
# Network configuration — matches onelast.net.xml (junction J1) exactly.
# ---------------------------------------------------------------------------
INCOMING_EDGE = {
    "N": "-E3",
    "S": "E2",
    "E": "-E1",
    "W": "E0",
}

# The edge you exit ONTO when leaving the junction heading in each
# direction. E.g. any vehicle exiting north leaves via edge "E3".
EXIT_EDGE = {
    "N": "E3",
    "S": "-E2",
    "E": "E1",
    "W": "-E0",
}

APPROACH_ORDER = ["N", "S", "E", "W"]  # fixed iteration/tie-break order

# Every approach can legally reach every other approach's exit edge at J1
# (straight/left/right are all signalized and connected) — confirmed
# against the net.xml <connection> list, so no per-approach restriction
# is needed here.
VALID_DESTINATIONS = {a: [d for d in APPROACH_ORDER if d != a] for a in APPROACH_ORDER}

# --- Staggered burst arrival parameters -------------------------------------
# All vehicles depart within roughly this many seconds of sim time...
BURST_WINDOW_SECONDS = 30.0
# ...spaced this many seconds apart within a single approach's own stream,
# to avoid SUMO insertion conflicts (can't place two vehicles on the same
# lane at the same instant/position).
BURST_SPACING_SECONDS = 2.5

VEHICLE_TYPE_ID = "car"
VEHICLE_LENGTH = 5.0
VEHICLE_MAX_SPEED = 13.89  # matches onelast.net.xml lane speeds (13.89 m/s)


def _max_vehicles_per_approach_in_window() -> int:
    """How many vehicles fit in BURST_WINDOW_SECONDS at BURST_SPACING_SECONDS."""
    return int(BURST_WINDOW_SECONDS // BURST_SPACING_SECONDS) + 1


def _build_entries(counts: dict) -> list:
    """
    Deterministically build (id, approach, depart, edges) for every vehicle.
    Destinations are assigned round-robin per approach (not random) so a
    given `counts` dict always produces the same output.
    """
    max_fit = _max_vehicles_per_approach_in_window()
    for approach, n in counts.items():
        if n > max_fit:
            raise ValueError(
                f"{approach} requests {n} vehicles, but only {max_fit} fit in the "
                f"{BURST_WINDOW_SECONDS:.0f}s burst window at "
                f"{BURST_SPACING_SECONDS:.1f}s spacing. Increase BURST_WINDOW_SECONDS "
                f"or decrease BURST_SPACING_SECONDS if you need more."
            )

    dest_cursor = {a: 0 for a in APPROACH_ORDER}
    entries = []

    for approach in APPROACH_ORDER:
        n = counts.get(approach, 0)
        dests = VALID_DESTINATIONS[approach]
        for i in range(n):
            dest = dests[dest_cursor[approach] % len(dests)]
            dest_cursor[approach] += 1

            depart = i * BURST_SPACING_SECONDS
            edges = [INCOMING_EDGE[approach], EXIT_EDGE[dest]]

            entries.append({
                "id": f"veh_{approach}_{i}",
                "approach": approach,
                "depart": depart,
                "edges": edges,
            })

    # SUMO requires non-decreasing depart order in the file. Sort by depart
    # time, tie-broken by fixed N,S,E,W order for full determinism.
    approach_rank = {a: i for i, a in enumerate(APPROACH_ORDER)}
    entries.sort(key=lambda e: (e["depart"], approach_rank[e["approach"]]))
    return entries


def generate_route_file(counts: dict, out_path: str) -> str:
    """
    counts: dict like {"N": 10, "S": 10, "E": 5, "W": 5}
    out_path: file path to write the .rou.xml to
    Returns out_path. Raises ValueError on bad input.
    """
    for key in counts:
        if key not in INCOMING_EDGE:
            raise ValueError(f"Unknown approach '{key}', expected one of {APPROACH_ORDER}")
    for key, val in counts.items():
        if not isinstance(val, int) or val < 0:
            raise ValueError(f"Count for '{key}' must be a non-negative int, got {val!r}")

    entries = _build_entries(counts)

    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd")

    vtype = ET.SubElement(root, "vType")
    vtype.set("id", VEHICLE_TYPE_ID)
    vtype.set("length", str(VEHICLE_LENGTH))
    vtype.set("maxSpeed", str(VEHICLE_MAX_SPEED))

    for e in entries:
        vehicle = ET.SubElement(root, "vehicle")
        vehicle.set("id", e["id"])
        vehicle.set("type", VEHICLE_TYPE_ID)
        vehicle.set("depart", f'{e["depart"]:.2f}')
        route = ET.SubElement(vehicle, "route")
        route.set("edges", " ".join(e["edges"]))

    rough = ET.tostring(root, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="    ")
    pretty = "\n".join(line for line in pretty.split("\n") if line.strip())

    with open(out_path, "w") as f:
        f.write(pretty)

    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description="Generate a deterministic onelast.net.xml-compatible .rou.xml from per-approach vehicle counts.")
    p.add_argument("--n", type=int, default=0, help="Vehicles from North (-E3)")
    p.add_argument("--s", type=int, default=0, help="Vehicles from South (E2)")
    p.add_argument("--e", type=int, default=0, help="Vehicles from East (-E1)")
    p.add_argument("--w", type=int, default=0, help="Vehicles from West (E0)")
    p.add_argument("--out", type=str, default="onelast_custom.rou.xml", help="Output .rou.xml path")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    path = generate_route_file(counts, args.out)
    print(f"Wrote {sum(counts.values())} vehicles ({counts}) to {path}")