"""
check_action_space.py

Confirms what train.py / clearance_metrics.py have been silently
assuming. Run this once, from simulation/onelast/, before trusting any
override logic that hardcodes "2 phases".
"""
import os
import sys

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    if tools not in sys.path:
        sys.path.append(tools)
else:
    sys.exit("ERROR: Please set SUMO_HOME environment variable")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(BASE_DIR)

from sumo_rl import SumoEnvironment
from rl_agent.observations import FourApproachQueueObservation

env = SumoEnvironment(
    net_file=os.path.join(BASE_DIR, "simulation", "onelast", "onelast.net.xml"),
    route_file=os.path.join(BASE_DIR, "simulation", "onelast", "onelast.rou.xml"),
    use_gui=False,
    num_seconds=100,
    single_agent=True,
    observation_class=FourApproachQueueObservation,
    delta_time=8,
    yellow_time=2,
    min_green=5,
)

print("action_space:", env.action_space)
ts_id = env.ts_ids[0]
ts = env.traffic_signals[ts_id]
print("ts_id:", ts_id)

# Attribute name varies slightly by sumo-rl version; try the common ones.
for attr in ("green_phases", "all_phases", "phases", "num_green_phases"):
    if hasattr(ts, attr):
        print(f"ts.{attr}:", getattr(ts, attr))

env.close()