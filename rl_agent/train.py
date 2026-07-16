import os
import sys

import sumo_rl

os.environ["SUMO_HOME"] = "/usr/share/sumo"

print("Python:", sys.executable)
print("SUMO_HOME:", os.environ.get("SUMO_HOME"))

from sumo_rl import SumoEnvironment
print(sumo_rl.__file__)

from observations import FourApproachQueueObservation

env = SumoEnvironment(
    net_file="simulation/onelast/onelast.net.xml",
    route_file="simulation/onelast/onelast.rou.xml",
    use_gui=True,
    num_seconds=1000,
    single_agent=True,
    observation_class=FourApproachQueueObservation,
)

print("Controlled TLS ids:", env.ts_ids)

print(env.observation_space)
print(env.action_space)

obs, info = env.reset()

for _ in range(20):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    import traci

    print(
        f"Action={action}, "
        f"Current Phase={traci.trafficlight.getPhase('J1')}"
    )
    print(
        f"action={action}, reward={reward:.3f}, "
        f"obs[N,S,E,W]={obs}, (sim_step={env.sim_step:.0f}s)")
    

env.close()
