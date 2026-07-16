import os
import sys

os.environ["SUMO_HOME"] = "/usr/share/sumo"

print("Python:", sys.executable)
print("SUMO_HOME:", os.environ.get("SUMO_HOME"))

from sumo_rl import SumoEnvironment

env=SumoEnvironment(
    net_file="simulation/onelast/onelast.net.xml",
    route_file="simulation/onelast/onelast.rou.xml",
    use_gui=True,
    num_seconds=1000,
)

print(env.observation_space)
print(env.action_space)

obs, info = env.reset()

for _ in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(action, reward)