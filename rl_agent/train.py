import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Fix A5: do NOT hardcode SUMO_HOME (the old line overwrote the valid Windows
# path with a Linux one, and did so after sumo_rl was already imported).
# Rely on the system environment variable instead; fail fast if it's missing.
if "SUMO_HOME" not in os.environ:
    print("ERROR: Please set the SUMO_HOME environment variable")
    print("Windows: setx SUMO_HOME \"C:\\Program Files (x86)\\Eclipse\\Sumo\"")
    sys.exit(1)

import sumo_rl
from sumo_rl import SumoEnvironment

from observations import FourApproachQueueObservation

print("Python:", sys.executable)
print("SUMO_HOME:", os.environ.get("SUMO_HOME"))
print("sumo_rl:", sumo_rl.__file__)

# Fix (minor #3): resolve paths relative to this script, not the launch cwd.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from stable_baselines3.common.callbacks import CheckpointCallback

env = SumoEnvironment(
    net_file=os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.net.xml"),
    route_file=os.path.join(PROJECT_ROOT, "simulation", "onelast", "onelast.rou.xml"),
    use_gui=False,
    num_seconds=3600,
    single_agent=True,
    observation_class=FourApproachQueueObservation,
    # Fix A2: decision interval must be >= yellow_time + min_green (2 + 5 = 7),
    # otherwise phase-change requests are silently rejected by sumo-rl.
    delta_time=8,
    yellow_time=2,
    min_green=5,
)
env = Monitor(env)

model_dir = os.path.join(PROJECT_ROOT, "rl_agent", "models")
os.makedirs(model_dir, exist_ok=True)

checkpoint_dir = os.path.join(PROJECT_ROOT, "rl_agent", "checkpoints")
log_dir = os.path.join(PROJECT_ROOT, "rl_agent", "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=checkpoint_dir, name_prefix="ppo_onelast")
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=20_000, callback=checkpoint_callback)
model.save(os.path.join(model_dir, "ppo_onelast_smoketest"))

# ts_id = env.ts_ids[0]  # "J1"
# print("Controlled TLS ids:", env.ts_ids)
# print("Observation space:", env.observation_space)
# print("Action space:", env.action_space)  # Discrete(2): 0 = N+S green, 1 = E+W green

# obs, info = env.reset()

# for _ in range(100):
#     action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(action)

#     # Fix A1 + A4: sumo-rl switches lights via setRedYellowGreenState(), so
#     # getPhase() never changes. Read the live state string / sumo-rl's own
#     # green_phase instead, and always go through env.sumo (labeled connection),
#     # never the bare `traci` module.
#     # state_str = env.sumo.trafficlight.getRedYellowGreenState(ts_id)
#     # green_phase = env.traffic_signals[ts_id].green_phase

#     # print(traci.edge.getLastStepVehicleNumber("E2"))  # DEBUG: test traci import for South Lane


#     # print(
#     #     f"Action={action}, GreenPhase={green_phase}, State={state_str}"
#     # )
#     print(
#         f"action={action}, reward={reward:.3f}, "
#         f"obs[N,S,E,W]={obs}, (sim_step={env.sim_step:.0f}s)"
#     )

obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"action={action}, reward={reward:.3f}, obs[N,S,E,W]={obs}")
    if terminated or truncated:
        break

env.close()