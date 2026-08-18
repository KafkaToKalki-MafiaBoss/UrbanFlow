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

--- Force-hold fix (added after diagnosing starvation-through-the-override) ---
Even with the max_green_seconds override wired in, some scenarios
(mod_w_only_heavy, severe_w_only, deterministic across all 3 v3 seeds)
still never cleared. TraCI-level debug (env.sumo.vehicle.getLaneID/
getWaitingTime/getRoute on every forced switch) showed the stuck
vehicle was NOT in an internal junction lane and NOT involved in any
collision/teleport -- it was a normal approach-lane vehicle (e.g.
veh_W_7 on E0_0, route E0->-E2) legally waiting at a red light, with
waiting time climbing continuously across many forced switches. Root
cause: the original override only forced ONE step onto the corrective
phase before immediately returning control to model.predict() on the
very next decision. If the policy's own behavior pulls it straight
back to the phase that starves this vehicle (observed: action=2 right
after the forced switch), the corrective phase never gets more than a
single delta_time window (8s, partly eaten by yellow_time overhead) to
actually clear it. FORCE_HOLD_STEPS makes the override HOLD the
corrective phase for several consecutive decisions once triggered,
guaranteeing the starved movement gets a real green window before
control reverts to the policy. This is still a pure eval/inference-time
wrapper -- no model or network file changes.

--- Rotation-cursor fix (added after force-hold alone still didn't clear
    mod_w_only_heavy / severe_w_only) ---
With force-hold in place, [STUCK] logs showed veh_W_7 (route E0->-E2,
linkIndex 8, a West-right turn only served by phase 2 / action=1) STILL
never clearing -- wait time climbed the entire 1800s regardless. Cause:
the override computed the forced action as (last_action + 1) % n_actions.
Every time control returned to the policy after a hold, it immediately
re-picked action=2 (confirmed in the [DEBUG] trace). So every subsequent
override trigger started from the same last_action=2 and always forced
(2+1)%4 == 3 -- oscillating between actions 2 and 3 forever, and NEVER
reaching action=1, the one phase that actually serves this vehicle.
Fix: track rotation progress with a cursor that is independent of
last_action / the policy's own choices, so each successive override
visits a genuinely different phase than the last override did, and
within at most n_actions trigger cycles every phase (including the one
the policy keeps avoiding) gets a real, held green window.

--- Biased rotation-order fix (added after tuning max_green_seconds/
    force_hold_steps down to 30/2 revealed the plain rotation_cursor is
    slower than necessary) ---
The 20-scenario batch showed a 100% correlation between forced_switches:1
and RL-slower-than-fixed-timer cases. Every stuck vehicle logged via
[STUCK] was on a through/right movement (never a left), and the policy
almost always fixates on action=2, the protected-left phase, whenever
E/W has any nonzero queue -- root cause is FourApproachQueueObservation
aggregating by approach edge, so the policy can't distinguish a
left-turn queue from a through/right queue on the same edge. The plain
independent rotation_cursor (0->1->2->3->0...) still eventually reaches
the correct phase, but on average wastes extra trigger cycles getting
there depending on where the cursor happens to be sitting when the
override first fires.

ROTATION_ORDER replaces the flat cursor with a per-stuck-action lookup:
whichever action the POLICY was actually trying to hold (not necessarily
what we last forced) picks a specific first guess -- for action=2 this
is action=1, since that's the observed common case (through/right
starved while the policy fixates on the protected-left phase) -- and
the remaining phases follow in a fixed order after that, so a full
cycle of all n_actions - 1 alternative phases still completes before
repeating. This does not weaken the starvation guarantee: every phase
is still visited within one full cycle, it's just reordered so the
statistically common fix comes first instead of last.

Progress through a stuck-action's rotation list is tracked in
rotation_progress, keyed by the action the policy was fixated on
(policy_last_action) rather than by whatever action we most recently
forced -- so if the policy keeps re-picking the same stubborn action
after each hold ends, subsequent overrides continue advancing through
THAT action's order instead of restarting or drifting onto an unrelated
sequence.

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
DEFAULT_MAX_GREEN_SECONDS = 30

# How many consecutive decisions the override HOLDS the corrective phase
# for once triggered, before handing control back to the policy. At
# delta_time=8s, 2 steps ~= 16s of held green (minus yellow overhead),
# enough for a starved approach-lane vehicle to actually depart instead
# of getting bumped back onto the phase that stranded it after just one
# 8s window. Tune upward if a scenario still freezes with this in place.
FORCE_HOLD_STEPS = 2

# Per-stuck-action rotation order. Key = the action the POLICY was
# fixated on (policy_last_action); value = the order in which the
# override tries the OTHER n_actions - 1 phases before repeating.
#
# First entry per key is a biased guess based on observed [STUCK] data
# (action=2 -> try action=1 first, since that's the phase most often
# actually needed). Remaining entries fill out the rest of the cycle in
# a fixed, deterministic order so every phase is still guaranteed to be
# visited within one full pass -- this only changes WHICH phase gets
# tried first, not whether every phase eventually gets tried.
#
# General construction for n_actions=4, stuck action s:
#   order[0] = (s - 1) % 4
#   order[1] = (s + 1) % 4
#   order[2] = (s + 2) % 4
# (s itself is excluded since forcing the action the policy is already
# stuck on wouldn't be a "switch" at all.)
ROTATION_ORDER = {
    0: [3, 1, 2],
    1: [0, 2, 3],
    2: [1, 3, 0],
    3: [2, 0, 1],
}


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
    force_hold_steps: int = FORCE_HOLD_STEPS,
    show_rl_gui=True
) -> dict:
    """
    Runs the same scenario under the trained PPO agent, using env.sumo
    (sumo-rl's own labeled TraCI connection) rather than the bare traci
    module — matches the project's established TraCI-connection lesson.

    max_green_seconds: if set, forces a phase switch once the current
    phase has held for this long, overriding the policy's own choice.
    None (default) preserves the original unmodified-policy behavior.

    force_hold_steps: once a forced switch fires, how many consecutive
    decisions to keep forcing the corrective phase before handing
    control back to the policy. Set to 1 to restore the old
    single-step-only behavior.

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
        use_gui=show_rl_gui,
        additional_sumo_cmd="--delay 150",
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
    force_hold_remaining = 0

    # Tracks what the POLICY itself last predicted, updated only in the
    # model.predict() branch below -- this is what ROTATION_ORDER keys
    # off, since it represents what the policy is actually fixated on,
    # independent of whatever the override has been forcing.
    policy_last_action = None

    # Per-stuck-action progress through ROTATION_ORDER[stuck_action].
    # -1 means "not started yet" for that key so the first trigger for
    # a given stuck action lands on index 0 (the biased first guess).
    rotation_progress = {}

    try:
        while True:
            if force_hold_remaining > 0:
                # Keep holding the phase the override just switched to,
                # instead of immediately returning control to the policy.
                # This is what actually lets a starved vehicle clear --
                # a single forced step wasn't a long enough green window.
                action = last_action
                force_hold_remaining -= 1

            elif (
                max_green_seconds is not None
                and last_action is not None
                and time_in_phase >= max_green_seconds
            ):
                # Confirmed via check_action_space.py: action_space is
                # Discrete(4), NOT Discrete(2) (4 green phases including
                # 2 protected-left phases).
                #
                # Bias the rotation using ROTATION_ORDER, keyed by what
                # the policy itself was actually fixated on
                # (policy_last_action) rather than last_action (which
                # may just be whatever we last forced). Progress through
                # that action's order is tracked independently so a full
                # cycle of every other phase still completes before
                # repeating -- this only reorders which phase gets tried
                # first, it doesn't skip any.
                n_actions = env.action_space.n
                stuck_action = policy_last_action if policy_last_action is not None else last_action
                order = ROTATION_ORDER.get(stuck_action)
                if order is None:
                    # Fallback for any action not in the lookup (e.g. if
                    # n_actions ever changes): plain independent rotation.
                    order = [a for a in range(n_actions) if a != stuck_action]

                idx = rotation_progress.get(stuck_action, -1) + 1
                if idx >= len(order):
                    idx = 0  # full cycle completed for this stuck action, start over
                rotation_progress[stuck_action] = idx

                action = order[idx]
                forced_switches += 1
                force_hold_remaining = force_hold_steps - 1

                # Debug: on every forced switch, dump exactly what's
                # still stuck in the network -- vehicle id, current lane,
                # route, and waiting time. This bypasses SUMO's own
                # collision/teleport detector entirely (which stays
                # silent for a vehicle correctly waiting at a red light)
                # and tells us directly whether it's a routing dead-end
                # vs. a lane-geometry block.
                for veh_id in env.sumo.vehicle.getIDList():
                     lane = env.sumo.vehicle.getLaneID(veh_id)
                     wait = env.sumo.vehicle.getWaitingTime(veh_id)
                     route = env.sumo.vehicle.getRoute(veh_id)
                     print(f"[STUCK] veh={veh_id} lane={lane} wait={wait:.1f}s route={route}")

            else:
                predicted, _ = model.predict(obs, deterministic=True)
                action = int(predicted)
                policy_last_action = action
                print(f"[DEBUG] step={steps} obs={obs} action={action}")

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
    p.add_argument("--net", default="simulation/onelast/onelast.net.xml")
    p.add_argument("--model", default="rl_agent/models/ppo_onelast_v3_seed1.zip")
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--s", type=int, default=10)
    p.add_argument("--e", type=int, default=5)
    p.add_argument("--w", type=int, default=5)
    p.add_argument("--max-green", type=float, default=None,
                    help="Max seconds any phase can stay green before a forced switch (e.g. 44). Omit to disable override.")
    p.add_argument("--force-hold-steps", type=int, default=FORCE_HOLD_STEPS,
                    help="How many consecutive decisions to hold the corrective phase once a forced switch fires.")
    args = p.parse_args()

    counts = {"N": args.n, "S": args.s, "E": args.e, "W": args.w}
    rou_path = "clearance_test.rou.xml"
    generate_route_file(counts, rou_path)

    print(f"Scenario: {counts}")

    fixed_result = run_fixed_timer_clearance(args.net, rou_path)
    print(f"Fixed-timer: {fixed_result}")

    rl_result = run_rl_clearance(
        args.net, rou_path, args.model,
        max_green_seconds=args.max_green,
        force_hold_steps=args.force_hold_steps,
    )
    print(f"RL agent:    {rl_result}")

    if fixed_result["cleared"] and rl_result["cleared"]:
        diff = fixed_result["clearance_time"] - rl_result["clearance_time"]
        pct = (diff / fixed_result["clearance_time"]) * 100
        print(f"\nRL cleared {diff:.2f}s faster ({pct:.1f}% reduction)" if diff > 0
              else f"\nFixed-timer cleared {-diff:.2f}s faster")