# AI Traffic Simulator — Bug Report (onelast network)
 
A complete list of every issue found in `train.py`, `observations.py`, `Stage5_Onelast_Script.py`, `onelast.rou.xml`, `onelast.net.xml`, and `onelast.sumocfg`, with the reason each one happens and the exact fix.
 
---
 
## Symptom A: "The traffic light phase never changes"
 
Three separate issues combine to produce this symptom. The light actually *is* switching (or trying to) — but your diagnostic can't see it, and the environment's default timing silently rejects many of your actions.
 
### Bug A1 — `getPhase()` is the wrong probe (train.py, line 39)
 
**The error:**
 
```python
print(f"Current Phase={traci.trafficlight.getPhase('J1')}")
```
 
This always prints the same number, so it *looks* like the phase never changes.
 
**The reason:** sumo-rl does not control the traffic light with `setPhase()`. Internally, its `TrafficSignal` class calls:
 
```python
self.sumo.trafficlight.setRedYellowGreenState(self.id, phase_state_string)
```
 
`setRedYellowGreenState()` overrides the signal's light-state string directly, bypassing the phase program entirely. SUMO's phase index (what `getPhase()` returns) belongs to the *program*, and since the program is never advanced, the index is frozen — even while the actual lights are changing colors.
 
**The fix:** read the live state string or sumo-rl's own bookkeeping instead:
 
```python
# Option 1: the actual light state (e.g. "rrrrGGGgrrrrGGGg")
print(env.sumo.trafficlight.getRedYellowGreenState('J1'))
 
# Option 2: sumo-rl's current green-phase index (0 or 1 for your network)
print(env.traffic_signals['J1'].green_phase)
```
 
### Bug A2 — default timings silently reject phase changes (train.py, env construction)
 
**The error:** you construct `SumoEnvironment` without timing parameters, so it uses the defaults: `delta_time=5` (agent acts every 5 sim-seconds), `yellow_time=2`, `min_green=5`.
 
**The reason:** sumo-rl's `set_next_phase()` refuses to switch when:
 
```python
time_since_last_phase_change < yellow_time + min_green   # i.e. < 7 seconds
```
 
Since your agent acts every 5 seconds, any action taken 5 seconds after a switch falls inside the 7-second lockout and is **silently ignored** — no error, no warning, the current phase is just held. On top of that, your network has only 2 green phases (see A3), so a random action matches the current phase ~50% of the time and requests no change at all. Combined: in a short 20-step test, visible switches become rare, and A1 hides even the ones that happen.
 
**The fix:** make the decision interval at least as long as the lockout window:
 
```python
env = SumoEnvironment(
    net_file="simulation/onelast/onelast.net.xml",
    route_file="simulation/onelast/onelast.rou.xml",
    use_gui=True,
    num_seconds=1000,
    single_agent=True,
    observation_class=FourApproachQueueObservation,
    delta_time=8,      # >= yellow_time + min_green
    yellow_time=2,
    min_green=5,
)
```
 
(Alternatively, lower `min_green` to 3 and keep `delta_time=5`.)
 
### Bug A3 — misunderstanding of the action space (context, not a code error)
 
Your `tlLogic` in `onelast.net.xml` has 4 phases:
 
```xml
<phase duration="42" state="rrrrGGGgrrrrGGGg"/>   <!-- phase 0: North + South green -->
<phase duration="3"  state="rrrryyyyrrrryyyy"/>   <!-- yellow -->
<phase duration="42" state="GGGgrrrrGGGgrrrr"/>   <!-- phase 2: East + West green -->
<phase duration="3"  state="yyyyrrrryyyyrrrr"/>   <!-- yellow -->
</tlLogic>
```
 
sumo-rl strips the yellow phases (it inserts its own transitions automatically) and exposes only the green phases as actions. So your action space is `Discrete(2)`: action 0 = N+S green, action 1 = E+W green. `getPhase` values 0–3 from the static program are not what actions map to.
 
Also note: the comments in `Stage5_Onelast_Script.py` (lines 228–232) say "Phase 1 = E0 + E1 Green" — that's backwards. In the state string, link indices 0–3 are East (`-E1`), 4–7 are South (`E2`), 8–11 are West (`E0`), 12–15 are North (`-E3`). Phase 0 (`rrrrGGGgrrrrGGGg`) is therefore **North + South** green; phase 2 is East + West. Fix the comments/printout so your baseline report isn't mislabeled.
 
### Bug A4 — using the bare `traci` module alongside the env (train.py, lines 35–40)
 
**The error:** `import traci` inside the loop, then calling `traci.trafficlight...` directly.
 
**The reason:** sumo-rl opens its TraCI connection with a *label* and keeps its own handle in `env.sumo`. The bare `traci` module talks to whatever the "current" connection is. With one env it happens to work; with libsumo enabled or multiple/parallel environments it breaks or reads a stale connection. Also, re-importing inside a loop is dead weight — imports belong at the top of the file.
 
**The fix:** delete the in-loop import and always go through `env.sumo`.
 
### Bug A5 — SUMO_HOME hardcoded to a Linux path (train.py, line 6)
 
**The error:**
 
```python
os.environ["SUMO_HOME"] = "/usr/share/sumo"
```
 
**The reason:** you're on Windows — `/usr/share/sumo` doesn't exist there. This line *overwrites* your valid `SUMO_HOME` (something like `C:\Program Files\SUMO`) with a broken path, which can break binary discovery (`sumolib.checkBinary`) and tools imports. Worse, it's set *after* `import sumo_rl`, so even if the value were right, anything sumo-rl resolved at import time already used the old value. Your Stage 5 script does this correctly (checks the env var and errors out if missing).
 
**The fix:** delete the line. If you ever need to override it (e.g. running the same script on a Linux box), set it *before* any sumo-related import and guard it per-OS.
 
---
 
## Symptom B: "The south lane never gets any cars"
 
### Bug B1 — the south approach has no `<flow>`, only two one-shot `<trip>`s (onelast.rou.xml)
 
**The error:** traffic per approach in your route file:
 
| Approach | Edge | Generators | Result |
|---|---|---|---|
| East | `-E1` | flows `f_0` (500/h) + `f_1` (350/h) | continuous traffic |
| North | `-E3` | flow `f_2` (100/h) + trips `t_2`,`t_3` | continuous traffic |
| West | `E0` | flows `f_3` (50/h) + `f_4` (200/h bikes) | continuous traffic |
| **South** | **`E2`** | **only trips `t_0`, `t_1`, both `depart="0.00"`** | **2 cars total, then nothing** |
 
**The reason:** a `<trip>` spawns exactly **one** vehicle at its `depart` time; a `<flow>` spawns vehicles repeatedly between `begin` and `end`. The south approach only has two single trips, both departing in the first second. Once those two vehicles clear the junction (within the first minute), the south approach is empty for the rest of the simulation — hence your South observation reads 0 forever and the GUI shows an empty lane.
 
**The fix:** add continuous flows from `E2`:
 
```xml
<flow id="f_5" type="DEFAULT_TAXITYPE" begin="0.00" end="3600.00"
      perHour="300.00" from="E2" to="E3"/>
<flow id="f_6" type="DEFAULT_TAXITYPE" begin="0.00" end="3600.00"
      perHour="100.00" from="E2" to="-E0"/>
```
 
You can keep or delete `t_0`/`t_1` — they're harmless either way. (Note this was NOT an edge-direction bug: the net file confirms `E2` runs J3→J1, i.e. it really is the incoming south edge, so `observations.py` and Stage 5 use the correct IDs.)
 
### Bug B2 — Stage 5's bare `except` would have hidden B1-style errors (Stage5_Onelast_Script.py, lines 80–91)
 
**The error:**
 
```python
try:
    E0_queue = traci.edge.getLastStepHaltingNumber("E0")
    ...
except:
    E0_queue = 0; E1_queue = 0; E2_queue = 0; E3_queue = 0
```
 
**The reason:** if *any* edge ID were wrong, TraCI would raise, and this bare `except` would silently zero **all four** queues at every step — the database would fill with zeros and you'd never see the error. Silent exception swallowing is why the south problem was hard to diagnose: a permanently-zero column looks identical to "wrong edge name" and "no traffic."
 
**The fix:** remove the try/except (or at minimum log the exception). Correct edge IDs never raise, so there's nothing legitimate to catch:
 
```python
E0_queue = traci.edge.getLastStepHaltingNumber("E0")    # West in
E1_queue = traci.edge.getLastStepHaltingNumber("-E1")   # East in
E2_queue = traci.edge.getLastStepHaltingNumber("E2")    # South in
E3_queue = traci.edge.getLastStepHaltingNumber("-E3")   # North in
```
 
---
 
## Minor issues (won't cause your two symptoms, but worth fixing)
 
1. **`onelast.sumocfg` is never used by train.py.** `SumoEnvironment` builds its own sumo command from `net_file` + `route_file`, so your `step-length 0.05` and GUI `delay` there have no effect on the RL run. Only Stage 5 (which passes `-c onelast.sumocfg`) uses it. Reason to care: don't be surprised when RL runs use a 1 s step length.
 
2. **Deprecated TraCI call** (Stage5, line 76): `traci.simulation.getCurrentTime()/1000` → use `traci.simulation.getTime()`, which returns seconds directly.
 
3. **Relative paths in train.py** (`simulation/onelast/onelast.net.xml`): these resolve from the directory you *launch* from, not the script's location. If your files actually live in `simulation/` (not `simulation/onelast/`), or you run from another folder, you'll get a file-not-found. Safer: build paths from `os.path.dirname(os.path.abspath(__file__))` like Stage 5 does.
 
4. **evaluate.py is empty.** The uploaded file had no content — if it's supposed to contain the evaluation loop, it still needs to be written (or re-check what you saved).
 
---
 
## Quick checklist
 
- [ ] rou.xml: add `<flow>`s from `E2` (south)
- [ ] train.py: delete `os.environ["SUMO_HOME"] = "/usr/share/sumo"`
- [ ] train.py: pass `delta_time=8` (or `min_green=3`) to `SumoEnvironment`
- [ ] train.py: replace `getPhase` print with `getRedYellowGreenState` / `env.traffic_signals['J1'].green_phase`
- [ ] train.py: use `env.sumo`, move `import traci` out of the loop (or drop it)
- [ ] Stage 5: remove bare `except` around edge queries
- [ ] Stage 5: fix phase-labeling comments (phase 0 = N+S green, phase 2 = E+W green)
- [ ] Stage 5: `getCurrentTime()/1000` → `getTime()`
 