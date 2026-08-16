1) sequential looping of all lanes before repeating a phase



2) Good catch — and this is worth thinking through carefully, because you're right that a judge would immediately ask "why 81% here but only ~7% there?" if you present both numbers side by side without framing them.

**Why the numbers genuinely aren't comparable — not just due to the 8s rounding caveat:**

They're measuring two structurally different things, under two structurally different traffic conditions:

1. **What's being measured is different.** `evaluate.py`'s 81% is a reduction in *average queue length sampled continuously* over a long, steady simulation. `clearance_metrics.py`'s 4.25s is a reduction in *total time to zero vehicles* for one small burst. Average-queue-length and total-clearance-duration are not the same quantity, and a big improvement in one doesn't imply a proportionally big improvement in the other — they're different lenses on "better," not two measurements of the same effect.

2. **The traffic conditions are wildly different in scale, and that matters a lot.** The 81% run used ~600 veh/hr per approach *sustained* over a long simulation — real, compounding congestion where a fixed 42/42 split repeatedly wastes green time on light directions while a heavy direction backs up, cycle after cycle. Your clearance-time test used **30 total vehicles** in a single ~30s burst. That's not enough traffic to ever really stress a 90-second fixed cycle — the fixed-timer barely needs to get through one or two phase cycles before everything's gone. There's almost no room for a bad fixed-timer decision to compound, so there's very little advantage for RL to *have*. You're comparing "RL's edge over hours of sustained congestion" against "RL's edge clearing a light one-off burst" — of course the second number is much smaller. It's not a weaker result, it's a different (and easier) scenario for the baseline to handle even without being smart.

**What this means for your demo, concretely:**

The single-burst clearance-time metric, at these small counts, is not going to reproduce the 81% story — and it shouldn't be presented as if it's trying to. If you want the clearance-time demo to actually *show* RL's adaptive advantage (rather than just confirm it's not worse), you need scenarios that create real imbalance and real backlog, closer to what stressed the fixed-timer in the first place:

- Push toward the burst-window capacity ceiling (~13/approach) with **heavy imbalance** — e.g. `--n 13 --s 13 --e 1 --w 1` — so one direction pair is genuinely overloaded relative to the fixed 50/50 split.
- Run several such imbalanced scenarios and report a *range or average* clearance-time improvement, rather than one 10/10/5/5 data point — one lightly-loaded balanced case is the least flattering (and least informative) input you could have picked for a first test.

**For the writeup/defense**, frame these as two separate, honestly-scoped experiments rather than "two versions of the same headline stat":
- *evaluate.py*: "Under sustained training-distribution traffic, RL reduces average queue length by 81% vs. fixed-timer."
- *clearance_metrics.py demo*: "For arbitrary user-specified traffic bursts, RL clears intersections at least as fast as fixed-timer, with the advantage growing as approach loads become more imbalanced." (True only once you've actually shown it growing with imbalance — worth confirming before claiming it.)




3) This data is genuinely important — and it's telling you something different (and more useful) than "does the advantage grow with imbalance." Let's look at what actually happened before deciding how to report it.

**The trend is not what we hypothesized — it's the opposite, and it's unstable.**

Look at the balanced-only rows, ignoring approach direction entirely:

| Scenario | Vehicles | Improvement |
|---|---|---|
| balanced_light (3/3/3/3) | 12 | +45.6% |
| balanced_medium (7/7/7/7) | 28 | +15.9% |
| balanced_heavy (10/10/10/10) | 40 | +0.5% |
| balanced_max (13/13/13/13) | 52 | **−15.9%** |

RL's advantage *shrinks and reverses* as load increases — even with zero directional imbalance. That already contradicts the "imbalance helps RL" hypothesis at its root.

Then look at the severe tier, which should be RL's best case if the hypothesis were right: results range from **+27.3%** (severe_ew_heavy) down to **−39.9%** (severe_s_only) — two scenarios with the *same* 13x ratio, opposite sign, ~67 percentage points apart. And three scenarios (`mod_ew_heavy`, `mixed_1`, `mixed_2`) didn't clear at all — the RL run hit the 1800s cap with vehicles still stuck in the network. That's not "slower," that's a policy failure.

**Why this is happening — and why it's actually the right finding, not a broken experiment:**

Your training traffic (per earlier notes) was a long, steady, symmetric flow at ~600 veh/hr per approach. These 20 scenarios are short bursts (all departures within ~30s) with ratios up to 13x — a traffic *shape* the policy essentially never saw during training. RL policies are notoriously brittle outside their training distribution: they don't degrade gracefully, they degrade unpredictably, which is exactly the "erratic, not just worse" pattern you're seeing (severe_ns_heavy negative, severe_ew_heavy strongly positive, no consistent direction). The three gridlocks are the sharp end of that same problem — the policy likely got stuck cycling in a way it never had to unlearn, because nothing in training ever punished it for that behavior under this input shape.

This directly explains the 81% vs 4-9% discrepancy you flagged earlier: `evaluate.py`'s 81% was measured **in-distribution** (same traffic shape as training). This batch is **out-of-distribution**, and now you have hard evidence of exactly how much that matters — which is a stronger, more defensible thing to show a mentor than a single flattering number.

Let me build one more diagnostic to understand the gridlock cases specifically, since "why does it get stuck" is the one open question worth answering before you write this up.**What to run:** the smallest gridlock case, so it's fastest to inspect —
```
python diagnose_rl_gridlock.py --net onelast.net.xml --n 2 --s 2 --e 10 --w 10
```
It prints per-decision `action`, the full normalized observation, and vehicles remaining, for up to 60 decisions. Look for: does `action` alternate normally, or get stuck repeating the same value while one queue component in the observation stays high/climbs? That will tell us whether it's "agent picked one phase and never revisited it" (a real policy defect worth naming) vs. "agent keeps switching but SUMO/geometry has some other bottleneck" (a different, non-RL explanation worth ruling out).

**Suggested reporting method, revised given what the data actually shows:**

Don't report a single "% improvement" headline for this experiment — the data doesn't support one, and presenting it as if it does would be indefensible under a mentor's first follow-up question. Instead:

1. **Raw CSV** — keep as-is, it's your evidence.

2. **Grouped table showing spread and failure rate, not just mean:**

| Tier | n | Mean | Std dev | Min | Max | Failures (gridlock) |
|---|---|---|---|---|---|---|
| Balanced | 4 | ~11.5% | ~26 | −15.9% | +45.6% | 0/4 |
| Mild | 4 | ~20.8% | ~7 | +13.9% | +30.7% | 0/4 |
| Moderate | 4 | ~9.8%* | — | −5.3% | +29.9% | 1/4 |
| Severe | 6 | ~0.5%* | — | −39.9% | +27.3% | 0/6 |
| Mixed | 2 | — | — | — | — | 2/2 |

(*excluding gridlocked rows from the mean — compute exact numbers from your CSV; don't silently drop the failures from the table, just don't let them corrupt an averaged percentage that implies "it cleared, just slowly.")

3. **One chart: ratio (x) vs. improvement (y), scatter, not line** — a line implies a trend that isn't there. Mark the 3 gridlock points distinctly (e.g. red X below the axis or a separate "did not clear" marker) rather than omitting them — omitting failures from a chart is the kind of thing that looks fine until someone asks "did you exclude any runs?"

4. **Writeup framing** — this is your strongest material, honestly stated: *"RL shows a strong, reliable advantage on in-distribution traffic (evaluate.py, 81%), but on out-of-distribution burst/imbalanced scenarios (this batch), performance is inconsistent and occasionally fails to clear entirely (3/20 scenarios). This indicates the trained policy has not generalized beyond its training traffic distribution — a known RL limitation, and a concrete direction for future work (e.g. training on a more diverse traffic curriculum)."* That's a stronger, more technically credible statement than a clean percentage, and it matches how you already framed the YOLO fine-tuning failure — a documented, analyzed limitation rather than a hidden one.

Run the diagnostic on `mod_ew_heavy` when you get a chance, and paste the output — that'll tell us whether this is worth digging into further or whether it's simply confirmed distribution shift and ready to write up as-is.



4) "The original policy (v1) exhibited a starvation failure on 3/20 out-of-distribution test scenarios, traced via diagnostic logging to a lack of training exposure to near-empty network states. A curriculum retraining (v2) that mixed the original sustained-flow episodes with randomized burst-to-clearance episodes eliminated all 3 failures (100% clearance success, 0 forced interventions needed), while leaving typical-case performance on already-working scenarios statistically unchanged (9.1% vs 9.5% average improvement). This confirms the fix targeted the diagnosed failure mode specifically, rather than incidentally changing unrelated behavior."