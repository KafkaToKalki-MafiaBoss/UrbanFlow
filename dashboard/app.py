"""
demo_comparison_page.py

Phase D — standalone Streamlit page for the live hackathon demo.
Meant to be merged into Member 4's existing dashboard/app.py as a
page/section (or run standalone for testing).

Flow:
  4 number inputs (N/S/E/W) -> Run button -> run_scenario_comparison()
  -> SUMO-GUI pops up and shows the RL run live (use_gui=True) ->
  results panel shows both clearance times once the run finishes.

Fixed-timer runs headless in the background (as designed) - only the
RL run is shown visually, per the hackathon time-budget decision.

Merge notes for Member 4's dashboard:
  - The two sys.path.append lines below assume this file sits in
    dashboard/. If placed elsewhere, adjust PROJECT_ROOT accordingly.
  - run_scenario_comparison() is a plain function call - blocking,
    synchronous. Streamlit will show a spinner via st.spinner() while
    it runs; no async/threading needed for a single-user hackathon
    demo.
  - If you want to log this run via your DB logger, the `result` dict
    returned below has everything needed - insert a call to your
    logger right after `result = run_scenario_comparison(...)`,
    before results are rendered.
"""

import os
import sys

import streamlit as st

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "integration"))

from integration.compare_scenario import run_scenario_comparison, DEFAULT_MAX_GREEN_SECONDS


def render_comparison_page():
    st.header("UrbanFlow — Live Signal Comparison")
    st.caption(
        "Enter approach vehicle counts, then run. The RL agent's run "
        "displays live in a SUMO-GUI window; the fixed-timer baseline "
        "runs in the background for comparison."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n = st.number_input("North", min_value=0, max_value=50, value=10, step=1)
    with col2:
        s = st.number_input("South", min_value=0, max_value=50, value=10, step=1)
    with col3:
        e = st.number_input("East", min_value=0, max_value=50, value=5, step=1)
    with col4:
        w = st.number_input("West", min_value=0, max_value=50, value=5, step=1)

    run_clicked = st.button("Run Comparison", type="primary")

    if run_clicked:
        counts = {"N": int(n), "S": int(s), "E": int(e), "W": int(w)}

        if sum(counts.values()) == 0:
            st.warning("Enter at least one vehicle before running.")
            return

        with st.spinner("Running scenario — watch the SUMO-GUI window for the RL run..."):
            try:
                result = run_scenario_comparison(
                    counts,
                    max_green_seconds=DEFAULT_MAX_GREEN_SECONDS,
                )
                sys.path.append(os.path.join(PROJECT_ROOT, "db"))
                from db.logger import log_run
                log_run(result)
            except Exception as exc:
                st.error(f"Run failed: {exc}")
                return

        st.success("Run complete.")
        _render_results(result)


def _render_results(result: dict):
    st.subheader("Results")

    fixed = result["fixed_timer"]
    rl = result["rl_agent"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            label="Fixed-Timer Clearance",
            value=f"{fixed['clearance_time']:.2f}s" if fixed["cleared"] else "Did not clear",
        )
    with col2:
        st.metric(
            label="RL Agent Clearance",
            value=f"{rl['clearance_time']:.2f}s" if rl["cleared"] else "Did not clear",
            delta=(
                f"{result['clearance_time_diff_s']:.2f}s faster"
                if result["pct_improvement"] is not None
                else None
            ),
        )

    if result["pct_improvement"] is not None:
        st.info(f"RL improved clearance time by **{result['pct_improvement']:.1f}%** over the fixed-timer baseline.")
    else:
        st.warning("One or both controllers did not clear the scenario — no improvement percentage available.")

    with st.expander("Raw result (debug)"):
        st.json(result)


if __name__ == "__main__":
    st.set_page_config(page_title="UrbanFlow Demo", layout="centered")
    render_comparison_page()