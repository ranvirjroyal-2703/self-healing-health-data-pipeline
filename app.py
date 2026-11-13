"""Streamlit UI. One button produces the two moments other teams skip."""

import os

import pandas as pd
import streamlit as st

from agent import STORY, DataPipelineAgent
from sandbox_runner import SandboxRunner

SAMPLE_PATH = os.path.join(os.path.dirname(__file__), "data", "messy_health.csv")
GOAL = (
    "Drop duplicate patients, remove empty rows, strip names, "
    "coerce age and lab_result to numeric, and normalize status and dates."
)

st.set_page_config(page_title="Health Data Cleaner", layout="wide")
st.title("Self-Healing Health Data Pipeline")
st.caption("One messy public-health export. Every clean runs in a sandbox. A bad guess is repaired from the real error.")
st.info(STORY)

with st.sidebar:
    st.header("Sandbox limits")
    memory = st.selectbox("Memory", ["128m", "256m", "512m"], index=1)
    timeout = st.slider("Timeout (seconds)", 5, 30, 12)
    st.caption("Cost = seconds × (memory / 256MB). Lower successful cost wins.")

if "csv_text" not in st.session_state:
    with open(SAMPLE_PATH, encoding="utf-8") as handle:
        st.session_state.csv_text = handle.read()

raw = st.session_state.csv_text
st.subheader("This week's field export")
st.dataframe(pd.read_csv(pd.io.common.StringIO(raw)), use_container_width=True)

if st.button("Run judge demo", type="primary"):
    if not os.getenv("VULTR_INFERENCE_API_KEY"):
        st.error("Set VULTR_INFERENCE_API_KEY (or compatible OpenAI endpoint). Strategy generation and the heal step both use the LLM.")
    else:
        agent = DataPipelineAgent()
        agent.sandbox.memory_limit = memory
        agent.sandbox.timeout_sec = timeout
        sample = "\n".join(raw.splitlines()[:12])

        with st.spinner("Planning, running, and repairing inside sandboxes..."):
            strategies = agent.propose_strategies(sample, GOAL)
            attempts = agent.run_with_healing(strategies, raw, GOAL)

        failed = next((item for item in attempts if item["round"] == 1 and not item["result"].success), None)
        healed = next((item for item in attempts if item["round"] == 2), None)

        st.subheader("1. Failure, then the repair")
        left, right = st.columns(2)
        if failed:
            left.markdown("**Round 1 — rejected by the sandbox**")
            left.code(failed["code"], language="python")
            left.error((failed["result"].stderr or "no stderr")[:900])
        if healed:
            right.markdown("**Round 2 — rewritten from that stderr**")
            right.code(healed["code"], language="python")
            if healed["result"].success:
                right.success(f"Repaired run succeeded in {healed['result'].duration_sec}s. Cost {healed['cost']}.")
                right.code((healed["result"].stdout or "")[:900])
            else:
                right.warning("Repair ran, but this attempt still failed.")
                right.code((healed["result"].stderr or "")[:700])

        st.subheader("2. Which clean is cheapest")
        ranked = agent.rank(attempts)
        st.dataframe(pd.DataFrame([
            {
                "Rank": index + 1,
                "Strategy": item["strategy"]["name"],
                "Round": item["round"],
                "Success": bool(item["result"].success and not item["result"].timed_out),
                "Seconds": item["result"].duration_sec,
                "Memory": item["memory_limit"],
                "Cost": item["cost"],
            }
            for index, item in enumerate(ranked)
        ]), use_container_width=True)
        best = agent.select_best(attempts)
        if best and best["result"].success:
            st.success(
                f"Ship `{best['strategy']['name']}` for the health report. "
                f"Cost {best['cost']} ({best['result'].duration_sec}s at {best['memory_limit']})."
            )

        st.subheader("3. Blast radius = zero")
        st.write("The same sandbox now runs an infinite loop. The host process is not the one that dies.")
        killer = SandboxRunner(memory_limit=memory, timeout_sec=8)
        trapped = killer.run("import time\nprint('loop started')\nwhile True:\n    time.sleep(0.1)\n")
        if trapped.timed_out or trapped.killed_by_limit:
            st.success(f"Killed after {trapped.duration_sec}s. The app is still here. Blast radius = zero.")
        else:
            st.warning("The loop was not contained. Docker or the sandbox image is missing.")
        st.code(trapped.stderr[:500] or trapped.stdout[:500])
