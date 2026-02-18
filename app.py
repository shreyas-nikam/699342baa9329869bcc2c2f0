import streamlit as st
import json
from source import *

# ----------------------------
# App Config + Global Helpers
# ----------------------------
st.set_page_config(page_title="QuLab: Lab 31: Fundamental Screener Agent", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 31: Fundamental Screener Agent")
st.divider()


TICKERS = ["NVDA", "AAPL", "MSFT", "GOOG", "TSLA", "AMZN", "META", "JPM"]

TICKER_LABELS = {
    "NVDA": "NVDA — NVIDIA (Semis / AI Infra)",
    "AAPL": "AAPL — Apple (Consumer Tech)",
    "MSFT": "MSFT — Microsoft (Software / Cloud)",
    "GOOG": "GOOG — Alphabet (Internet / Ads)",
    "TSLA": "TSLA — Tesla (Autos / EV)",
    "AMZN": "AMZN — Amazon (E-comm / Cloud)",
    "META": "META — Meta (Internet / Ads)",
    "JPM": "JPM — JPMorgan (Financials)",
}

EVIDENCE_TOOLS = [
    ("get_stock_overview", "Company snapshot (valuation + risk context)"),
    ("get_financials", "Financial statements (directional trend check)"),
    ("get_recent_news", "Catalysts / headline risk radar"),
    ("get_peer_comparison", "Relative context vs sector benchmark"),
]


def _assumptions_box():
    st.info(
        "Assumptions & data limits (read before interpreting outputs)\n\n"
        "- This is a *screening* brief, not a full valuation model or a recommendation.\n"
        "- Financial statements shown are typically limited (e.g., last ~2 periods) and may miss one-offs.\n"
        "- News is a catalyst/risk radar; it is not a fundamentals model.\n"
        "- Peer “sector medians” are teaching benchmarks (directional context), not a live index.\n"
        "- Any numeric claim must be traceable to an evidence output (see Evidence Log)."
    )


def _require_api_key() -> bool:
    if not st.session_state.get("openai_api_key"):
        st.error("Please enter your OpenAI API key in the sidebar to run the Agent / Workflow.")
        return False
    return True


def _safe_json_loads(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    if isinstance(x, dict):
        return x
    return None


def _render_json_with_takeaways(title: str, payload: dict, takeaways: list[str], show_raw_default=False):
    st.markdown(f"#### {title}")
    if takeaways:
        st.markdown("**Key takeaways (how to read this):**")
        for t in takeaways:
            st.markdown(f"- {t}")
    with st.expander("Show raw output (audit view)", expanded=show_raw_default):
        st.json(payload)


def _extract_trace_table(agent_result: dict):
    """
    Normalizes source.py 'trace' into a list of rows for display.
    Each trace entry is expected to have: iteration, thought, action, observation.
    """
    trace = agent_result.get("trace") if isinstance(agent_result, dict) else None
    if not isinstance(trace, list):
        return []

    rows = []
    for step in trace:
        if not isinstance(step, dict):
            continue
        rows.append(
            {
                "Iteration": step.get("iteration"),
                "Thought (why)": step.get("thought", ""),
                "Action (what evidence)": step.get("action", ""),
                "Observation (what came back)": step.get("observation", ""),
            }
        )
    return rows


def _derive_evidence_coverage(trace_rows: list[dict]) -> dict:
    """
    Heuristic: looks for tool names in Action field.
    """
    coverage = {name: False for name, _ in EVIDENCE_TOOLS}
    for r in trace_rows:
        a = (r.get("Action (what evidence)") or "").lower()
        for tool_name, _ in EVIDENCE_TOOLS:
            if tool_name.lower() in a:
                coverage[tool_name] = True
    return coverage


def _coverage_meter(coverage: dict):
    st.markdown("**Evidence completeness (checklist):**")
    cols = st.columns(4)
    for idx, (tool_name, label) in enumerate(EVIDENCE_TOOLS):
        ok = bool(coverage.get(tool_name))
        with cols[idx]:
            st.success(f"✅ {tool_name}") if ok else st.warning(f"⬜ {tool_name}")
            st.caption(label)


def _validate_and_render_guardrails(agent_result: dict):
    checks, passed, total = validate_agent_output(agent_result)

    st.markdown("#### Validation checks (guardrails)")
    st.caption("Interpretation: these are minimum due-diligence checks for a *screening* output.")

    # Show pass/fail per check (finance-native labels)
    label_map = {
        "has_company_overview": "Contains company overview context",
        "has_valuation": "Mentions valuation framing (e.g., P/E, P/B)",
        "has_financial_trends": "Mentions financial trend(s) (growth/margins/earnings)",
        "has_catalysts_risks": "Mentions catalysts/risks (news/headlines)",
        "has_conclusion": "Has a conclusion (attractive/neutral/unattractive)",
        "used_data_tools": "Used minimum evidence tools (≥ 2 tool calls)",
        "within_iteration_limit": "Stayed within iteration limit (≤ 10)",
    }

    for k, v in checks.items():
        label = label_map.get(k, k)
        if v:
            st.success(f"PASS: {label}")
        else:
            st.error(f"FAIL: {label}")
        

    st.markdown(f"**Score:** {passed}/{total} checks passed.")

    # Guardrail interpretation line
    if passed < total:
        st.warning(
            "Guardrail note: If checks fail, treat this output as incomplete. "
            "Re-run or use the fixed workflow baseline for consistent coverage."
        )
    else:
        st.success(
            "Guardrail note: Minimum screening completeness is satisfied. "
            "Next step is to verify any numeric claims in the Evidence Log."
        )


# ----------------------------
# Session State Initialization
# ----------------------------
if "agent_result" not in st.session_state:
    st.session_state["agent_result"] = None
if "workflow_result" not in st.session_state:
    st.session_state["workflow_result"] = None
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "learning_step" not in st.session_state:
    st.session_state["learning_step"] = "1) Overview"


# ----------------------------
# Sidebar: Configuration + Path
# ----------------------------
st.sidebar.markdown("### Configuration")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    type="password",
    value=st.session_state["openai_api_key"],
    help="Used only for this session to run the agent/workflow. Not stored permanently.",
)
st.session_state["openai_api_key"] = api_key_input

st.sidebar.divider()

st.sidebar.markdown("### Suggested learning path")
st.sidebar.caption("Follow this order once, then explore freely.")
step = st.sidebar.radio(
    "Go to:",
    [
        "1) Overview",
        "2) Evidence Library (Tools Playground)",
        "3) Agent: Screening Brief + Evidence Log",
        "4) Compare: Agent vs Fixed Workflow",
    ],
    index=[
        "1) Overview",
        "2) Evidence Library (Tools Playground)",
        "3) Agent: Screening Brief + Evidence Log",
        "4) Compare: Agent vs Fixed Workflow",
    ].index(st.session_state["learning_step"]),
)
st.session_state["learning_step"] = step

# Map "step" to internal page name (keeps the original nav options but makes it pedagogy-first)
if step.startswith("1)"):
    page = "Overview"
elif step.startswith("2)"):
    page = "Tools Reference and Playground"
elif step.startswith("3)"):
    page = "Agent Analysis"
else:
    page = "Fixed Workflow Comparison"


# ----------------------------
# Page 1: Overview
# ----------------------------
if page == "Overview":
    st.markdown("# Overview")
    st.markdown("## What you’ll be able to do after this lab")
    st.markdown(
        "Build and *audit* a screening brief for a stock using a research loop (Reason → Evidence → Update thesis), "
        "with explicit guardrails so outputs are not ‘numbers from nowhere’."
    )

    _assumptions_box()

    st.markdown("## Core idea (finance-native)")
    st.markdown(
        "- Think of the agent as an analyst-in-training: it forms a hypothesis, pulls evidence, and updates its view.\n"
        "- The goal is not storytelling; it is **decision-useful screening** with an inspectable evidence log.\n"
        "- The fixed workflow is your baseline: a checklist process that is consistent but not adaptive."
    )

    st.markdown("## Key terms (plain English)")
    st.markdown(
        "- **Research loop (ReAct)**: Hypothesis → pull evidence → update thesis.\n"
        "- **Evidence tools**: standard first-pass inputs (overview, financials, news, peers).\n"
        "- **Guardrails**: minimum checks for completeness (e.g., did we look at enough evidence?)\n"
        "- **Dynamic vs fixed**: discretion and adaptivity vs checklist consistency."
    )

    st.divider()

    # Checkpoint question (learning)
    st.markdown("## Quick checkpoint (30 seconds)")
    q = st.radio(
        "A screening output says: “premium valuation vs peers, but attractive.” What is the *minimum* next step before you act on it?",
        [
            "Accept it if the narrative sounds coherent",
            "Verify which evidence supports the valuation claim (peer comparison) and whether fundamentals justify the premium",
            "Ignore valuation because it is always noisy",
        ],
        index=1,
    )
    if q == "Verify which evidence supports the valuation claim (peer comparison) and whether fundamentals justify the premium":
        st.success(
            "Correct. Screening is about disciplined verification: first confirm the comparative claim, "
            "then check whether growth/margins/quality plausibly justify a premium."
        )
    else:
        st.info(
            "Not quite. The key is *traceability*: premium/discount claims require evidence (peers) "
            "and a fundamentals justification (growth, margins, durability)."
        )


# ----------------------------
# Page 2: Tools Reference + Playground
# ----------------------------
elif page == "Tools Reference and Playground":
    st.header("Evidence Library: what each tool tells you")

    _assumptions_box()

    st.markdown(
        "Use this page to build intuition about **what each evidence source answers**, what it *does not* answer, "
        "and how to interpret it for a first-pass screen."
    )

    tool_selection = st.selectbox(
        "Choose an evidence source to inspect:",
        ["get_stock_overview", "get_financials", "get_recent_news", "get_peer_comparison"],
        help="Pick one evidence type. Read the ‘Key takeaways’ before opening the raw output.",
    )

    st.divider()

    st.markdown("### Select tickers to test (1–4)")
    test_tickers = st.multiselect(
        "Tickers",
        TICKERS,
        default=["NVDA", "AAPL", "MSFT"],
        max_selections=4,
        format_func=lambda t: TICKER_LABELS.get(t, t),
        help="Comparing 2–4 tickers is the fastest way to develop screening intuition.",
    )

    st.divider()

    if tool_selection == "get_stock_overview":
        st.markdown("### Company snapshot (overview)")
        st.caption("Use for: quick context (size/sector), valuation framing, and risk context (beta, margins).")

        if st.button("Run overview for selected tickers", key="run_overview"):
            with st.spinner("Fetching company snapshots..."):
                tabs = st.tabs([TICKER_LABELS.get(t, t) for t in test_tickers])
                for idx, ticker in enumerate(test_tickers):
                    with tabs[idx]:
                        raw = get_stock_overview(ticker)
                        payload = _safe_json_loads(raw) or {"raw": raw}

                        takeaways = [
                            "Price/market cap are *context*, not a thesis.",
                            "P/E and P/B are *descriptors*; they are not conclusions by themselves.",
                            "Beta is a *risk context* input (required return / drawdown sensitivity).",
                            "Margins + revenue growth are first-pass signals for durability vs cyclicality.",
                        ]
                        _render_json_with_takeaways("Output", payload, takeaways)

                        st.markdown("**Decision translation:**")
                        st.markdown(
                            "- If valuation is high *and* beta is high, require stronger evidence of durable growth.\n"
                            "- If margins are strong and stable, it raises the bar for ‘cheap’ arguments based only on P/E."
                        )

    elif tool_selection == "get_financials":
        st.markdown("### Financial statements (directional trend check)")
        st.caption("Use for: a quick sanity check on growth/margins/earnings quality before any conclusion.")

        statement = st.selectbox(
            "Statement type",
            ["income", "balance", "cashflow"],
            index=0,
            help="For screening, income often anchors growth/margins; cashflow helps with earnings quality.",
        )

        if st.button("Run financial statements for selected tickers", key="run_financials"):
            with st.spinner(f"Fetching {statement} statements..."):
                tabs = st.tabs([TICKER_LABELS.get(t, t) for t in test_tickers])
                for idx, ticker in enumerate(test_tickers):
                    with tabs[idx]:
                        raw = get_financials(ticker, statement)
                        payload = _safe_json_loads(raw) or {"raw": raw}

                        takeaways = [
                            "This is usually limited to ~2 periods: treat as *directional*, not definitive.",
                            "For screening, focus on a few lines that move the thesis: revenue, operating income, margins, cash generation.",
                            "One-offs can distort period-to-period comparisons; verify in filings before conviction.",
                        ]
                        _render_json_with_takeaways("Output", payload, takeaways)

                        st.markdown("**Decision translation:**")
                        st.markdown(
                            "- If margins deteriorate while valuation remains premium, downgrade confidence until explained.\n"
                            "- If cash flow diverges from earnings, increase skepticism about earnings quality."
                        )

                        st.warning(
                            "Watch-out: Two-period data can mislead around cyclicality and one-offs. "
                            "Use this as a screening gate, not a full historical analysis."
                        )

    elif tool_selection == "get_recent_news":
        st.markdown("### News (catalyst / headline risk radar)")
        st.caption("Use for: identifying catalysts, regulatory/legal risks, and narrative shocks that affect scenarios.")

        limit = st.slider(
            "Number of headlines (per ticker)",
            min_value=1,
            max_value=10,
            value=3,
            help="News is context and scenario shaping; it is not a substitute for fundamentals.",
        )

        if st.button("Run news for selected tickers", key="run_news"):
            with st.spinner("Fetching recent news..."):
                tabs = st.tabs([TICKER_LABELS.get(t, t) for t in test_tickers])
                for idx, ticker in enumerate(test_tickers):
                    with tabs[idx]:
                        raw = get_recent_news(ticker, limit=limit)
                        payload = _safe_json_loads(raw) or {"raw": raw}

                        takeaways = [
                            "Treat as a *risk/catalyst list* (what could move the stock), not a thesis engine.",
                            "A headline matters only if it plausibly changes cash flows, discount rate, or competitive position.",
                            "Overweighting news is a common screening error for fundamentals-driven decisions.",
                        ]
                        _render_json_with_takeaways("Output", payload, takeaways)

                        st.markdown("**Decision translation:**")
                        st.markdown(
                            "- If headline risk rises, widen scenario dispersion and revisit downside cases.\n"
                            "- If a catalyst is credible (earnings, product cycle, regulation), identify what metric would confirm it."
                        )

    elif tool_selection == "get_peer_comparison":
        st.markdown("### Peer comparison (relative context)")
        st.caption("Use for: premium/discount context vs a benchmark; then ask if fundamentals justify the spread.")

        if st.button("Run peer comparisons for selected tickers", key="run_peers"):
            with st.spinner("Fetching peer comparisons..."):
                tabs = st.tabs([TICKER_LABELS.get(t, t) for t in test_tickers])
                for idx, ticker in enumerate(test_tickers):
                    with tabs[idx]:
                        raw = get_peer_comparison(ticker)
                        payload = _safe_json_loads(raw) or {"raw": raw}

                        takeaways = [
                            "Sector medians here are **teaching benchmarks** (directional context), not a live peer index.",
                            "Premium/discount is not good/bad by itself—justify via growth, margins, ROIC durability, moat.",
                            "Always confirm you’re comparing like-for-like (sector ≠ business model).",
                        ]
                        _render_json_with_takeaways("Output", payload, takeaways)

                        st.markdown("**Decision translation:**")
                        st.markdown(
                            "- If the stock is at a premium, require a clear durability argument (growth + margins + risk).\n"
                            "- If it’s at a discount, ask whether it is mispriced or structurally lower quality."
                        )

                        st.warning(
                            "Guardrail: Treat benchmark medians as instructional. "
                            "For real coverage work, replace with your own peer set and date-stamped medians."
                        )


# ----------------------------
# Page 3: Agent Analysis
# ----------------------------
elif page == "Agent Analysis":
    st.header("Agent: Screening Brief + Evidence Log")

    # Keep existing formulae exactly as in the user's current markdown blocks (constraint)
    st.markdown("### ReAct Framework (Reason + Act)")
    st.markdown(r"$$ T_t \to A_t \to O_t \to T_{t+1} $$")
    st.markdown(r"Where $T_t$ is thought, $A_t$ is action, $O_t$ is observation.")

    st.markdown(
        "Interpretation (finance-native): *Hypothesis → pull evidence → update thesis* until you can write a screening brief "
        "with traceable support."
    )

    _assumptions_box()

    st.markdown("### Define your screening task")
    colA, colB = st.columns([2, 3])

    with colA:
        agent_ticker = st.selectbox(
            "Ticker",
            TICKERS,
            index=0,
            format_func=lambda t: TICKER_LABELS.get(t, t),
            key="agent_ticker",
            help="Pick one ticker. You’ll validate the brief by inspecting the evidence log.",
        )

    with colB:
        prior = st.text_input(
            "Your prior (optional, 1 sentence)",
            value="",
            placeholder="Example: 'Premium valuation may be justified by durable growth and margin expansion.'",
            help="This is a learning device: you’ll compare your prior vs what evidence supports.",
        )

    st.divider()

    st.markdown("### Guardrails (minimum due diligence)")
    st.markdown(
        "- Minimum evidence: **at least 2 evidence tool calls** before a conclusion.\n"
        "- Maximum iterations: **10** (prevents looping).\n"
        "- Output quality checks: overview/valuation/trends/risks/conclusion should appear in the brief."
    )

    st.divider()

    if st.button("Run Agent (generate screening brief)"):
        if _require_api_key():
            with st.spinner("Running research loop (Reason → Evidence → Update thesis)..."):
                try:
                    initialize_openai_client(api_key=st.session_state["openai_api_key"])
                    result = run_agent(agent_ticker)
                    st.session_state["agent_result"] = result
                    st.success("Agent run complete. Now verify the brief using the Evidence Log.")
                except Exception as e:
                    st.error(f"An error occurred during agent execution: {e}")

    # Display Agent Results
    if st.session_state.get("agent_result"):
        results = st.session_state["agent_result"]
        st.divider()

        # Tabs (pedagogy-first naming)
        tab1, tab2, tab3 = st.tabs(
            [
                "Screening Brief (thesis + drivers)",
                "Evidence Log (audit trail)",
                "Validation Checks (guardrails)",
            ]
        )

        with tab1:
            st.markdown("#### Screening Brief")
            st.caption("Read the thesis, then verify claims below. Treat as a screening memo, not an investment recommendation.")

            brief = results.get("brief") if isinstance(results, dict) else None
            if brief:
                if prior.strip():
                    st.markdown("**Your prior:**")
                    st.write(prior)
                    st.markdown("**Generated screening brief:**")
                st.write(brief)
            else:
                st.write(results)

            st.markdown("**Decision translation (how to use this):**")
            st.markdown(
                "- If attractiveness improves due to growth/margins, consider putting it on a watchlist and scheduling deeper diligence.\n"
                "- If attractiveness worsens due to valuation premium, explicitly test whether durability justifies the spread before acting."
            )

        with tab2:
            st.markdown("#### Evidence Log (Thought → Action → Observation)")
            st.caption("This is the trust layer. If a claim is not supported by evidence, downgrade or reject it.")

            trace_rows = _extract_trace_table(results)
            if trace_rows:
                st.dataframe(trace_rows, use_container_width=True)
            else:
                st.info("No structured trace found. Showing raw output instead.")
                st.write(results)

            st.markdown("**Guardrail for interpretation:**")
            st.warning(
                "If the brief makes numeric or comparative claims (e.g., premium vs peers) but the Evidence Log does not show "
                "the corresponding tool call, treat that claim as unsupported."
            )

        with tab3:
            if isinstance(results, dict):
                _validate_and_render_guardrails(results)
                
            else:
                st.info("Guardrail validation expects a structured dict result.")

       


# ----------------------------
# Page 4: Fixed Workflow Comparison
# ----------------------------
elif page == "Fixed Workflow Comparison":
    st.header("Compare: Agent vs Fixed Workflow")

    _assumptions_box()

    st.markdown(
        "Use this as a discipline tool:\n"
        "- The **fixed workflow** is a checklist baseline (consistent coverage).\n"
        "- The **agent** is adaptive (may skip/sequence evidence differently) but must be audited.\n"
        "The comparison is most useful when you inspect **evidence completeness** and **claim traceability**."
    )

    st.divider()

    st.markdown("### Step 1 — Choose a ticker")
    workflow_ticker = st.selectbox(
        "Ticker",
        TICKERS,
        index=0,
        format_func=lambda t: TICKER_LABELS.get(t, t),
        key="workflow_ticker",
        help="For a clean comparison, use the same ticker you ran in the Agent tab.",
    )

    col1, col2 = st.columns([1, 2])
    with col1:
        run_workflow = st.button("Run fixed workflow (baseline)")

    with col2:
        st.caption("Suggested sequence: 1) Run Agent  2) Run Workflow  3) Compare coverage + claims")

    if run_workflow:
        if _require_api_key():
            with st.spinner("Running fixed checklist workflow..."):
                try:
                    initialize_openai_client(api_key=st.session_state["openai_api_key"])
                    res = fixed_workflow(workflow_ticker)
                    st.session_state["workflow_result"] = res
                    st.success("Fixed workflow complete.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

    st.divider()

    # Comparative view (with an explicit comparison table + warnings)
    if st.session_state.get("workflow_result"):
        agent_res = st.session_state.get("agent_result")

        st.subheader("Side-by-side briefs")
        colA, colB = st.columns(2)

        with colA:
            st.markdown("#### Agent (adaptive evidence collection)")
            if agent_res and isinstance(agent_res, dict) and agent_res.get("brief"):
                st.write(agent_res["brief"])
            elif agent_res:
                st.write(agent_res)
            else:
                st.warning("Run the Agent page first to generate an agent brief for comparison.")

        with colB:
            st.markdown("#### Fixed workflow (checklist baseline)")
            st.write(st.session_state["workflow_result"])

        st.divider()

        st.subheader("Structured comparison (what a skeptical reviewer checks)")
        agent_tool_calls = agent_res.get("tool_calls") if isinstance(agent_res, dict) else None
        agent_iters = agent_res.get("iterations") if isinstance(agent_res, dict) else None

        # Coverage from trace (agent)
        trace_rows = _extract_trace_table(agent_res) if isinstance(agent_res, dict) else []
        coverage = _derive_evidence_coverage(trace_rows) if trace_rows else {k: False for k, _ in EVIDENCE_TOOLS}

        comparison_rows = [
            {
                "Metric": "Tool calls (evidence pulls)",
                "Fixed workflow": "4 (always calls all tools)",
                "Agent": f"{agent_tool_calls} (dynamic)" if agent_tool_calls is not None else "N/A",
            },
            {
                "Metric": "Evidence completeness (overview/financials/news/peers)",
                "Fixed workflow": "Expected: complete",
                "Agent": ", ".join([k for k, ok in coverage.items() if ok]) or "No tools detected",
            },
            {
                "Metric": "Predictability",
                "Fixed workflow": "High (same steps every time)",
                "Agent": "Medium (adaptive sequence)",
            },
            {
                "Metric": "Auditability",
                "Fixed workflow": "Medium (final brief only)",
                "Agent": "High (thought→action→observation log)" if trace_rows else "Medium/Unknown",
            },
            {
                "Metric": "Iteration count",
                "Fixed workflow": "N/A",
                "Agent": str(agent_iters) if agent_iters is not None else "N/A",
            },
        ]
        st.dataframe(comparison_rows, use_container_width=True)

        st.markdown("**Guardrails to prevent misinterpretation:**")
        st.warning(
            "If the Agent claims a premium/discount without peer evidence, or discusses margins without financial evidence, "
            "use the fixed workflow as the baseline and treat the agent output as incomplete."
        )


# ----------------------------
# License (must remain)
# ----------------------------
st.caption(
    """
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
"""
)
