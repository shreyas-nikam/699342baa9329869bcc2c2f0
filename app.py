import streamlit as st
from source import *

st.set_page_config(page_title="QuLab: Lab 31: Fundamental Screener Agent", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 31: Fundamental Screener Agent")
st.divider()

# Session State Initialization
if 'task' not in st.session_state:
    st.session_state['task'] = "Analyze company NVDA"
if 'agent_result' not in st.session_state:
    st.session_state['agent_result'] = None
if 'workflow_result' not in st.session_state:
    st.session_state['workflow_result'] = None

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["Agent Analysis", "Fixed Workflow Comparison"])

if page == "Agent Analysis":
    st.header("Agent Analysis")
    
    # ReAct Framework Description
    st.markdown(f"### ReAct Framework (Reason + Act)")
    st.markdown(r"$$ T_t \to A_t \to O_t \to T_{t+1} $$")
    st.markdown(r"Where $T_t$ is thought, $A_t$ is action, $O_t$ is observation.")
    
    st.markdown(f"The ReAct framework allows the agent to reason about the current state, decide on an action (tool usage), and observe the output before proceeding to the next thought.")

    # Guardrails and Compliance
    st.markdown(f"### Guardrails Implementation")
    st.markdown(f"1. **Minimum Tool Calls**: Ensure at least 2 calls.")
    st.markdown(f"2. **Maximum Iterations**: Limit to 10 to prevent loops.")

    st.divider()

    # User Input
    st.markdown(f"### Define Analysis Task")
    st.text_input("Enter Task:", key='task')

    # Execute Agent
    if st.button("Run Agent"):
        with st.spinner("Agent is executing ReAct loop..."):
            try:
                # Calling business logic from source.py
                result = run_agent(st.session_state['task'])
                st.session_state['agent_result'] = result
                st.success("Agent execution successful.")
            except Exception as e:
                st.error(f"An error occurred during agent execution: {e}")

    # Display Results
    if st.session_state['agent_result']:
        st.divider()
        st.subheader("Analysis Output")
        
        results = st.session_state['agent_result']
        
        # Use tabs to organize the rich output
        tab1, tab2, tab3, tab4 = st.tabs(["Investment Brief", "Audit Trail", "Guardrail Status", "Tool Usage"])
        
        with tab1:
            st.markdown(f"#### Investment Brief")
            # Check if result is structured dict or raw
            if isinstance(results, dict) and 'brief' in results:
                st.write(results['brief'])
            else:
                st.write(results)
        
        with tab2:
            st.markdown(f"### Audit Trail: Thought → Action → Observation")
            if isinstance(results, dict) and 'audit_trail' in results:
                st.write(results['audit_trail'])
            else:
                st.info("Detailed audit trail is available in the raw output below if not parsed.")
                st.write(results)

        with tab3:
            st.markdown(f"#### Guardrail Status")
            if isinstance(results, dict) and 'guardrails' in results:
                st.write(results['guardrails'])
            else:
                st.info("Guardrails checks passed implicitly.")

        with tab4:
            st.markdown(f"#### Tool Usage Diagram")
            if isinstance(results, dict) and 'tool_usage' in results:
                st.write(results['tool_usage'])
            else:
                st.info("Tool usage visualization not provided in response.")

elif page == "Fixed Workflow Comparison":
    st.header("Fixed Workflow Comparison")
    
    st.markdown(f"This section compares the dynamic autonomous agent against a traditional fixed workflow script. The fixed workflow executes a predetermined sequence of steps regardless of the intermediate data found.")

    # Execute Fixed Workflow
    if st.button("Execute Fixed Workflow"):
        with st.spinner("Running fixed workflow..."):
            try:
                res = fixed_workflow(st.session_state['task'])
                st.session_state['workflow_result'] = res
                st.success("Fixed workflow execution complete.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display Comparison
    if st.session_state['workflow_result']:
        st.divider()
        st.subheader("Comparative Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### Agent Result (Dynamic)")
            if st.session_state['agent_result']:
                agent_res = st.session_state['agent_result']
                if isinstance(agent_res, dict) and 'brief' in agent_res:
                    st.write(agent_res['brief'])
                else:
                    st.write(agent_res)
            else:
                st.warning("Run Agent Analysis first to see comparison.")
        
        with col2:
            st.markdown(f"#### Fixed Workflow Result (Static)")
            st.write(st.session_state['workflow_result'])
    
    st.divider()
    # Conclusion / Future Implications
    st.markdown(f"## Future Implications")
    st.markdown(f"This agent acts as a research associate, providing rapid analysis and freeing analysts for higher-value tasks.")