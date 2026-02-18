
# Streamlit Application Specification

## 1. Application Overview

The purpose of this application is to demonstrate the capabilities of an LLM-powered agent for autonomous financial analysis. It showcases how the agent intelligently decides which financial tools to use, adheres to the ReAct framework, and generates structured investment briefs.

### High-level Flow
1. **User Input**: Provide a task for the agent, such as "Analyze company ABC."
2. **Agent Execution**: The agent uses the ReAct pattern to gather data and analyze the company.
3. **Output Display**: Present the investment brief and audit trail for review.

---

## 2. Code Requirements

### Imports
```python
from source import *
import streamlit as st
```

### Application Structure

#### Main Application Layout
- **Sidebar for Navigation** with options for:
  - Agent Analysis
  - Fixed Workflow Comparison
- **Session State Management** to maintain continuity:
  - `st.session_state['task']`: Stores the user-provided task.
  - `st.session_state['agent_result']`: Logs the results of the agent's execution.
  - `st.session_state['workflow_result']`: Logs the results of the fixed workflow.

#### Agent Analysis

1. **User Input**: 
   ```python
   st.text_input("Enter Task:", key='task')
   ```

2. **Execute Agent**:
   - Button to run the agent
   - Calls `run_agent(st.session_state['task'])`

3. **Display Results**:
   - Investment Brief
   - Audit Trail
   - Guardrail Status
   - Tool Usage Diagram
  
4. **State Initialization and Updates**:
   - Initialize `st.session_state['task']` on load
   - Update `st.session_state['agent_result']` after agent execution

#### Fixed Workflow Comparison

1. **Execute Fixed Workflow**:
   - Button for workflow execution
   - Calls `fixed_workflow(st.session_state['task'])`

2. **Display Comparison**:
   - Workflow vs. Agent results table

### Markdown Content

- **ReAct Framework Description**:
  ```python
  st.markdown("### ReAct Framework (Reason + Act)")
  st.markdown(r"$$ T_t \to A_t \to O_t \to T_{t+1} $$")
  st.markdown(r"Where $T_t$ is thought, $A_t$ is action, $O_t$ is observation.")
  ```

- **Guardrails and Compliance**:
  ```python
  st.markdown("### Guardrails Implementation")
  st.markdown("1. **Minimum Tool Calls**: Ensure at least 2 calls.")
  st.markdown("2. **Maximum Iterations**: Limit to 10 to prevent loops.")
  ```

- **Audit Trail**:
  ```python
  st.markdown("### Audit Trail: Thought → Action → Observation")
  ```

- **Conclusion**:
  ```python
  st.markdown("## Future Implications")
  st.markdown(
    "This agent acts as a research associate, providing rapid analysis and freeing analysts for higher-value tasks."
  )
  ```

---

### Conclusion

This Streamlit app specification outlines the setup for an autonomous financial analysis agent using LLM and the ReAct framework. It integrates structured function calls and presents a user-friendly interface for both detailed audit trails and dynamic or fixed workflow comparisons.

