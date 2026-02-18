Here's a comprehensive `README.md` for your Streamlit application lab project:

---

# QuLab: Lab 31 - Fundamental Screener Agent

![Quant University Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

**A Streamlit-based educational lab demonstrating an AI agent using the ReAct (Reason + Act) framework for fundamental financial screening, compared against a traditional fixed workflow.**

## 1. Project Description

This project, "QuLab: Lab 31," introduces an interactive Streamlit application designed to showcase the power and flexibility of autonomous AI agents in financial analysis. Specifically, it features a "Fundamental Screener Agent" capable of performing dynamic company analysis based on user-defined tasks. The agent leverages the ReAct (Reason + Act) framework, allowing it to intelligently determine a sequence of actions (tool calls) to achieve its goal, learn from observations, and adapt its approach.

To highlight the agent's benefits, the application also includes a comparison section, running the same task through a predetermined, fixed workflow. This allows users to visually and analytically compare the dynamic, adaptive nature of an AI agent against a static, rule-based approach. This agent aims to act as a research associate, providing rapid, insightful analysis and freeing human analysts for higher-value strategic tasks.

## 2. Features

*   **Autonomous Agent (ReAct Framework):** Implements the ReAct (Reason + Act) paradigm where the agent iteratively reasons, acts (uses tools), and observes to achieve its goal.
    *   $$ T_t \to A_t \to O_t \to T_{t+1} $$
    *   Where $T_t$ is thought, $A_t$ is action, $O_t$ is observation.
*   **Dynamic Financial Screening:** The agent can analyze companies based on various fundamental data, adapting its process to the specific query.
*   **Configurable Analysis Tasks:** Users can input custom tasks (e.g., "Analyze company NVDA", "Provide a brief on GOOGL's financials and future outlook") directly into the UI.
*   **Guardrails Implementation:**
    *   **Minimum Tool Calls:** Ensures the agent performs a meaningful number of actions (e.g., at least 2 tool calls).
    *   **Maximum Iterations:** Prevents infinite loops by limiting the agent's reasoning-action cycles (e.g., max 10 iterations).
*   **Comprehensive Output:** Displays rich results organized into interactive tabs:
    *   **Investment Brief:** A concise summary of the agent's findings.
    *   **Audit Trail:** A detailed log of the agent's thought process, actions taken, and observations at each step.
    *   **Guardrail Status:** Reports on whether the implemented guardrails were met or triggered.
    *   **Tool Usage Diagram:** (Placeholder) Intended to visualize the sequence and type of tools used by the agent.
*   **Fixed Workflow Comparison:** Executes the same task using a traditional, static workflow for direct comparison with the agent's dynamic approach.
*   **Interactive Streamlit UI:** A user-friendly web interface for seamless interaction and visualization of results.
*   **Session State Management:** Preserves user inputs and agent/workflow results across interactions.

## 3. Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.8+**
*   **pip** (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/quantuniversity/qu-lab-31-fundamental-screener-agent.git
    cd qu-lab-31-fundamental-screener-agent
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

4.  **Install the required dependencies:**
    Create a `requirements.txt` file in the root directory if it doesn't exist, and add the necessary libraries. Based on the code, at minimum you'll need:
    ```
    streamlit
    # Add other dependencies required by your source.py,
    # e.g., langchain, openai, yfinance, pandas, etc.
    # For a lab, these might include:
    # langchain
    # openai
    # duckduckgo-search # Example for a search tool
    # yfinance # Example for financial data
    ```
    Then, install:
    ```bash
    pip install -r requirements.txt
    ```

5.  **Configure API Keys (if applicable):**
    If your `source.py` relies on external services like OpenAI, financial data APIs (e.g., Alpha Vantage, Yahoo Finance), ensure you set up your API keys. This is typically done via environment variables. Create a `.env` file in the root directory:
    ```
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY"
    # FUNDAMENTAL_DATA_API_KEY="YOUR_FINANCIAL_API_KEY"
    ```
    And load them in your `source.py` or `app.py` using `python-dotenv`.

## 4. Usage

To run the Streamlit application:

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project's root directory** (where `app.py` is located).
3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
4.  Your browser should automatically open to `http://localhost:8501`. If not, open your browser and navigate to this address.

### Basic Interaction:

1.  **Sidebar Navigation:** Use the sidebar to switch between "Agent Analysis" and "Fixed Workflow Comparison."
2.  **Agent Analysis:**
    *   Enter your desired analysis task in the "Enter Task:" text box (e.g., "Analyze company NVDA").
    *   Click the "Run Agent" button.
    *   Explore the results in the "Investment Brief," "Audit Trail," "Guardrail Status," and "Tool Usage" tabs.
3.  **Fixed Workflow Comparison:**
    *   Click the "Execute Fixed Workflow" button.
    *   Observe the "Agent Result (Dynamic)" alongside the "Fixed Workflow Result (Static)" for a comparative view.

## 5. Project Structure

```
.
├── app.py                     # Main Streamlit application
├── source.py                  # Contains the core business logic:
│                              # - run_agent() function (ReAct agent implementation)
│                              # - fixed_workflow() function (static workflow for comparison)
│                              # - Tool definitions (e.g., search, financial data tools)
│                              # - LLM orchestration setup
├── requirements.txt           # List of Python dependencies
├── .env                       # Environment variables (e.g., API keys - DO NOT COMMIT TO GIT)
├── .gitignore                 # Specifies intentionally untracked files to ignore
└── README.md                  # This documentation file
```

## 6. Technology Stack

*   **Frontend/UI:** [Streamlit](https://streamlit.io/)
*   **Backend/Logic:** Python 3.8+
*   **Agent Framework:** ReAct (Reason + Act) framework, likely implemented using an LLM orchestration library like [LangChain](https://www.langchain.com/) or [LlamaIndex](https://www.llamaindex.ai/) (details are within `source.py`).
*   **Large Language Models (LLMs):** Underlying AI models (e.g., OpenAI GPT series, Llama, etc.) power the agent's reasoning.
*   **Tools:**
    *   Financial Data APIs (e.g., yfinance, Alpha Vantage - implemented in `source.py`).
    *   Search Tools (e.g., DuckDuckGo Search, Google Search API - implemented in `source.py`).
    *   Other custom tools as defined in `source.py`.

## 7. Contributing

This project is primarily developed for educational purposes within the QuLab framework. While we are not actively seeking external contributions, you are welcome to fork the repository, experiment with the code, and adapt it for your own learning or projects.

For any specific questions or suggestions related to the lab, please refer to the course instructions or contact Quant University directly.

## 8. License

This project is licensed under the [MIT License](LICENSE.md).

```
MIT License

Copyright (c) 2023 Quant University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 9. Contact

For more information about Quant University and its programs, please visit:

*   **Website:** [Quant University](https://www.quantuniversity.com/)
*   **QuLab Portal:** [QuLab](https://www.quantuniversity.com/qu-lab/)

---

## License

## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
