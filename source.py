import openai
import json
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timezone

# --- OpenAI Client Initialization ---
# This client will be initialized once when the module is used in an application.
# It's kept as a module-level variable to avoid re-initialization.
_openai_client = None

def initialize_openai_client(api_key: str = None):
    """
    Initializes the OpenAI client.
    If api_key is not provided, it attempts to read from the OPENAI_API_KEY environment variable.
    This function should be called once at the application's startup.
    """
    global _openai_client
    if _openai_client is not None:
        print("OpenAI client already initialized.")
        return

    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OpenAI API key not provided and not found in environment variables. "
                         "Please set OPENAI_API_KEY or pass it to initialize_openai_client().")
    _openai_client = openai.OpenAI(api_key=api_key)
    print("OpenAI client initialized.")

def get_openai_client():
    """
    Returns the initialized OpenAI client instance.
    Raises a RuntimeError if the client has not been initialized.
    """
    if _openai_client is None:
        raise RuntimeError("OpenAI client not initialized. Call initialize_openai_client() first.")
    return _openai_client

# --- Tool Functions ---

def get_stock_overview(ticker: str) -> str:
    """
    Retrieve current price, market cap, sector, and key ratios for a stock ticker.
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        overview = {
            'ticker': ticker,
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'price': info.get('currentPrice', 'N/A'),
            'market_cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap') else 'N/A',
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'forward_pe': info.get('forwardPE', 'N/A'),
            'pb_ratio': info.get('priceToBook', 'N/A'),
            'dividend_yield': f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') is not None else 'N/A',
            'beta': info.get('beta', 'N/A'),
            'revenue_growth': f"{info.get('revenueGrowth', 0)*100:.1f}%" if info.get('revenueGrowth') is not None else 'N/A',
            'profit_margin': f"{info.get('profitMargins', 0)*100:.1f}%" if info.get('profitMargins') is not None else 'N/A',
            '52wk_high': info.get('fiftyTwoWeekHigh', 'N/A'),
            '52wk_low': info.get('fiftyTwoWeekLow', 'N/A'),
        }
        return json.dumps(overview, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "message": f"Could not retrieve stock overview for ticker '{ticker}'."})

def get_financials(ticker: str, statement: str = "income") -> str:
    """
    Retrieve income statement, balance sheet, or cash flow for the last 2 years/periods.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    - statement (str): Type of statement ('income', 'balance', 'cashflow'). Defaults to 'income'.

    Returns:
    - str: JSON string with statement data or error payload.
    """
    try:
        t = (ticker or "").strip().upper()
        st = (statement or "income").strip().lower()

        if not t:
            return json.dumps({"error": "Invalid ticker", "message": "Ticker cannot be empty."}, indent=2)

        stock = yf.Ticker(t)

        statement_attr_candidates = {
            "income": ["income_stmt", "financials"],
            "balance": ["balance_sheet", "balancesheet"],
            "cashflow": ["cashflow", "cash_flow"],
        }

        if st not in statement_attr_candidates:
            return json.dumps(
                {"error": "Invalid statement type", "message": "Choose 'income', 'balance', or 'cashflow'."},
                indent=2,
            )

        df = None
        for attr in statement_attr_candidates[st]:
            df = getattr(stock, attr, None)
            if isinstance(df, pd.DataFrame) and not df.empty:
                break
            df = None

        if df is None or df.empty:
            return json.dumps(
                {
                    "error": "No data available",
                    "message": f"Could not retrieve {st} statement for {t}.",
                },
                indent=2,
            )

        df2 = df.iloc[:, :2].copy()

        df2.index = df2.index.astype(str)
        df2.columns = [str(c) for c in df2.columns]
        df2 = df2.where(pd.notnull(df2), None)

        payload = {
            "ticker": t,
            "statement": st,
            "periods": list(df2.columns),
            "data": df2.to_dict(),
        }

        return json.dumps(payload, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"error": str(e), "message": f"Could not retrieve {statement} statement for ticker '{ticker}'."},
            indent=2,
        )

def _parse_iso_to_epoch(iso_str: str):
    """Convert ISO8601 like '2026-02-17T21:01:19Z' to epoch seconds (int)."""
    if not iso_str or not isinstance(iso_str, str):
        return None
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except Exception:
        return None

def get_recent_news(ticker: str, limit: int = 5) -> str:
    """
    Retrieve recent news headlines for a ticker. Returns up to `limit` articles.
    Works with the new nested yfinance news format and older flat format.

    Parameters:
    - ticker (str): stock ticker (e.g., "AAPL")
    - limit (int): max items to return (default 5)

    Returns:
    - str: JSON string payload
    """
    try:
        t = (ticker or "").strip().upper()
        if not t:
            return json.dumps({"error": "Invalid ticker", "message": "Ticker cannot be empty."}, indent=2)

        stock = yf.Ticker(t)
        news = getattr(stock, "news", None) or []

        if not isinstance(news, list) or len(news) == 0:
            return json.dumps({"ticker": t, "news": [], "message": "No recent news returned."}, indent=2)

        items = []
        for n in news[: max(1, int(limit))]:
            c = n.get("content", {}) if isinstance(n, dict) else {}
            if isinstance(c, dict) and c:
                title = c.get("title", "") or ""
                publisher = (c.get("provider") or {}).get("displayName", "") if isinstance(c.get("provider"), dict) else ""
                pub_iso = c.get("pubDate") or c.get("displayTime")
                url = ""
                canon = c.get("canonicalUrl")
                click = c.get("clickThroughUrl")
                if isinstance(canon, dict):
                    url = canon.get("url") or ""
                if not url and isinstance(click, dict):
                    url = click.get("url") or ""
                if not url:
                    url = c.get("previewUrl") or ""

                epoch = _parse_iso_to_epoch(pub_iso)

                items.append(
                    {
                        "id": n.get("id") or c.get("id"),
                        "title": title,
                        "publisher": publisher,
                        "published": pub_iso,
                        "published_epoch": epoch,
                        "url": url,
                        "summary": c.get("summary", "") or "",
                        "content_type": c.get("contentType", "") or "",
                        "region": (canon or {}).get("region") if isinstance(canon, dict) else None,
                        "lang": (canon or {}).get("lang") if isinstance(canon, dict) else None,
                        "thumbnail": (
                            (c.get("thumbnail") or {}).get("originalUrl")
                            if isinstance(c.get("thumbnail"), dict)
                            else None
                        ),
                    }
                )
                continue

            title = n.get("title", "") if isinstance(n, dict) else ""
            publisher = n.get("publisher", "") if isinstance(n, dict) else ""
            ts = n.get("providerPublishTime") if isinstance(n, dict) else None
            url = (n.get("link") or n.get("url") or "") if isinstance(n, dict) else ""
            published_iso = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if isinstance(ts, (int, float)) and ts > 0
                else None
            )

            items.append(
                {
                    "id": n.get("id") if isinstance(n, dict) else None,
                    "title": title or "",
                    "publisher": publisher or "",
                    "published": published_iso,
                    "published_epoch": ts if isinstance(ts, (int, float)) else None,
                    "url": url,
                }
            )

        return json.dumps({"ticker": t, "news": items}, indent=2, ensure_ascii=False)

    except Exception as e:
        return json.dumps(
            {"error": str(e), "message": f"Could not retrieve recent news for ticker '{ticker}'."},
            indent=2,
        )

def get_peer_comparison(ticker: str) -> str:
    """
    Compare valuation metrics (P/E, P/B, Profit Margin) to hardcoded sector peers.
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    """
    try:
        stock = yf.Ticker(ticker)
        sector = stock.info.get('sector', 'N/A')

        # Hardcoded sector medians for comparison
        sector_medians = {
            'Technology': {'pe': 25, 'pb': 6, 'margin': 20},
            'Financial Services': {'pe': 12, 'pb': 1.5, 'margin': 25},
            'Healthcare': {'pe': 20, 'pb': 4, 'margin': 15},
            'Consumer Cyclical': {'pe': 22, 'pb': 4, 'margin': 10},
            'Industrials': {'pe': 18, 'pb': 2.5, 'margin': 8},
            'Consumer Defensive': {'pe': 20, 'pb': 3, 'margin': 12},
            'Energy': {'pe': 10, 'pb': 1.2, 'margin': 15},
            'Utilities': {'pe': 16, 'pb': 1.8, 'margin': 10},
            'Real Estate': {'pe': 28, 'pb': 1.5, 'margin': 18},
            'Materials': {'pe': 15, 'pb': 1.5, 'margin': 10},
            'Communication Services': {'pe': 20, 'pb': 3, 'margin': 15},
        }

        # Default medians if sector not found or hardcoded list doesn't cover
        medians = sector_medians.get(sector, {'pe': 18, 'pb': 3, 'margin': 12})

        company_pe = stock.info.get('trailingPE', 'N/A')
        company_pb = stock.info.get('priceToBook', 'N/A')
        company_profit_margin = stock.info.get('profitMargins', 'N/A')

        valuation_vs_peers = 'N/A'
        if isinstance(company_pe, (int, float)) and isinstance(medians['pe'], (int, float)) and company_pe != 0:
            valuation_vs_peers = 'Premium' if company_pe > medians['pe'] else 'Discount'

        comparison = {
            'ticker': ticker,
            'sector': sector,
            'company_pe': company_pe,
            'sector_median_pe': medians['pe'],
            'company_pb': company_pb,
            'sector_median_pb': medians['pb'],
            'company_margin': f"{company_profit_margin*100:.1f}%" if isinstance(company_profit_margin, (int, float)) else 'N/A',
            'sector_median_margin': f"{medians['margin']}%",
            'valuation_vs_peers': valuation_vs_peers
        }
        return json.dumps(comparison, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e), "message": f"Could not perform peer comparison for ticker '{ticker}'."})

# --- Agent Configuration ---

# Register our tools for internal execution
TOOLS = {
    'get_stock_overview': get_stock_overview,
    'get_financials': get_financials,
    'get_recent_news': get_recent_news,
    'get_peer_comparison': get_peer_comparison,
}

# Define tools for OpenAI function calling with proper input/output schemas
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_overview",
            "description": "Retrieve current price, valuation ratios, and key metrics for a stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_financials",
            "description": "Retrieve income statement, balance sheet, or cash flow data for the last 2 years.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"},
                    "statement": {"type": "string", "enum": ["income", "balance", "cashflow"], "description": "Type of financial statement"}
                },
                "required": ["ticker", "statement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_news",
            "description": "Retrieve recent news headlines for a stock ticker. Returns up to the last 5 articles.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_peer_comparison",
            "description": "Compare valuation metrics to sector peers (P/E, P/B, Profit Margin).",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol (e.g., AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    }
]

# Define the agent's system prompt - its persona, process, and desired output format
AGENT_SYSTEM_PROMPT = """You are a senior equity research analyst conducting fundamental analysis. You have access to financial data tools.
Your PROCESS is to:
1. First, get the stock overview to understand the company.
2. Check financials (income statement) for trends.
3. Get recent news for potential catalysts or risks.
4. Compare to peers for relative valuation context.
5. Synthesize everything into a structured investment brief.

Your OUTPUT FORMAT for the final brief must cover:
COMPANY OVERVIEW: Name, sector, market cap, current price, key ratios.
VALUATION: P/E, P/B, vs peers (noting premium/discount).
FINANCIAL TRENDS: Revenue growth, margin trajectory, and any notable income statement items.
CATALYSTS & RISKS: From news and analysis.
CONCLUSION: Attractiveness rating (Attractive/Neutral/Unattractive) with a 2-3 sentence justification.

Think step by step. Use tools to gather data before drawing conclusions. Never recommend a stock without checking the data first.
"""

# --- Agent Orchestration Functions ---

def run_agent(ticker: str, max_iterations: int = 10) -> dict:
    """
    Runs a fundamental screener agent on a ticker using a ReAct-like loop.
    The agent interacts with the OpenAI LLM, which decides whether to call a tool
    or provide a final response based on the AGENT_SYSTEM_PROMPT.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").
    - max_iterations (int): Maximum number of Thought-Action-Observation loops
                            to prevent infinite execution.

    Returns:
    - dict: A dictionary containing:
        - 'brief' (str): The final investment brief generated by the agent.
        - 'trace' (list): An audit trail of the agent's thoughts, actions, and observations.
        - 'iterations' (int): The total number of iterations taken.
        - 'tool_calls' (int): The total number of tool functions called.
    """
    client = get_openai_client() # Ensure client is initialized and retrieved

    messages = [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"Conduct a fundamental analysis of {ticker} and produce an investment brief."}
    ]
    trace = []
    tool_calls_count = 0

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2
        )
        msg = response.choices[0].message
        messages.append(msg)

        current_trace_entry = {
            'iteration': iteration,
            'thought': msg.content or '(reasoning leading to tool call)',
            'action': None,
            'observation': None
        }

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                current_trace_entry['action'] = f"{function_name}({function_args})"

                if function_name in TOOLS:
                    print(f"[Agent] Iteration {iteration}: Calling Tool: {function_name}({function_args})")
                    result = TOOLS[function_name](**function_args)
                    tool_calls_count += 1
                else:
                    result = json.dumps({"error": "Tool not found", "message": f"Agent tried to call an unregistered tool: {function_name}"})

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
                current_trace_entry['observation'] = result[:500] + "..." if len(result) > 500 else result
            trace.append(current_trace_entry)
        else:
            current_trace_entry['thought'] = 'Final synthesis'
            current_trace_entry['action'] = 'generate_brief'
            current_trace_entry['observation'] = msg.content[:500] if len(msg.content) > 500 else msg.content
            trace.append(current_trace_entry)

            return {
                'brief': msg.content,
                'trace': trace,
                'iterations': iteration + 1,
                'tool_calls': tool_calls_count
            }

    return {
        'brief': 'Max iterations reached without producing a final investment brief.',
        'trace': trace,
        'iterations': max_iterations,
        'tool_calls': tool_calls_count
    }


def validate_agent_output(result: dict) -> tuple[dict, int, int]:
    """
    Checks if the agent's brief meets minimum quality standards and compliance guardrails
    based on expected sections and tool usage.

    Parameters:
    - result (dict): The dictionary returned by run_agent containing 'brief', 'trace', etc.

    Returns:
    - tuple: A tuple containing:
        - checks (dict): A dictionary mapping check names to boolean pass/fail status.
        - passed_count (int): The number of checks that passed.
        - total_checks (int): The total number of checks performed.
    """
    brief = result['brief'].lower()

    checks = {
        'has_company_overview': any(kw in brief for kw in ['company overview', 'name', 'sector', 'market cap']),
        'has_valuation': any(kw in brief for kw in ['p/e', 'pe ratio', 'valuation', 'price-to-book', 'p/b']),
        'has_financial_trends': any(kw in brief for kw in ['revenue growth', 'margin trajectory', 'earnings', 'revenue', 'profit margin']),
        'has_catalysts_risks': any(kw in brief for kw in ['catalysts', 'risks', 'news', 'headlines']),
        'has_conclusion': any(kw in brief for kw in ['attractive', 'neutral', 'unattractive', 'recommendation', 'conclusion']),
        'used_data_tools': result.get('tool_calls', 0) >= 2,
        'within_iteration_limit': result.get('iterations', 0) <= 10
    }

    passed_count = sum(checks.values())
    total_checks = len(checks)

    return checks, passed_count, total_checks


def fixed_workflow(ticker: str) -> str:
    """
    Executes a predefined 4-step workflow for stock analysis without dynamic reasoning.
    It calls specific tools in a fixed order and then synthesizes the results in a single LLM call.

    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
    - str: The synthesized investment brief.
    """
    client = get_openai_client() # Ensure client is initialized and retrieved

    # Step 1: Get Stock Overview
    overview = get_stock_overview(ticker)

    # Step 2: Get Financial Statements (Income)
    financials = get_financials(ticker, 'income')

    # Step 3: Get Recent News
    news = get_recent_news(ticker)

    # Step 4: Get Peer Comparison
    peers = get_peer_comparison(ticker)

    # Synthesize all gathered data with a single LLM call
    prompt = f"""Based on the following data, produce an investment brief for {ticker}.

    The brief must follow this structure:
    COMPANY OVERVIEW: Name, sector, market cap, current price, key ratios.
    VALUATION: P/E, P/B, vs peers (noting premium/discount).
    FINANCIAL TRENDS: Revenue growth, margin trajectory, and any notable income statement items.
    CATALYSTS & RISKS: From news and analysis.
    CONCLUSION: Attractiveness rating (Attractive/Neutral/Unattractive) with a 2-3 sentence justification.

    OVERVIEW: {overview}
    FINANCIALS: {financials[:2000]}
    NEWS: {news}
    PEERS: {peers}
    """

    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    return response.choices[0].message.content


# --- Example Usage (for direct script execution/testing) ---
if __name__ == "__main__":
    print("--- Stock Analysis Module Demonstration ---")
    print("To run this demonstration, please ensure your OPENAI_API_KEY environment variable is set.")

    try:
        # Initialize the OpenAI client for the demo
        # In an actual app.py, this would be done once at the application's startup.
        initialize_openai_client(api_key=os.environ.get("OPENAI_API_KEY"))

        # --- Verifying individual tool functions ---
        print("\n--- Verifying tool functions ---")
        TEST_TICKER = "AAPL"
        print(f"\nOverview for {TEST_TICKER}:\n{get_stock_overview(TEST_TICKER)}")
        print(f"\nIncome Statement for {TEST_TICKER}:\n{get_financials(TEST_TICKER, 'income')}")
        print(f"\nRecent News for {TEST_TICKER}:\n{get_recent_news(TEST_TICKER)}")
        print(f"\nPeer Comparison for {TEST_TICKER}:\n{get_peer_comparison(TEST_TICKER)}")

        # --- Verifying run_agent function with a dummy run ---
        print("\n--- Verifying 'run_agent' function (dummy run) ---")
        dummy_ticker = "MSFT"
        dummy_max_iterations = 2
        print(f"Running a dummy agent test on {dummy_ticker} with max_iterations={dummy_max_iterations}. Expected to be incomplete.\n")
        dummy_agent_result = run_agent(dummy_ticker, max_iterations=dummy_max_iterations)
        print(f"Dummy agent run finished. Iterations: {dummy_agent_result['iterations']}, Tool calls: {dummy_agent_result['tool_calls']}.")
        print(f"Brief snippet: {dummy_agent_result['brief'][:200]}...")

        # --- Full agent run and validation ---
        TARGET_TICKER = "GOOG"
        print(f"\n--- Running full fundamental screener agent on {TARGET_TICKER} ---")
        agent_result = run_agent(TARGET_TICKER, max_iterations=10)

        print("\n" + "="*60)
        print(f"Agent completed in {agent_result['iterations']} iterations with {agent_result['tool_calls']} tool calls.")
        print("="*60)
        print("GENERATED INVESTMENT BRIEF:")
        print(agent_result['brief'])

        print("\n--- Running Guardrail Validation ---")
        agent_checks, agent_passed_count, agent_total_checks = validate_agent_output(agent_result)
        print("AGENT OUTPUT VALIDATION REPORT")
        print("=" * 40)
        for check, status in agent_checks.items():
            print(f" {'PASS' if status else 'FAIL'}: {check.replace('_', ' ').capitalize()}")
        print(f"\nScore: {agent_passed_count}/{agent_total_checks} checks passed.")
        if not agent_checks['used_data_tools']:
            print("WARNING: Agent made recommendation without fetching sufficient data!")

        print("\n" + "="*60)
        print("AGENT AUDIT TRAIL REPORT (Thought -> Action -> Observation Log)")
        print("="*60)
        for step in agent_result['trace']:
            print(f"--- Iteration {step['iteration']} ---")
            print(f"Thought: {step['thought']}")
            if step['action']:
                print(f"Action: {step['action']}")
            if step['observation']:
                print(f"Observation: {step['observation']}")

        # --- Fixed workflow run and comparison ---
        print(f"\n--- Running Fixed Workflow for Comparison for {TARGET_TICKER} ---")
        workflow_brief = fixed_workflow(TARGET_TICKER)

        print("\n" + "="*60)
        print("WORKFLOW VS. AGENT COMPARISON")
        print("="*60)

        agent_tool_calls_count = agent_result['tool_calls']
        workflow_tool_calls_count = 4 # Fixed workflow always calls all 4 tools

        comparison_data = {
            'Metric': [
                'Tool calls',
                'Can skip irrelevant tools?',
                'Can add extra tools if needed?',
                'Reasoning is logged?',
                'Predictable execution?',
                'Compliance-friendly?'
            ],
            'Workflow': [
                f'{workflow_tool_calls_count} (fixed)',
                'No',
                'No',
                'No (only final brief)',
                'Yes',
                'More'
            ],
            'Agent': [
                f'{agent_tool_calls_count} (dynamic)',
                'Yes',
                'Yes',
                'Yes (Thought-Action-Observation)',
                'Mostly',
                'More (with audit trail)'
            ]
        }

        # Print comparison table
        print(f"{comparison_data['Metric'][0]:<30s} {comparison_data['Workflow'][0]:>15s} {comparison_data['Agent'][0]:>15s}")
        print("-" * 60)
        for i in range(1, len(comparison_data['Metric'])):
            print(f"{comparison_data['Metric'][i]:<30s} {comparison_data['Workflow'][i]:>15s} {comparison_data['Agent'][i]:>15s}")

        print("\nFixed Workflow Brief Sample (first 500 chars):")
        print(workflow_brief[:500] + "...")
        print("\nAgent Brief Sample (first 500 chars):")
        print(agent_result['brief'][:500] + "...")

    except ValueError as e:
        print(f"Initialization Error: {e}")
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during demonstration: {e}")

