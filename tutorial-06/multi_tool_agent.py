"""
Tutorial 6: Custom Tools and APIs
Each agent gets its own toolkit:
- Research agent: gold price + currency converter
- Analysis agent: portfolio calculator
Demonstrates tools with parameters and agent-specific toolkits.
"""

import os
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# ============================================================
# TOOLS — Research Agent
# ============================================================

@tool
def get_gold_price() -> str:
    """Fetch the current gold price in USD per troy ounce, including
    today's change and percentage change from the previous close."""

    api_key = os.environ.get("GOLDAPI_KEY")
    if not api_key:
        return "Error: GOLDAPI_KEY environment variable is not set."

    url = "https://www.goldapi.io/api/XAU/USD"
    headers = {"x-access-token": api_key}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()

        return (
            f"Gold price: ${data['price']:.2f} per troy ounce. "
            f"Previous close: ${data['prev_close_price']:.2f}. "
            f"Change: ${data['ch']:+.2f} ({data['chp']:+.2f}%). "
            f"Day range: ${data['low_price']:.2f} – ${data['high_price']:.2f}. "
            f"24K per gram: ${data['price_gram_24k']:.2f}."
        )
    except requests.RequestException as e:
        return f"Error fetching gold price: {e}"
    except (KeyError, TypeError) as e:
        return f"Error parsing gold price data: {e}"


@tool
def convert_currency(amount: float, to_currency: str) -> str:
    """Convert a USD amount to another currency using live exchange rates.
    
    Args:
        amount: The amount in USD to convert.
        to_currency: The target currency code (e.g., EUR, GBP, AUD, JPY, CAD).
    """

    to_currency = to_currency.upper().strip()

    try:
        url = f"https://api.frankfurter.dev/v1/latest?base=USD&symbols={to_currency}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        if to_currency not in data.get("rates", {}):
            return f"Error: Currency '{to_currency}' not found. Use standard codes like EUR, GBP, AUD, JPY, CAD."

        rate = data["rates"][to_currency]
        converted = amount * rate

        return (
            f"${amount:,.2f} USD = {converted:,.2f} {to_currency} "
            f"(rate: 1 USD = {rate:.4f} {to_currency}, as of {data['date']})"
        )
    except requests.RequestException as e:
        return f"Error converting currency: {e}"


# ============================================================
# TOOLS — Analysis Agent
# ============================================================

@tool
def calculate_portfolio(ounces: float, price_per_ounce: float) -> str:
    """Calculate the total value of a gold portfolio.
    
    Args:
        ounces: Number of troy ounces of gold held.
        price_per_ounce: Current price per troy ounce in USD.
    """

    total_usd = ounces * price_per_ounce
    per_gram = price_per_ounce / 31.1035  # troy ounce to grams

    return (
        f"Portfolio: {ounces:.2f} troy ounces at ${price_per_ounce:,.2f}/oz. "
        f"Total value: ${total_usd:,.2f} USD. "
        f"Per gram value: ${per_gram:,.2f}. "
        f"At 10% gain: ${total_usd * 1.10:,.2f}. "
        f"At 10% loss: ${total_usd * 0.90:,.2f}."
    )


# ============================================================
# LLMs — each agent gets its own tools
# ============================================================

research_tools = [get_gold_price, convert_currency]
analysis_tools = [calculate_portfolio]

research_llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(research_tools)
analysis_llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(analysis_tools)


# ============================================================
# SYSTEM PROMPTS
# ============================================================

RESEARCH_PROMPT = """You are a Gold Market Research Agent. Your job is to gather 
accurate, factual data about gold prices and currency conversions.

You have two tools:
1. get_gold_price — fetches the current gold price in USD
2. convert_currency — converts USD amounts to other currencies

Use these tools to answer data questions. Present the raw data clearly 
without adding analysis or opinions. If the user asks about gold prices 
in another currency, fetch the gold price first, then convert it."""

ANALYSIS_PROMPT = """You are a Gold Market Analysis Agent. Your job is to interpret 
gold price data and provide insightful analysis.

You have one tool:
1. calculate_portfolio — calculates the value of a gold holding

Use this tool when the user mentions specific gold holdings (e.g., "I have 5 ounces").
For general analysis, work with the data already in the conversation.
Keep your analysis concise and actionable. Think like a financial analyst."""


# ============================================================
# NODES
# ============================================================

def research_agent(state: MessagesState) -> MessagesState:
    msg_count = len(state["messages"])
    print(f"  → [Research Agent] ({msg_count} messages in memory)")
    messages = [SystemMessage(content=RESEARCH_PROMPT)] + state["messages"]
    response = research_llm.invoke(messages)
    return {"messages": [response]}


def analysis_agent(state: MessagesState) -> MessagesState:
    msg_count = len(state["messages"])
    print(f"  → [Analysis Agent] ({msg_count} messages in memory)")
    messages = [SystemMessage(content=ANALYSIS_PROMPT)] + state["messages"]
    messages.append(HumanMessage(content="Based on the data and conversation above, provide your analysis."))
    response = analysis_llm.invoke(messages)
    return {"messages": [response]}


# Tool nodes — one per agent's toolkit
research_tool_node = ToolNode(research_tools)
analysis_tool_node = ToolNode(analysis_tools)


# ============================================================
# ROUTING
# ============================================================

DATA_KEYWORDS = ["price", "cost", "worth", "how much", "current",
                 "today", "right now", "latest", "check", "convert",
                 "currency", "euro", "eur", "pound", "gbp", "yen",
                 "jpy", "aud", "cad", "yesterday"]

ANALYSIS_KEYWORDS = ["analyse", "analyze", "analysis", "tell us", "what does",
                     "insight", "outlook", "investment", "should i", "recommend",
                     "opinion", "think about", "suggest", "implication", "mean",
                     "portfolio", "holding", "own", "have", "bought"]


def route_question(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    content = last_message.content.lower() if hasattr(last_message, 'content') else ""

    needs_data = any(kw in content for kw in DATA_KEYWORDS)

    if needs_data:
        print("  → [Router] Data needed → Research Agent")
        return "research_agent"
    print("  → [Router] Analysis → Analysis Agent")
    return "analysis_agent"


def after_research(state: MessagesState) -> str:
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "research_tools"

    # Check if original question also needs analysis
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            if any(kw in content for kw in ANALYSIS_KEYWORDS):
                print("  → [Handoff] Research complete → Analysis Agent")
                return "analysis_agent"
            break

    print("  → [Complete] Data delivered")
    return END


def after_analysis(state: MessagesState) -> str:
    last_message = state["messages"][-1]

    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "analysis_tools"

    return END


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(MessagesState)

# Nodes
graph.add_node("research_agent", research_agent)
graph.add_node("analysis_agent", analysis_agent)
graph.add_node("research_tools", research_tool_node)
graph.add_node("analysis_tools", analysis_tool_node)

# Routing from START
graph.add_conditional_edges(
    START, route_question,
    {"research_agent": "research_agent", "analysis_agent": "analysis_agent"}
)

# After research agent: tool, analyst, or done
graph.add_conditional_edges(
    "research_agent", after_research,
    {"research_tools": "research_tools", "analysis_agent": "analysis_agent", END: END}
)

# After research tools: back to research agent
graph.add_edge("research_tools", "research_agent")

# After analysis agent: tool or done
graph.add_conditional_edges(
    "analysis_agent", after_analysis,
    {"analysis_tools": "analysis_tools", END: END}
)

# After analysis tools: back to analysis agent
graph.add_edge("analysis_tools", "analysis_agent")

# Compile with memory
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "multi-tool-session-1"}}

    print("=" * 60)
    print("  AI Orchestration Lab — Multi-Tool Agent System")
    print("=" * 60)

    # Test 1: Research agent with gold price tool (familiar)
    print("\nQ1: What is the current gold price?")
    print("-" * 40)
    result = app.invoke(
        {"messages": [HumanMessage(content="What is the current gold price?")]},
        config=config
    )
    print(result["messages"][-1].content)

    print()

    # Test 2: Research agent with currency tool (new — tool with parameters)
    print("Q2: How much is $10,000 in euros?")
    print("-" * 40)
    result = app.invoke(
        {"messages": [HumanMessage(content="How much is $10,000 in euros?")]},
        config=config
    )
    print(result["messages"][-1].content)

    print()

    # Test 3: Research agent using BOTH tools in one question
    print("Q3: What's the gold price in British pounds?")
    print("-" * 40)
    result = app.invoke(
        {"messages": [HumanMessage(content="What's the gold price in British pounds?")]},
        config=config
    )
    print(result["messages"][-1].content)

    print()

    # Test 4: Analysis agent with portfolio tool (new — tool with parameters)
    print("Q4: I own 3 ounces of gold. What's my portfolio worth?")
    print("-" * 40)
    result = app.invoke(
        {"messages": [HumanMessage(content="I own 3 ounces of gold. What's my portfolio worth?")]},
        config=config
    )
    print(result["messages"][-1].content)