"""
Tutorial 6: Interactive Multi-Tool Agent System
Research agent has gold price + currency converter.
Analysis agent has portfolio calculator.
"""

import os
import uuid
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


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


@tool
def calculate_portfolio(ounces: float, price_per_ounce: float) -> str:
    """Calculate the total value of a gold portfolio.
    
    Args:
        ounces: Number of troy ounces of gold held.
        price_per_ounce: Current price per troy ounce in USD.
    """

    total_usd = ounces * price_per_ounce
    per_gram = price_per_ounce / 31.1035

    return (
        f"Portfolio: {ounces:.2f} troy ounces at ${price_per_ounce:,.2f}/oz. "
        f"Total value: ${total_usd:,.2f} USD. "
        f"Per gram value: ${per_gram:,.2f}. "
        f"At 10% gain: ${total_usd * 1.10:,.2f}. "
        f"At 10% loss: ${total_usd * 0.90:,.2f}."
    )


research_tools = [get_gold_price, convert_currency]
analysis_tools = [calculate_portfolio]

research_llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(research_tools)
analysis_llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(analysis_tools)

RESEARCH_PROMPT = """You are a Gold Market Research Agent. Your job is to gather 
accurate, factual data about gold prices and currency conversions.
You have two tools: get_gold_price and convert_currency.
Use these tools to answer data questions. Present raw data clearly 
without adding analysis. If asked about gold prices in another currency, 
fetch the gold price first, then convert it."""

ANALYSIS_PROMPT = """You are a Gold Market Analysis Agent. Your job is to interpret 
gold price data and provide insightful analysis.
You have one tool: calculate_portfolio — use it when the user mentions specific 
gold holdings. For general analysis, work with conversation data.
Keep analysis concise and actionable."""


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


research_tool_node = ToolNode(research_tools)
analysis_tool_node = ToolNode(analysis_tools)

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
    if any(kw in content for kw in DATA_KEYWORDS):
        print("  → [Router] Data needed → Research Agent")
        return "research_agent"
    print("  → [Router] Analysis → Analysis Agent")
    return "analysis_agent"


def after_research(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "research_tools"
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


graph = StateGraph(MessagesState)
graph.add_node("research_agent", research_agent)
graph.add_node("analysis_agent", analysis_agent)
graph.add_node("research_tools", research_tool_node)
graph.add_node("analysis_tools", analysis_tool_node)

graph.add_conditional_edges(
    START, route_question,
    {"research_agent": "research_agent", "analysis_agent": "analysis_agent"}
)
graph.add_conditional_edges(
    "research_agent", after_research,
    {"research_tools": "research_tools", "analysis_agent": "analysis_agent", END: END}
)
graph.add_edge("research_tools", "research_agent")
graph.add_conditional_edges(
    "analysis_agent", after_analysis,
    {"analysis_tools": "analysis_tools", END: END}
)
graph.add_edge("analysis_tools", "analysis_agent")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 60)
    print("  AI Orchestration Lab — Multi-Tool Agent System")
    print(f"  Session: {session_id}")
    print("  Research: gold price + currency converter")
    print("  Analysis: portfolio calculator")
    print("  Type 'quit' to exit")
    print("=" * 60)

    while True:
        print()
        question = input("You: ").strip()

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        print(f"\nAgent: {result['messages'][-1].content}")