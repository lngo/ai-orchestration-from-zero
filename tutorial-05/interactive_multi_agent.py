"""
Tutorial 5: Interactive Multi-Agent Gold System
Two agents collaborate with memory across the conversation.
"""

import os
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


tools = [get_gold_price]
research_llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)
analysis_llm = ChatAnthropic(model="claude-sonnet-4-20250514")

RESEARCH_PROMPT = """You are a Gold Market Research Agent. Your job is to gather 
accurate, factual gold price data. When asked about gold prices, ALWAYS use your 
get_gold_price tool to fetch live data. Be thorough and include all available 
data points. Present the raw data clearly without adding analysis or opinions."""

ANALYSIS_PROMPT = """You are a Gold Market Analysis Agent. Your job is to interpret 
gold price data and provide insightful analysis. You do NOT have access to live 
data tools — you work with the data already gathered in the conversation. 
Provide observations about price movement significance, volatility, and brief 
outlook. Keep your analysis concise and actionable."""


def research_agent(state: MessagesState) -> MessagesState:
    msg_count = len(state["messages"])
    print(f"  → [Research Agent] Gathering data... ({msg_count} messages in memory)")
    messages = [SystemMessage(content=RESEARCH_PROMPT)] + state["messages"]
    response = research_llm.invoke(messages)
    return {"messages": [response]}


def analysis_agent(state: MessagesState) -> MessagesState:
    msg_count = len(state["messages"])
    print(f"  → [Analysis Agent] Generating insights... ({msg_count} messages in memory)")
    messages = [SystemMessage(content=ANALYSIS_PROMPT)] + state["messages"]
    messages.append(HumanMessage(content="Based on the data and conversation above, provide your market analysis."))
    response = analysis_llm.invoke(messages)
    return {"messages": [response]}


tool_node = ToolNode(tools)

DATA_KEYWORDS = ["price", "cost", "worth", "how much", "current",
                 "today", "right now", "latest", "check"]

ANALYSIS_KEYWORDS = ["analyse", "analyze", "analysis", "tell us", "what does",
                     "insight", "outlook", "investment", "should i", "recommend",
                     "opinion", "think about", "suggest", "implication", "mean"]


def route_question(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    content = last_message.content.lower() if hasattr(last_message, 'content') else ""
    if any(kw in content for kw in DATA_KEYWORDS):
        print("  → [Router] Data needed → Research Agent")
        return "research_agent"
    print("  → [Router] Analysis only → Analysis Agent")
    return "analysis_agent"


def after_research(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            content = msg.content.lower()
            if any(kw in content for kw in ANALYSIS_KEYWORDS):
                print("  → [Handoff] Research complete → Analysis Agent")
                return "analysis_agent"
            break
    print("  → [Complete] Data delivered — no analysis needed")
    return END


graph = StateGraph(MessagesState)
graph.add_node("research_agent", research_agent)
graph.add_node("analysis_agent", analysis_agent)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    START, route_question,
    {"research_agent": "research_agent", "analysis_agent": "analysis_agent"}
)
graph.add_conditional_edges(
    "research_agent", after_research,
    {"tools": "tools", "analysis_agent": "analysis_agent", END: END}
)
graph.add_edge("tools", "research_agent")
graph.add_edge("analysis_agent", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    import uuid
    session_id = f"session-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    print("=" * 60)
    print("  AI Orchestration Lab — Multi-Agent Gold System")
    print(f"  Session: {session_id}")
    print("  Research agent gathers data. Analysis agent interprets it.")
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