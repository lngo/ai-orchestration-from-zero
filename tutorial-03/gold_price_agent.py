"""
Tutorial 3: Gold Price Agent
A LangGraph agent that fetches live gold prices using a tool.
This introduces: tools, ToolNode, conditional edges, and message-based state.
"""

import os
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


# --- Tool ---
# This is the function the agent can call to get live gold price data.
# The @tool decorator registers it as a tool that Claude can use.
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


# --- LLM ---
# Bind the tools to Claude so it knows they are available.
tools = [get_gold_price]
llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)


# --- Nodes ---
# The agent node: sends the conversation to Claude (with tool access).
def agent(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


# The tool node: executes any tool calls that Claude requested.
tool_node = ToolNode(tools)


# --- Conditional edge ---
# After the agent runs, check: did Claude request a tool call?
# If yes → route to the tool node. If no → we're done.
def should_use_tool(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


# --- Graph ---
# Build the graph with the agent-tool loop.
graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

agent_app = graph.compile()


# --- Run ---
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Orchestration Lab — Gold Price Agent")
    print("=" * 60)
    print()

    # Test 1: A question that requires the tool
    print("Q: What is the current gold price?")
    print("-" * 40)
    result = agent_app.invoke(
        {"messages": [("human", "What is the current gold price? Give me a brief market summary.")]}
    )
    print(result["messages"][-1].content)
    print()

    # Test 2: A question that does NOT require the tool
    print("Q: What factors affect the gold price?")
    print("-" * 40)
    result = agent_app.invoke(
        {"messages": [("human", "What factors typically affect the gold price?")]}
    )
    print(result["messages"][-1].content)