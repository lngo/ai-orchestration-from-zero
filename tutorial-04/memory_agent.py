"""
Tutorial 4: Gold Price Agent WITH Memory
Uses MemorySaver to persist conversation state between invoke() calls.
The agent remembers what you asked before and can answer follow-up questions.
"""

import os
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
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
llm = ChatAnthropic(model="claude-sonnet-4-20250514").bind_tools(tools)


def agent(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_use_tool(state: MessagesState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_node("tools", tool_node)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tool, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

# ✅ ADD MEMORY: create a checkpointer and compile with it
checkpointer = MemorySaver()
agent_app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("=" * 60)
    print("  WITH MEMORY — the agent remembers!")
    print("=" * 60)
    print()

    # The config tells LangGraph which conversation thread to use.
    # Same thread_id = same conversation = shared memory.
    config = {"configurable": {"thread_id": "gold-session-1"}}

    # Question 1: Get the gold price
    print("Q1: What is the current gold price?")
    print("-" * 40)
    result = agent_app.invoke(
        {"messages": [("human", "What is the current gold price?")]},
        config=config
    )
    print(result["messages"][-1].content)
    print()

    # Question 2: Same question as the no-memory test — NOW it works!
    print("Q2: What did I just ask you?")
    print("-" * 40)
    result = agent_app.invoke(
        {"messages": [("human", "What did I just ask you?")]},
        config=config
    )
    print(result["messages"][-1].content)
    print()

    # Question 3: A follow-up that uses context from Q1
    print("Q3: How much would 10 ounces cost at that price?")
    print("-" * 40)
    result = agent_app.invoke(
        {"messages": [("human", "How much would 10 ounces cost at that price?")]},
        config=config
    )
    print(result["messages"][-1].content)