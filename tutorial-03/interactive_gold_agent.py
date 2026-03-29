"""
Tutorial 3: Interactive Gold Price Agent
Same tool-using agent, with a loop so you can ask multiple questions.
"""

import os
import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode


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

agent_app = graph.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Orchestration Lab — Interactive Gold Price Agent")
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

        result = agent_app.invoke(
            {"messages": [("human", question)]}
        )
        print(f"\nAgent: {result['messages'][-1].content}")