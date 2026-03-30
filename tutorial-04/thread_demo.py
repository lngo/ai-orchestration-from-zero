"""
Tutorial 4: Thread ID Demo
Shows how different thread_ids create separate, independent conversations.
"""

from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver


llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def agent(state: MessagesState) -> MessagesState:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

checkpointer = MemorySaver()
agent_app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("=" * 60)
    print("  Thread ID Demo — separate conversations")
    print("=" * 60)

    # Thread 1: Talk about Python
    thread1 = {"configurable": {"thread_id": "thread-python"}}
    print("\n--- Thread 1: Python ---")
    print("You: My favourite programming language is Python.")
    result = agent_app.invoke(
        {"messages": [("human", "My favourite programming language is Python.")]},
        config=thread1
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Thread 2: Talk about cooking
    thread2 = {"configurable": {"thread_id": "thread-cooking"}}
    print("\n--- Thread 2: Cooking ---")
    print("You: My favourite dish to cook is pasta carbonara.")
    result = agent_app.invoke(
        {"messages": [("human", "My favourite dish to cook is pasta carbonara.")]},
        config=thread2
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Back to Thread 1: Ask a generic follow-up
    print("\n--- Thread 1 (continued): Python ---")
    print("You: What do you recommend I try next?")
    result = agent_app.invoke(
        {"messages": [("human", "What do you recommend I try next?")]},
        config=thread1
    )
    print(f"Agent: {result['messages'][-1].content}")

    # Back to Thread 2: Same exact question — different context
    print("\n--- Thread 2 (continued): Cooking ---")
    print("You: What do you recommend I try next?")
    result = agent_app.invoke(
        {"messages": [("human", "What do you recommend I try next?")]},
        config=thread2
    )
    print(f"Agent: {result['messages'][-1].content}")

    print()
    print("Same question, different answers — each thread remembers its own context.")