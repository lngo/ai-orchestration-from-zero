"""
Tutorial 1: Interactive Agent
Same hello world agent, but with a loop so you can ask multiple questions.
"""

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict


class AgentState(TypedDict):
    question: str
    answer: str


llm = ChatAnthropic(model="claude-sonnet-4-20250514")


def think(state: AgentState) -> AgentState:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_edge(START, "think")
graph.add_edge("think", END)
agent = graph.compile()


if __name__ == "__main__":
    print("=" * 60)
    print("  AI Orchestration Lab — Interactive Agent")
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

        result = agent.invoke({"question": question})
        print(f"\nAgent: {result['answer']}")