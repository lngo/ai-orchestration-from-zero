"""
Tutorial 1: Hello World Agent
A minimal LangGraph agent that sends a question to Claude and prints the answer.
This is the simplest possible orchestration graph: one node, one edge.
"""

from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from typing import TypedDict


# --- State ---
# This defines what data flows through the agent graph.
# For now: a question goes in, an answer comes out.
class AgentState(TypedDict):
    question: str
    answer: str


# --- LLM ---
# Claude Sonnet is the model powering our agent's thinking.
llm = ChatAnthropic(model="claude-sonnet-4-20250514")


# --- Node ---
# A node is a single step in the graph. This one sends the question to Claude.
def think(state: AgentState) -> AgentState:
    response = llm.invoke(state["question"])
    return {"answer": response.content}


# --- Graph ---
# Build the graph: START -> think -> END
# This is the simplest possible graph. Later tutorials will add
# branching, loops, tools, and multiple agents as separate nodes.
graph = StateGraph(AgentState)
graph.add_node("think", think)
graph.add_edge(START, "think")
graph.add_edge("think", END)

# Compile the graph into a runnable agent
agent = graph.compile()


# --- Run ---
if __name__ == "__main__":
    print("=" * 60)
    print("  AI Orchestration Lab — Hello World Agent")
    print("=" * 60)
    print()

    # First test: a simple question
    result = agent.invoke({"question": "What is AI orchestration in one sentence?"})
    print("Q: What is AI orchestration in one sentence?")
    print(f"A: {result['answer']}")
    print()

    # Second test: something that requires reasoning
    result = agent.invoke({
        "question": "What is the difference between an AI agent and a chatbot? "
                    "Give me three key differences."
    })
    print("Q: What is the difference between an AI agent and a chatbot?")
    print(f"A: {result['answer']}")