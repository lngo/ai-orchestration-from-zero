"""
The orchestrator. Its tools are the sub-agents themselves.
This file exposes analyse_with_perspectives() - the public interface
that Tutorial 11 will import.
"""

from agent_loop import run_agent
from bull_analyst import consult_bull_analyst
from bear_analyst import consult_bear_analyst


ORCHESTRATOR_SYSTEM_PROMPT = """You are a balanced gold market analyst.

When asked about gold-related decisions or positions, you consult two specialist
sub-analysts:
  - consult_bull_analyst: argues the case FOR buying gold
  - consult_bear_analyst: argues the case AGAINST buying gold

Your workflow:
1. For any substantive gold question, consult BOTH analysts in parallel.
2. Read both arguments carefully. Identify where they agree and disagree.
3. Produce a balanced synthesis that names the strongest points on each side
   and offers a reasoned conclusion grounded in the evidence both raised.

For trivial questions ("what's the gold price today?") you may answer directly
without consulting analysts. Use judgement.

Do not invent numbers. Defer to the analysts for any factual claims about
prices or markets.

End your final response with a clear conclusion that takes a position, but
acknowledges the legitimate counterpoints."""


ORCHESTRATOR_TOOLS = [
    {
        "name": "consult_bull_analyst",
        "description": (
            "Consult the bull-case gold analyst. Use this when you need an "
            "evidence-based argument FOR buying or holding gold. The analyst has "
            "access to live gold prices and currency conversion. Returns a 4-6 "
            "sentence bull case with supporting data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific question to ask the bull analyst.",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "consult_bear_analyst",
        "description": (
            "Consult the bear-case gold analyst. Use this when you need an "
            "evidence-based argument AGAINST buying or holding gold. The analyst "
            "has access to live gold prices and currency conversion. Returns a "
            "4-6 sentence bear case with supporting data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific question to ask the bear analyst.",
                },
            },
            "required": ["query"],
        },
    },
]


def execute_orchestrator_tool(name: str, params: dict) -> str:
    """Dispatch the orchestrator's tool call to the matching sub-agent."""
    if name == "consult_bull_analyst":
        return consult_bull_analyst(params["query"])
    if name == "consult_bear_analyst":
        return consult_bear_analyst(params["query"])
    return f"Unknown tool: {name}"


# -----------------------------------------------------------
# PUBLIC INTERFACE
# -----------------------------------------------------------

def analyse_with_perspectives(query: str) -> str:
    """Top-level entry point. Run the orchestrator on a query.

    This is the function Tutorial 11 imports. Stable interface:
    string in, string out.
    """
    return run_agent(
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        user_query=query,
        tools=ORCHESTRATOR_TOOLS,
        tool_executor=execute_orchestrator_tool,
        model="claude-opus-4-7",   # Stronger model for synthesis
        max_iterations=8,
        label="orchestrator",
    )


# -----------------------------------------------------------
# RUNNER
# -----------------------------------------------------------

if __name__ == "__main__":
    queries = [
        "What's the current gold price in USD?",
        "Should I be buying gold right now? Give me a balanced view.",
        "I'm considering putting AUD 50,000 into a kilogram of gold. Worth it?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'='*70}")
        print(f"  QUERY {i}: {query}")
        print('='*70)
        result = analyse_with_perspectives(query)
        print(f"\n[orchestrator's final answer]\n{result}\n")