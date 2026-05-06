"""
The bull-case gold analyst. A sub-agent that argues FOR buying gold,
using the gold tools to gather supporting evidence.
"""

from agent_loop import run_agent
from gold_tools import GOLD_TOOLS, execute_gold_tool


BULL_SYSTEM_PROMPT = """You are a bull-case gold analyst. Your job is to argue
the strongest case for buying gold based on current market data.

You have access to gold price tools. Use them to gather facts. Do not invent
numbers - if you need a price or conversion, call the tool.

Build your argument from data. Cite specific figures. Acknowledge counterpoints
briefly only if directly relevant. Be confident but evidence-based.

Keep your final answer concise: a clear bull case in 4-6 sentences, with the
key data points that support it."""


def consult_bull_analyst(query: str) -> str:
    """Run the bull analyst sub-agent on a query and return its argument."""
    return run_agent(
        system_prompt=BULL_SYSTEM_PROMPT,
        user_query=query,
        tools=GOLD_TOOLS,
        tool_executor=execute_gold_tool,
        label="bull",
    )