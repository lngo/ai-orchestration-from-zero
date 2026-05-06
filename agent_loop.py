"""
The reusable agent loop. Knows nothing about gold or analysts -
just runs whatever tools and system prompt you pass in.
"""

import json
from anthropic import Anthropic

client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment


def run_agent(
    system_prompt: str,
    user_query: str,
    tools: list,
    tool_executor,
    model: str = "claude-sonnet-4-6",
    max_iterations: int = 10,
    label: str = "agent",
) -> str:
    """Run an agent loop until the model emits a final text response.

    Args:
        system_prompt: Defines the agent's role and behaviour.
        user_query: The initial query to answer.
        tools: List of tool schemas (Claude's JSON format).
        tool_executor: Function (name, params) -> str result.
        model: Which Claude model to use.
        max_iterations: Safety cap to prevent infinite loops.
        label: Logging label so you can tell whose loop is running.

    Returns:
        The agent's final text response.
    """
    messages = [{"role": "user", "content": user_query}]

    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        # Append the assistant's response to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool use requested, the agent is done
        if response.stop_reason == "end_turn":
            for block in response.content:
                if block.type == "text":
                    return block.text
            return "(empty response)"

        # If tool use requested, execute every tool_use block and feed results back
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  [{label}] calling {block.name}({json.dumps(block.input)})")
                    result = tool_executor(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # Any other stop reason - bail out
        return f"(stopped: {response.stop_reason})"

    return "(reached max_iterations)"