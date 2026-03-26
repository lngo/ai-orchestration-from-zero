# AI Orchestration from Zero

Code and examples for the **AI Orchestration from Zero** tutorial series on [AI For Your Work](https://www.youtube.com/@AIForYourWork).

Learn all three approaches to AI orchestration — from developer-controlled graphs to autonomous agents to LLM-orchestrated teams — with hands-on code you can run yourself.

## Tutorials

| # | Tutorial | What you build | Code |
|---|----------|---------------|------|
| 1 | [The AI Orchestration Landscape](https://www.youtube.com/@AIForYourWork) | Conceptual overview — no code | — |
| 2 | Setup and Hello World | A working LangGraph agent that answers questions via Claude | [tutorial-02/](tutorial-02/) |
| 3 | First Tool-Using Agent | A gold price checker that fetches live data | Coming soon |
| 4 | State and Memory | An agent that remembers context between runs | Coming soon |
| 5 | Multi-Agent Basics | Two agents collaborating on a research task | Coming soon |
| 6 | Custom Tools and APIs | Agents that connect to real-world services | Coming soon |
| 7 | Security Fundamentals | Securing your agent pipeline against real threats | Coming soon |
| 8 | Claude Code Deep Dive | Autonomous coding and task execution | Coming soon |
| 9 | Autonomous Trade-offs | Controlled vs. autonomous comparison | Coming soon |
| 10 | Agent Teams | LLM-directed delegation and coordination | Coming soon |
| 11 | Hybrid Architectures | Combining all three orchestration approaches | Coming soon |

## Prerequisites

- Ubuntu 24.04 LTS (VM works fine — VMware, VirtualBox, cloud instance)
- Internet connection
- An Anthropic API key — get one at [console.anthropic.com](https://console.anthropic.com)
- Basic command line comfort

No prior AI or machine learning experience required.

## Quick start

```bash
# Clone this repo
git clone https://github.com/YOUR_USERNAME/ai-orchestration-from-zero.git
cd ai-orchestration-from-zero

# Set up the environment (detailed walkthrough in Tutorial 2)
cd tutorial-02
python3 -m venv venv
source venv/bin/activate
pip install langgraph langchain-anthropic

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"

# Run the hello world agent
python3 hello_agent.py
```

## Series structure

The series covers three approaches to AI orchestration, in order:

**Phase 1: Developer-controlled orchestration (Tutorials 2–7)**
You define the workflow as a graph. The LLM executes within it. Built with LangGraph.

**Phase 2: LLM-as-agent (Tutorials 8–9)**
The LLM decides the steps on its own. Covers Claude Code and autonomous agent patterns.

**Phase 3: LLM-as-orchestrator (Tutorials 10–11)**
The LLM coordinates other LLM agents. Agent teams, delegation, and hybrid architectures.

## Links

- YouTube: [@AIForYourWork](https://www.youtube.com/@AIForYourWork)
- Channel: [AI For Your Work](https://www.youtube.com/@AIForYourWork)

## License

Code in this repository is provided for educational purposes. Feel free to use it in your own projects.
