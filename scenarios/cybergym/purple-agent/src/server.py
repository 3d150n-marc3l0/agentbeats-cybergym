import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9019, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument("--agent_llm", type=str, choices=["gemini", "open-ia", "claude", "grok"], default="gemini", help="LLM model to use")
    args = parser.parse_args()

    # Add environment variable for LLM model
    #os.environ.setdefault("CYBERGYM_AGENT_LLM", args.agent_llm)

    # Purple Agent Card Configuration
    # See: https://a2a-protocol.org/latest/tutorials/python/3-agent-skills-and-card/
    
    skill = AgentSkill(
        id="purple-agent-hybrid-security",
        name="Hybrid Offensive/Defensive Cybersecurity",
        description=(
            "Purple Agent combines red team (offensive) and blue team (defensive) "
            "capabilities for comprehensive security testing and protection. "
            "Uses an LLM to reason through CyberGym tasks and execute actions via tools."
        ),
        tags=[
            "cybersecurity",
            "purple-team",
            "offensive-security",
            "defensive-security",
            "vulnerability-assessment",
            "incident-response",
            "cybergym",
            "a2a-coordination",
            "llm-reasoning"
        ],
        examples=[
            "Detect attacks on web-server-01",
            "Scan vulnerabilities on target system",
            "Execute PoC exploit against test environment",
            "Respond to security incident on database server",
            "Coordinate with red team agent at http://red-agent:9010",
            "Perform fuzzing on application binary"
        ]
    )

    agent_card = AgentCard(
        name="Purple Agent",
        description=(
            "Hybrid offensive/defensive cybersecurity agent for the AgentX-AgentBeats "
            "competition (CyberGym benchmark). Combines red team and blue team capabilities "
            "to test vulnerabilities, detect attacks, respond to incidents, and coordinate "
            "with other security agents via A2A protocol. Powered by LLM reasoning."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.1.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
