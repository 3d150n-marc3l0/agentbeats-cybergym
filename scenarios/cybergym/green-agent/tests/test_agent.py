
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from a2a.types import Message, Task, TaskStatus, TaskState, TextPart, Part, Role
from a2a.server.tasks import TaskUpdater

# Import the code to test (assuming python path is set correctly or relative import)
# Since we are running from root, we need to adjust imports or sys.path
import sys
import os
sys.path.append(os.path.abspath("scenarios/cybergym/green-agent/src"))

from agent import Agent, EvalRequest, EvalConfig
from agent import CyberGymExecutionResult, PurpleResponse, ExplanationGrade, PoCGrade

# ============================================================================
# MOCKS
# ============================================================================

@pytest.fixture
def mock_messenger():
    with patch("agent.Messenger") as MockMessenger:
        messenger_instance = MockMessenger.return_value
        # Default mock behavior for talk_to_agent
        messenger_instance.talk_to_agent = AsyncMock(return_value=json.dumps({
            "explanation": "Test explanation",
            "poc": "dGVzdF9wb2M=" # "test_poc" in base64
        }))
        messenger_instance.reset = MagicMock()
        yield messenger_instance

@pytest.fixture
def mock_proxy():
    with patch("agent.CyberGymProxy") as MockProxy:
        proxy_instance = MockProxy.return_value
        # Default success
        proxy_instance.submit_poc = AsyncMock(return_value=CyberGymExecutionResult(
            task_id="test-task", exit_code=11, output="Segmentation Fault", error=None
        ))
        proxy_instance.submit_poc_to_fix = AsyncMock(return_value=CyberGymExecutionResult(
            task_id="test-task", exit_code=0, output="Safe exit", error=None
        ))
        yield proxy_instance

@pytest.fixture
def mock_cybergym_client():
    with patch("agent.CyberGymClient") as MockClient:
        client_instance = MockClient.return_value
        # Return dummy task content
        dummy_content = MagicMock()
        dummy_content.description = "Task desc"
        dummy_content.source = "int main() { ... }"
        dummy_content.error = ""
        client_instance.get_task_content.return_value = dummy_content
        yield client_instance

@pytest.fixture
def mock_litellm():
    with patch("agent.litellm.completion") as mock_completion:
        # We need to distinguish calls based on prompt content or order, 
        # but for simplicity we can return a mock that works for both or use side_effect.
        
        def side_effect(*args, **kwargs):
            messages = kwargs.get("messages", [])
            system_msg = messages[0]["content"] if messages else ""
            
            if "grade_explanation" in str(messages) or "vulnerability_identified" in system_msg:
                 # Return ExplanationGrade JSON
                content = json.dumps({
                    "vulnerability_identified": 25,
                    "root_cause_analysis": 25,
                    "exploitation_path": 25,
                    "fix_understanding": 25,
                    "total_score": 100,
                    "reasoning": "Perfect"
                })
            else:
                # Return PoCGrade JSON
                content = json.dumps({
                    "crash_quality": 20,
                    "specificity": 20,
                    "output_quality": 10,
                    "reproducibility": 10,
                    "severity_estimation": 10,
                    "total_score": 70,
                    "reasoning": "Perfect PoC"
                })

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content=content))]
            return mock_response
            
        mock_completion.side_effect = side_effect
        yield mock_completion

@pytest.fixture
def mock_updater():
    updater = AsyncMock(spec=TaskUpdater)
    updater.update_status = AsyncMock()
    updater.add_artifact = AsyncMock()
    updater.reject = AsyncMock()
    return updater

# ============================================================================
# TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_run_full_flow_success(mock_messenger, mock_proxy, mock_cybergym_client, mock_litellm, mock_updater):
    agent = Agent()
    agent.messenger = mock_messenger  # injecting mock manually since init creates new one
    agent.proxy = mock_proxy
    agent.cybergym_client = mock_cybergym_client
    
    # Input Message
    input_json = json.dumps({
        "config": {
            "task_id": "arvo:123",
            "difficulty": "level1",
            "llm_model": "gpt-4"
        },
        "participants": [
            {"role": "security_analyst", "endpoint": "http://purple:9000"}
        ]
    })
    
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=input_json))],
        message_id="msg-1",
    )

    await agent.run(msg, mock_updater)

    # Verifications
    # 1. Check Orchestration: Messenger called
    agent.messenger.talk_to_agent.assert_called_once()
    args, _ = agent.messenger.talk_to_agent.call_args
    assert "Task ID: arvo:123" in args[0]
    assert args[1] == "http://purple:9000"

    # 2. Check Submission: Proxy called
    agent.proxy.submit_poc.assert_called_once()
    agent.proxy.submit_poc_to_fix.assert_called_once()

    # 3. Check Grading: Artifact added
    mock_updater.add_artifact.assert_called_once()
    artifact_args = mock_updater.add_artifact.call_args[1]
    assert artifact_args["name"] == "Resultado final"
    
    # Verify score in artifact
    parts = artifact_args["parts"]
    score_text = parts[0].root.text
    assert "Puntaje total: 100" in score_text or "100/100" in score_text


@pytest.mark.asyncio
async def test_run_poc_failure(mock_messenger, mock_proxy, mock_cybergym_client, mock_litellm, mock_updater):
    agent = Agent()
    agent.messenger = mock_messenger
    agent.proxy = mock_proxy
    agent.cybergym_client = mock_cybergym_client

    # Simulate PoC execution failure
    mock_proxy.submit_poc = AsyncMock(return_value=CyberGymExecutionResult(
        task_id="test", exit_code=1, output="", error="Execution Error"
    ))

    # Input Message
    input_json = json.dumps({
        "config": { "task_id": "arvo:123", "difficulty": "level1" },
        "participants": [{"role": "security_analyst", "endpoint": "http://purple"}]
    })
    
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=input_json))],
        message_id="msg-1",
    )

    await agent.run(msg, mock_updater)

    # Verify score 0
    mock_updater.add_artifact.assert_called_once()
    parts = mock_updater.add_artifact.call_args[1]["parts"]
    assert "Puntaje total: 0" in parts[0].root.text


@pytest.mark.asyncio
async def test_validation_error(mock_updater):
    agent = Agent()
    
    # Missing participants
    input_json = json.dumps({
        "config": { "task_id": "arvo:123", "difficulty": "level1" },
        "participants": [] 
    })
    
    msg = Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=input_json))],
        message_id="msg-1",
    )

    await agent.run(msg, mock_updater)

    # Check rejection
    mock_updater.reject.assert_called()
    call_arg = mock_updater.reject.call_args[0][0] # The message object
    assert "At least one participant is required" in call_arg.parts[0].root.text

