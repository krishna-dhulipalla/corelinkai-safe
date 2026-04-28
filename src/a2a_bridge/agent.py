from uuid import uuid4

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Role, TaskState
from a2a.utils import new_agent_text_message

from adapters.pibench import data_part, extract_request
from runtime.core import PolicyCaseRuntime


class Agent:
    def __init__(self, runtime: PolicyCaseRuntime | None = None):
        self.runtime = runtime or PolicyCaseRuntime()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Handle Pi-Bench A2A DataPart messages and normal text A2A messages."""
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        inbound = extract_request(message)
        response = await self.runtime.handle(inbound)

        await updater.update_status(
            TaskState.completed,
            Message(
                kind="message",
                role=Role.agent,
                parts=[data_part(response.data)],
                message_id=uuid4().hex,
            ),
        )
