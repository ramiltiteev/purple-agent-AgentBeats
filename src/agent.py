import os
from openai import AsyncOpenAI
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from messenger import Messenger

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        self.history = []

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )

        user_text = get_message_text(message)
        self.history.append({"role": "user", "content": user_text})

        response = await client.chat.completions.create(
            model="qwen/qwen3.6-plus:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer service agent. "
                        "Help users with their requests clearly and concisely. "
                        "Use the tools available to complete tasks. "
                        "Always be polite and solution-oriented."
                    ),
                },
                *self.history,
            ],
        )

        reply = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": reply})

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=reply))],
            name="response",
        )