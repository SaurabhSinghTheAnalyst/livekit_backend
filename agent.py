import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli
from livekit.agents.llm import ChatChunk
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import deepgram, openai, silero

load_dotenv()

class FunctionAgent(Agent):
    """A simple LiveKit voice assistant agent."""

    def __init__(self):
        super().__init__(
            instructions="""
                You are a helpful assistant communicating through voice.
            """,
            stt=deepgram.STT(),
            llm=openai.LLM(model="gpt-4o"),
            tts=openai.TTS(),
            vad=silero.VAD.load(),
            allow_interruptions=True
        )

    async def llm_node(self, chat_ctx, tools, model_settings):
        async for chunk in super().llm_node(chat_ctx, tools, model_settings):
            yield chunk

async def entrypoint(ctx: JobContext):
    """Main entrypoint for the LiveKit agent application."""
    agent = FunctionAgent()
    await ctx.connect()
    session = AgentSession()
    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))