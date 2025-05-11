from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
import chainlit as cl
import httpx

model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',
    client=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv("OPENROUTER_API_KEY"),
        http_client=httpx.AsyncClient(verify=False)
    ),
)

chat_agent = Agent(
    model=model,
    system_prompt='You are a helpful bot, you always reply in Traditional Chinese',
)

@cl.on_message
async def on_message(message: cl.Message):
    result = await chat_agent.run(message.content)
    await cl.Message(content=result.output).send()