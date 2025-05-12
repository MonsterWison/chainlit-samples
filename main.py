from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
import chainlit as cl
import httpx

client = OpenAIProvider(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    http_client=httpx.AsyncClient(verify=False)
)

model = OpenAIModel(
    provider=client,
    model_name="deepseek/deepseek-r1"
)

response = Agent(
    model=model,
    messages=(
        'You are a helpful bot, you always reply in Traditional Chinese',
    ),
)

@cl.on_message
async def on_message(message: cl.Message):
    result = await response.run(message.content)
    await cl.Message(content=result.output).send()