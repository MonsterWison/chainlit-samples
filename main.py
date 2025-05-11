from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from dotenv import load_dotenv
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

provider = OpenAIProvider(
    api_key=os.getenv("OPENROUTER_API_KEY")
)

model = OpenAIModel(
    model_name='gpt-3.5-turbo',
    provider=provider
)

chat_agent = Agent(
    model=model,
    system_prompt='You are a helpful bot, you always reply in Traditional Chinese',
)

@cl.on_message
async def on_message(message: cl.Message):
    result = await chat_agent.run(message.content)
    await cl.Message(content=result.output).send()