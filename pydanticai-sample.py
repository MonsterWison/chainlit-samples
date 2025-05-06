from pydantic_ai import Agent
from pydantic_ai.models import OpenAImodel
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAImodel(
    'anthropic/claude-3.5-h

agent = Agent('openai:gpt-4o')

result_sync = agent.run_sync('What is the capital of Italy?')
print(result_sync.output)
#> Rome


async def main():
    result = await agent.run('What is the capital of France?')
    print(result.output)
    #> Paris

    async with agent.run_stream('What is the capital of the UK?') as response:
        print(await response.get_output())
        #> London