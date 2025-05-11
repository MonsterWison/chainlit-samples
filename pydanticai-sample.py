from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import httpx

model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key="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",
        http_client=httpx.AsyncClient(verify=False)
    ),
)

sample_agent = Agent(
    model=model,
    # 'Be concise, reply with one sentence.' is enough for some models (like openai) to use
    # the below tools appropriately, but others like anthropic and gemini require a bit more direction.
    system_prompt=(
        'please answer everything in traditional chinese'
    ),
)
result_sync = sample_agent.run_sync('What is the capital of Italy?')
print(result_sync.output)