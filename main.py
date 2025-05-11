from openai import AsyncOpenAI
import os
import chainlit as cl
import httpx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from dataclasses import dataclass

@dataclass
class Deps:
    client: httpx.AsyncClient
    weather_api_key: str | None
    geo_api_key: str | None

# Initialize the OpenAI client for Chainlit
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    http_client=httpx.AsyncClient(verify=False)
)

# Initialize the model for weather agent
model = OpenAIModel(
    'google/gemini-2.0-flash-lite-001',
    provider=OpenAIProvider(
        base_url='https://openrouter.ai/api/v1',
        api_key=os.getenv("OPENROUTER_API_KEY"),
        http_client=httpx.AsyncClient(verify=False)
    ),
)

# Create the weather agent
weather_agent = Agent(
    model=model,
    system_prompt=(
        'Be concise, reply with one sentence. '
        'Use the `get_lat_lng` tool to get the latitude and longitude of the locations, '
        'then use the `get_weather` tool to get the weather.'
    ),
    deps_type=Deps,
    retries=2,
    instrument=True,
)

# Instrument the OpenAI client for Chainlit
cl.instrument_openai()

settings = {
    "model": "google/gemini-2.0-flash-lite-001",
    "temperature": 0,
}

@weather_agent.tool
async def get_lat_lng(
    ctx: RunContext[Deps], location_description: str
) -> dict[str, float]:
    """Get the latitude and longitude of a location."""
    if ctx.deps.geo_api_key is None:
        return {'lat': 51.1, 'lng': -0.1}

    params = {
        'q': location_description,
        'api_key': ctx.deps.geo_api_key,
    }
    r = await ctx.deps.client.get('https://geocode.maps.co/search', params=params)
    r.raise_for_status()
    data = r.json()

    if data:
        return {'lat': data[0]['lat'], 'lng': data[0]['lon']}
    else:
        raise ModelRetry('Could not find the location')

@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], lat: float, lng: float) -> dict[str, Any]:
    """Get the weather at a location."""
    if ctx.deps.weather_api_key is None:
        return {'temperature': '21 °C', 'description': 'Sunny'}

    params = {
        'apikey': ctx.deps.weather_api_key,
        'location': f'{lat},{lng}',
        'units': 'metric',
    }
    r = await ctx.deps.client.get(
        'https://api.tomorrow.io/v4/weather/realtime', params=params
    )
    r.raise_for_status()
    data = r.json()

    values = data['data']['values']
    code_lookup = {
        1000: 'Clear, Sunny',
        1100: 'Mostly Clear',
        1101: 'Partly Cloudy',
        1102: 'Mostly Cloudy',
        1001: 'Cloudy',
        2000: 'Fog',
        2100: 'Light Fog',
        4000: 'Drizzle',
        4001: 'Rain',
        4200: 'Light Rain',
        4201: 'Heavy Rain',
        5000: 'Snow',
        5001: 'Flurries',
        5100: 'Light Snow',
        5101: 'Heavy Snow',
        6000: 'Freezing Drizzle',
        6001: 'Freezing Rain',
        6200: 'Light Freezing Rain',
        6201: 'Heavy Freezing Rain',
        7000: 'Ice Pellets',
        7101: 'Heavy Ice Pellets',
        7102: 'Light Ice Pellets',
        8000: 'Thunderstorm',
    }
    return {
        'temperature': f'{values["temperatureApparent"]:0.0f}°C',
        'description': code_lookup.get(values['weatherCode'], 'Unknown'),
    }

@cl.on_message
async def on_message(message: cl.Message):
    # Check if the message is about weather
    if any(keyword in message.content.lower() for keyword in ['weather', 'temperature', 'forecast']):
        # Initialize dependencies for weather agent
        deps = Deps(
            client=httpx.AsyncClient(verify=False),
            weather_api_key=os.getenv('WEATHER_API_KEY'),
            geo_api_key=os.getenv('GEO_API_KEY')
        )
        
        # Get weather information
        result = await weather_agent.run(message.content, deps=deps)
        await cl.Message(content=result.output).send()
    else:
        # Use the original Chainlit chat functionality
        response = await client.chat.completions.create(
            messages=[
                {
                    "content": "You are a helpful bot, you always reply in Traditional Chinese",
                    "role": "system"
                },
                {
                    "content": message.content,
                    "role": "user"
                }
            ],
            **settings
        )
        await cl.Message(content=response.choices[0].message.content).send()