# %%
import os
from re import search
from unittest import result

import certifi
import dotenv


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tavily import TavilySearch
from langsmith import Client as LangSmithClient
from langchain.tools import tool
import requests



# %%
from langchain_classic.agents import create_react_agent, AgentExecutor

# %%
from dotenv import load_dotenv

# ==========================================
# LOAD ENV VARIABLES
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
# %%
search_tool = TavilySearch(max_results=2)
# %%
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )
# %%
print(get_weather_data.invoke("hadera"))
# %%
result = search_tool.invoke("Give me the latest news on AI")
result
# %%
# ==========================================
# LLM
# ==========================================

# gemini-flash-lite-latest has free-tier quota available on this key
# (gemini-3.5-flash caps at 20 req/day; gemini-2.0-flash-lite/flash
# had zero free quota on this key/region). A ReAct agent burns one
# request per Thought/Action step, so pace requests client-side too.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,  # one request every 2s, stays clear of per-minute caps
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    max_retries=3,
    rate_limiter=rate_limiter,
)
print(llm)
# %%
response = llm.invoke("Tell me a jock about AI")
response.content
# %%
# ==========================================
# PROMPT
# ==========================================

prompt = LangSmithClient().pull_prompt(
    "hwchase17/react", dangerously_pull_public_prompt=True
)
# %%
prompt
# %%
#======
# TOOLS
#======
tools = [search_tool, get_weather_data]
# %%
#=======
# CREATE AGENT
# ======

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# %%
#=====
# EXECUTOR
#======
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    max_execution_time=20,
    early_stopping_method="force",
    handle_parsing_errors=True,
)
# %%
#========
# RUN
#========

response = agent_executor.invoke({
        "input": (
            "Which is better for tourists, Rome or Paris?"
        )
    })
# %%
print(response["output"])