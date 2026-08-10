import os
import time
import logging
import certifi
import requests
import streamlit as st

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_tavily import TavilySearch
from langsmith import Client as LangSmithClient
from langchain.tools import tool
from langchain_classic.agents import create_react_agent, AgentExecutor


# ==========================================
# APP CONFIG
# ==========================================

st.set_page_config(
    page_title="LangChain Agent",
    page_icon="🤖",
    layout="centered",
)


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

logger.info("Application started")


# ==========================================
# TOOLS
# ==========================================

search_tool = TavilySearch(
    max_results=2
)


@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    logger.info("Weather tool called | city=%s", city)

    start_time = time.time()

    try:
        url = (
            "https://api.weatherstack.com/current"
            f"?access_key={WEATHERSTACK_API_KEY}"
            f"&query={city}"
        )

        logger.info("Sending request to Weatherstack | city=%s", city)

        response = requests.get(
            url,
            timeout=10
        )

        duration = time.time() - start_time

        logger.info(
            "Weatherstack responded | status=%s | duration=%.2fs",
            response.status_code,
            duration,
        )

        data = response.json()

        if "current" not in data:
            logger.warning(
                "Weather data missing | city=%s | response=%s",
                city,
                data,
            )

            return f"Could not fetch weather data for {city}"

        logger.info(
            "Weather tool completed successfully | city=%s",
            city,
        )

        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather: {data['current']['weather_descriptions'][0]}\n"
            f"Humidity: {data['current']['humidity']}%"
        )

    except Exception:
        logger.exception(
            "Weather tool failed | city=%s",
            city,
        )
        raise


tools = [
    search_tool,
    get_weather_data,
]


# ==========================================
# LLM
# ==========================================

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

logger.info("Creating Gemini LLM")

llm = ChatGoogleGenerativeAI(
    model="gemini-flash-lite-latest",
    temperature=0,
    google_api_key=GEMINI_API_KEY,
    max_retries=3,
    rate_limiter=rate_limiter,
)


# ==========================================
# PROMPT
# ==========================================

logger.info("Loading ReAct prompt from LangSmith")

prompt = LangSmithClient().pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True,
)


# ==========================================
# CREATE AGENT
# ==========================================

logger.info("Creating ReAct agent")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# ==========================================
# EXECUTOR
# ==========================================

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5,
    max_execution_time=20,
    early_stopping_method="force",
    handle_parsing_errors=True,
)


# ==========================================
# SESSION STATE
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# UI
# ==========================================

st.title("🤖 LangChain Agent")

st.caption(
    "Ask questions, search the web, or get current weather information."
)

st.divider()


# ==========================================
# SHOW CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

user_input = st.chat_input(
    "Ask me anything..."
)


# ==========================================
# HANDLE USER MESSAGE
# ==========================================

if user_input:

    logger.info(
        "New user message | input=%r",
        user_input,
    )

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # ======================================
    # RUN AGENT
    # ======================================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                logger.info("Agent started")

                start_time = time.time()

                response = agent_executor.invoke(
                    {
                        "input": user_input
                    }
                )

                duration = time.time() - start_time

                logger.info(
                    "Agent completed | duration=%.2fs",
                    duration,
                )

                logger.info(
                    "Agent response received | output=%r",
                    response.get("output"),
                )

                answer = response["output"]

            except Exception as error:

                logger.exception(
                    "Agent failed | error=%s",
                    error,
                )

                answer = (
                    "Something went wrong while running the agent.\n\n"
                    f"`{error}`"
                )

        st.markdown(answer)


    # ======================================
    # SAVE ASSISTANT MESSAGE
    # ======================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )

    logger.info("Assistant message saved to session")