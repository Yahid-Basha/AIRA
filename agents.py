from crewai import Agent, LLM
from tools import yt_tool

from dotenv import load_dotenv

load_dotenv()

import os
if 1:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.7
)


## Create a senior blog content researcher
blog_researcher=Agent(
    role='Blog Researcher from Website',
    goal='get the relevant website data/information for the topic {topic} from the provided Website',
    llm=llm,
    verboe=True,
    memory=True,
    backstory=(
       "Expert in understanding Websited in Automobiles, cars bikes etc and providing suggestion" 
    ),
    max_retry_limit=2,
    tools=[yt_tool],
    respect_context_window=True,
    allow_delegation=True
)

## creating a senior blog writer agent with YT tool

blog_writer=Agent(
    role='Blog Writer',
    goal='Explain the topic {topic} in a simple and engaging manner for drivers and technicicans to understand',
    verbose=True,
    llm=llm,
    memory=True,
    max_retry_limit=2,
    respect_context_window=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    tools=[yt_tool],
    allow_delegation=False


)