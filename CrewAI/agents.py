from crewai import Agent
from tools import yt_tool

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

#open source model from groq
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
#os.environ["GROQ_MODEL_NAME"] = "llama-3.1-8b-instant"

# open source LLM model from groq
api_key = os.getenv("GROQ_API_KEY")
groq_llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

## Creating a senior blog content researcher
blog_researcher=Agent(
    role='Blog Researcher from Youtube Videos',
    goal='Get the relevant video transcription for the topic {topic} from the provided Youtube channel',
    verbose=True,
    memory=True,
    backstory=(
       "Expert in understanding videos in AI, Data Science , Machine Learning , GEN AI and Agentic AI And providing suggestion" 
    ),
    llm = groq_llm,
    tools=[yt_tool],
    allow_delegation=True
)

## creating a senior blog writer agent with YT tool
blog_writer=Agent(
    role='Blog Writer',
    goal='Narrate compelling tech stories about the videos {topic} from YouTube Channel',
    verbose=True,
    memory=True,
    backstory=(
        "With a flair for simplifying complex topics, you craft"
        "engaging narratives that captivate and educate, bringing new"
        "discoveries to light in an accessible manner."
    ),
    llm = groq_llm,
    tools=[yt_tool],
    allow_delegation=False
)