import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


## Prompt template
prompt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a helpful AI assistant that helps people find information. Please respond to the question asked"),
    ("user", "Question:{question}")
    ])

#streamlit framework -Streamlit is a powerful Python library for building interactive web applications. 
# It allows us to build graphical user interfaces (GUIs) that allow users to explore and interact with our application
st.title("Langchain and Streamlit demo with ollama-gemma:2b model")
input_text = st.text_input("Enter your question here")

#ollama model
llm = OllamaLLM(model="gemma:2b")  # or llama2, mistral, etc.
output_parser = StrOutputParser()
chain=prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))
       





