#cd '.\LangchainProjects\1-Q&AChatbot\'
#(D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\1-Q&AChatbot> streamlit run main.py

import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#open source model from groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key = os.getenv("GROQ_API_KEY")

##prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

#temperature-model creativity setting between 0 to 1
def generate_response(question,llm,temperature,max_tokens):
    llm = ChatGroq(model=llm, groq_api_key=api_key)
    #llm = OllamaLLM(model="gemma:2b")
    output_parser=StrOutputParser()
    chain = prompt | llm | output_parser 
    answer = chain.invoke({'question':question})
    return answer

#title of the app
st.title("Enhanced Q&A chatbot with groq")

#sidebar for settings
st.sidebar.title("Settings")
#api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")

##check if groq api key is provided
#if api_key:
#    llm = ChatGroq(model_name='gemma2-9b-it', groq_api_key=api_key)

#dropdown to select various groq models
llm=st.sidebar.selectbox("Select a Groq Model",['deepseek-r1-distill-llama-70b','llama-3.1-8b-instant','gemma2-9b-it','qwen/qwen3-32b',"groq/compound-mini"])

#adjust response parameter
temperature = st.sidebar.slider("Temperature",min_value=0.0,max_value=1.0,value=0.7) 
max_tokens = st.sidebar.slider("Max Tokens",min_value=50,max_value=300,value=150) 

## Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

if user_input:
    response=generate_response(user_input,llm,temperature,max_tokens)
    st.write(response)
else:
    st.write("Please provide the query")


