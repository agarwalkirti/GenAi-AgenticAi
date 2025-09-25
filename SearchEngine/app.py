# cd .\LangchainProjects\SearchEngine\
# (D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\SearchEngine> streamlit run .\app.py
from langchain_groq import ChatGroq
import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler


import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
st.secrets["LANGCHAIN_API_KEY"]
st.secrets["LANGCHAIN_PROJECT"]
st.secrets["LANGCHAIN_TRACING_V2"]

# using inbuilt tool of wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 2, doc_content_chars_max=500) 
wikipedia = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

# using inbuilt tool of arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name='Search') #search → usually a web search tool (like SerpAPI, DuckDuckGoSearchRun, or Tavily).

#title of the app
st.title("Langchain - Chat with Search Engine Gen AI app! Used tools and agents with groq model")

#sidebar for settings
st.sidebar.title("Settings")
#chat groq key input from user
api_key=st.sidebar.text_input("Enter your Groq API key:",type="password")

#it ensures the chat starts with one Assistant greeting
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I am a chatbot who can search the web. How can I help you?",
        }
    ]

#to render the messages
#st.chat_message(role) opens a chat container block (for "user" or "assistant").
#Inside that block, we call st.write(...) (or st.markdown, etc.) to display the message.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

#Displays a text input box at the bottom of the Streamlit app. If the user types something and hits Enter, it returns that string.If the user does nothing, it returns None.
#The := (walrus operator) both assigns the input to prompt and checks if it's not None. 
# := assigns whatever the user typed into the variable prompt.At the same time, it evaluates prompt inside the if.
#Adds the new user message to st.session_state.messages, so the conversation history is preserved across reruns.
#If the user typed something, store it in prompt, save it into the conversation history, and display it in the chat.

if api_key:
    if prompt := st.chat_input(placeholder = "What is machine learning?"):
        st.session_state.messages.append({"role":"user","content":prompt})
        st.chat_message("user").write(prompt)

        # open source model from groq
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key,streaming=True)

        # our tools
        tools = [wikipedia,arxiv,search]

        # AgentType.ZERO_SHOT_REACT_DESCRIPTION
        # "Zero-shot" → means the agent is not fine-tuned with examples for each tool. Instead, it relies only on the tool descriptions we provide.
        # "ReAct" → Reasoning + Acting framework. The agent alternates between: Thought: reasoning about the question,Action: calling a tool,Observation: receiving tool output.
        # "Description" → the agent chooses which tool to used based on the tool’s description.
        # CHAT_ZERO_SHOT_REACT_DESCRIPTION - Default chat-based ReAct agent.Uses plain natural language to decide which tool to call and what arguments to pass.
        # Tool inputs are given as raw text strings.
        # Example reasoning: Thought: The user wants to know about LangSmith. I should use the langsmith-search tool. Action: langsmith-search Action Input: "LangSmith overview"
        # Simpler, works well if tools only expect a single string input. Can break if a tool requires structured or multiple inputs.
        # STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION: Same ReAct style, but tool inputs are validated with JSON schemas.Each tool defines its expected arguments (via Pydantic schema or LangChain’s structured tool system).
        # The LLM is forced to output structured arguments that match the schema. 
        # Example reasoning: Thought: The user wants to search LangSmith docs. Action: langsmith-search Action Input: { "query": "LangSmith overview", "top_k": 5 }
        # Much more reliable when tools require multiple arguments or complex input. Ensures correct typing and fewer errors. Slightly more overhead when defining tools (you must specify structured arguments).
        # With handle_parsing_errors=True, it won’t crash — it will fallback to treating the malformed output as a direct answer. 
        # LangChain will catch the parsing error and try to gracefully recover
        
        # convert entire tools into agents
        search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors= True)

        #with st.chat_message("assistant"): Streamlit provides st.chat_message(role) to show a chat bubble. "assistant" means this block of code will render the assistant’s response in a styled chat bubble. Everything inside the with block gets rendered in that chat bubble.
        #StreamlitCallbackHandler is a LangChain → Streamlit bridge.It lets you see the agent’s step-by-step reasoning (tools invoked, intermediate results, etc.) live in the UI.st.container() defines where to stream those thoughts. expand_new_thoughts=False keeps the intermediate thoughts collapsed by default (so the UI stays clean)
        #We are using StreamlitCallbackHandler to display the thoughts and actions of an agent in interactive streamlit app.
        #search_agent.run(...) executes the LangChain agent.Input: st.session_state.messages → this is your chat history (so the agent knows the conversation).callbacks=[st_cb] → tells LangChain: “while running, send all intermediate reasoning to the Streamlit UI”.
        # Adds the assistant’s final response to the conversation history.This way, the next time the user asks something, the model still has memory of what was said before.
        # Actually prints the agent’s final answer inside the assistant’s chat bubble. Without this, you’d only see the intermediate reasoning but not the final output.
        # This block runs the agent, shows live reasoning steps in the UI, stores the assistant’s answer in memory, and displays it in the chat bubble

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = search_agent.run(st.session_state.messages, callbacks= [st_cb])
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write(response)





