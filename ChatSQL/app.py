# cd .\LangchainProjects\ChatSQL\
# (D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\ChatSQL> streamlit run .\app.py

import streamlit as st
from pathlib import Path
from urllib.parse import quote_plus
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType, create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import sqlite3
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.tools import Tool
from langchain_huggingface import HuggingFaceEmbeddings

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# hugging face embeddings
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

# Set page configuration
st.set_page_config(page_title="Langchain: Chat with SQL DB",page_icon="🤖")
# title of the app
st.title("🤖 Langchain - Chat with Search Engine Gen AI app! Used SQL DB,In-built, custom tools & agents with groq model")

LOCALDB = "USE_LOCALDB"
MYSQL="USE_MYSQL"

# sidebar for settings
st.sidebar.title("Settings")
radio_opt = ["USE SQLLite-3 Database- Student.db","Connect to your SQL Database"]
selected_opt = st.sidebar.radio(label="Choose the DB with which you want to chat",options=radio_opt)

if radio_opt.index(selected_opt) == 1:
    db_uri = MYSQL
    st.sidebar.text("Provide Information:")
    mysql_host = st.sidebar.text_input("My SQL Host")
    mysql_user = st.sidebar.text_input("My SQL User")
    mysql_password = st.sidebar.text_input("My SQL Password",type="password")
    mysql_db = st.sidebar.text_input("My SQL Database")
else:
    db_uri = LOCALDB

# chat groq key input from user
api_key=st.sidebar.text_input(label ="Enter your Groq API key:",type="password")

if not db_uri:
    st.info("Please enter the database information and url")

if not api_key:
    st.info("Please enter the groq api key")

# open source LLM model from groq
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key,streaming=True)

# configuring entire DB
# decorator keeping db details in cache for 2 hours
@st.cache_resource(ttl="2h") 
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        #setup filepath to local db
        dbfilepath = (Path(__file__).parent/"student.db").absolute()
        print(f"Using local SQLite DB: {dbfilepath}")
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro",uri=True)
        return SQLDatabase(create_engine("sqlite:///",creator= creator))
    elif db_uri == MYSQL:
        #if any parameter is messed or not provided display error
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please provide all MYSQL connection details.")
            #raise ValueError("❌ Please provide all MySQL connection details.") #
            return None
        # escape username and password safely for URL
        # quote_plus() ensures special characters like @, #, !, /, : don’t break the connection string. Works for both SQLite local file and MySQL remote DB. Now, if our password is Password@123, it will automatically be turned into Password%40123 in the connection string.
        mysql_user_enc = quote_plus(mysql_user) #encoded username and password
        mysql_password_enc = quote_plus(mysql_password)
        connection_url = f"mysql+mysqlconnector://{mysql_user_enc}:{mysql_password_enc}@{mysql_host}/{mysql_db}"
        print(f"Connecting to MySQL with: {connection_url}")
        return SQLDatabase(create_engine(connection_url))
    else:
        return None
        
if db_uri == MYSQL:
    db = configure_db(db_uri,mysql_host,mysql_user,mysql_password,mysql_db)
else:
    db = configure_db(db_uri)

sqltoolkit = None
sql_tools =[]
INJECTION_WARNING = """
SQL agent can be vulnerable to user_query injection. Use a DB role with limited permissions.
"""

if db is None:
    st.warning("⚠️ Database is not connected. SQL agent will be disabled.")
else:
    # Creating toolkit
    sqltoolkit = SQLDatabaseToolkit(db=db, llm=llm)
    # Injection warning
    st.sidebar.warning(INJECTION_WARNING)

# Get SQL tools (list of Tool objects) - unwraps the toolkit into proper LangChain Tool objects (like query_db, describe_tables).
if sqltoolkit:
    sql_tools = sqltoolkit.get_tools()

#creating SQL agent
#sqlagent = create_sql_agent(
#    llm=llm,
#    toolkit=sqltoolkit,
#    verbose=True,
#    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
#)

# using inbuilt tool of wikipedia
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results = 2, doc_content_chars_max=500) 
wikipedia = WikipediaQueryRun(api_wrapper = api_wrapper_wiki)

# using inbuilt tool of arxiv
api_wrapper_arxiv = ArxivAPIWrapper(top_k_results = 1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper = api_wrapper_arxiv)

#using inbuilt tool of DuckDuckGoSearch for web search
search = DuckDuckGoSearchRun(name='Search') #search → usually a web search tool (like SerpAPI, DuckDuckGoSearchRun, or Tavily).

# RAG Setup
uploaded_files = st.file_uploader("Upload PDFs for Document Q&A", accept_multiple_files=True, type="pdf")
rag_tool = None

if uploaded_files:
    all_docs = []
    for f in uploaded_files:
        temppdf = f"./temp_{f.name}"
        with open(temppdf, "wb") as file:
            file.write(f.getvalue())
        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        all_docs.extend(docs)

    if all_docs:
        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(all_docs)

        # Build embeddings + vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",model_kwargs={"device": "cpu"}) # force CPU
        vectordb = Chroma.from_documents(split_docs, embeddings)
        rag_retriever = vectordb.as_retriever(search_kwargs={"k": 3})

        # Define prompt for RAG
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use the following context to answer the question:\n\n{context}"),
            ("human", "{input}")
        ])

        # Create chain: LLM + retriever
        document_chain = create_stuff_documents_chain(llm, rag_prompt)
        rag_chain = create_retrieval_chain(rag_retriever, document_chain)

        # Wrap as a LangChain Tool for the agent
        rag_tool = Tool(
            name="RAG",
            func=lambda q: rag_chain.invoke({"input": q})["answer"],
            description=(
                "Answer questions **only from the uploaded PDFs**. "
                "Use this if the user asks about documents they uploaded or PDF content."
            )
        )



#it ensures the chat starts with one Assistant greeting
if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [
        {
            "role": "assistant",
            "content": "Hi, I am a chatbot that can search the web, query SQL, and answer PDFs. How can I help you?",
        }
    ]

#to render the messages
#st.chat_message(role) opens a chat container block (for "user" or "assistant").
#Inside that block, we call st.write(...) (or st.markdown, etc.) to display the message.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_query = st.chat_input(placeholder = "Ask anything")

#Displays a text input box at the bottom of the Streamlit app. If the user types something and hits Enter, it returns that string.If the user does nothing, it returns None.
#The := (walrus operator) both assigns the input to user_query and checks if it's not None. 
# := assigns whatever the user typed into the variable user_query.At the same time, it evaluates user_query inside the if.
#Adds the new user message to st.session_state.messages, so the conversation history is preserved across reruns.
#If the user typed something, store it in user_query, save it into the conversation history, and display it in the chat.

if user_query and api_key:
        st.session_state.messages.append({"role":"user","content":user_query})
        st.chat_message("user").write(user_query)

        # our tools
        tools = [wikipedia,arxiv,search] + sql_tools
        if rag_tool:
            tools.append(rag_tool)

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
        agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors= True)

        #with st.chat_message("assistant"): Streamlit provides st.chat_message(role) to show a chat bubble. "assistant" means this block of code will render the assistant’s response in a styled chat bubble. Everything inside the with block gets rendered in that chat bubble.
        #StreamlitCallbackHandler is a LangChain → Streamlit bridge.It lets you see the agent’s step-by-step reasoning (tools invoked, intermediate results, etc.) live in the UI.st.container() defines where to stream those thoughts. expand_new_thoughts=False keeps the intermediate thoughts collapsed by default (so the UI stays clean)
        #We are using StreamlitCallbackHandler to display the thoughts and actions of an agent in interactive streamlit app.
        #agent.run(...) executes the LangChain agent.Input: st.session_state.messages → this is your chat history (so the agent knows the conversation).callbacks=[st_cb] → tells LangChain: “while running, send all intermediate reasoning to the Streamlit UI”.
        # Adds the assistant’s final response to the conversation history.This way, the next time the user asks something, the model still has memory of what was said before.
        # Actually prints the agent’s final answer inside the assistant’s chat bubble. Without this, you’d only see the intermediate reasoning but not the final output.
        # This block runs the agent, shows live reasoning steps in the UI, stores the assistant’s answer in memory, and displays it in the chat bubble

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = agent.run(user_query, callbacks= [st_cb]) #st.session_state.messages
            st.session_state.messages.append({'role':'assistant','content':response})
            st.write(response)



