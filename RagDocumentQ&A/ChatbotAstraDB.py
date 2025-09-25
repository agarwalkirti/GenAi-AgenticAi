#aquerying pdf with astra db -cassandra db used powered by vector search &langchain
#cd '.\LangchainProjects\RagDocumentQ&A\'
#(D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\RagDocumentQ&A> streamlit run ChatbotAstraDB.py
# the pipeline is: our PDF → embeddings → Astra DB → retriever → Groq LLM. 
# The LLM produces a concise answer, and we even see similarity scores in similarity search call

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PDFMinerLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.vectorstores import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

#with cassio, the engine powering the astra db integration in langchain
# ,we will also initialize the db connection
import cassio

# support for dataset retrieval with hugging face
from datasets import load_dataset

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# open source model from groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key = os.getenv("GROQ_API_KEY")

# hugging face embeddings technique
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

# Initialization of db env variables
os.environ["ASTRA_DB_API_ENDPOINT"] = os.getenv("ASTRA_DB_API_ENDPOINT")
os.environ["ASTRA_DB_APPLICATION_TOKEN"] = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
os.environ["ASTRA_DB_ID"] = os.getenv("ASTRA_DB_ID")

db = cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
#print(f"Connected to Astra Vector: {db.get_collections()}")

loader = PDFMinerLoader('research_papers/LLM.pdf') # data injetion
documents = loader.load() # document loading

#split and create embeddings for the documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500,length_function = len,separators="\n") # text splitting
texts_splits = text_splitter.split_documents(documents)

llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key) 
#convert document into vectors
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
    )

# creation of langchain vector store backed by astra db
astra_vector_store_db = Cassandra(
  embedding = embeddings,
  table_name="qa_mini_demo",
  session = None,
  keyspace=None # default_keyspace
)

# load the dataset into the vector store db
# astra_vector_store_db.add_texts(texts_splits) #texts_splits are documents
# Insert documents is the embedding + ingestion step — every time we run the app it will re-embed and push into Astra DB
astra_vector_store_db.add_documents(texts_splits)  
astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store_db)

retriever = astra_vector_store_db.as_retriever()

#set up streamlit app
#title of the app
st.title("Astra DB Conversational RAG with LangChain + Streamlit using groq model and hugging face embedding")
st.write("Q&A chatbot with pdf's content as context")

# statefully manage chat history
session_id=st.text_input("Session ID",value="default_session")

# Session Memory # save everything in session so my entire session have memory
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if "store" not in st.session_state:
        st.session_state.store = {}
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

#taking user query
user_input = st.text_input("Enter your question from the paper:")
user_input_answer = astra_vector_index.query(user_input,llm=llm).strip() # direct query from astra db without calling rag
print("\n answer \n",user_input_answer)
# based on similarity search top 4 relevant answers
for doc, score in astra_vector_store_db.similarity_search_with_score(user_input, k=4):
    print(f"[{score:.4f}] \"{doc.page_content[:84]}...\"")

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question"
    "which might reference context in the chat history"
    "formulate a standalone question which can be understood"
    "without the chat history. Do NOT answer the question,"
    "just reformulate it if needed and otherwise return it as is."
    "If the answer cannot be found in the context, say 'I don’t know'."
    )

#all my history of chat will be stored in variable chat_history given in MessagePlaceholder
# Contextualization prompt (for question rewriting only)
contextualize_qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

#Answer question
system_prompt = (
        "You are an assistant for question answering tasks." \
        "Use the following pieces of retrieved context to answer" \
        "the question. If you dont know the answer, say that you don't know." \
        "Use three sentences maximum and keep the answer concise"
        "\n\n"
        "{context}"
    )

# QA prompt (must include {context} for doc injection by stuffed document chain!)
qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

# History-aware retriever
history_aware_retriever = create_history_aware_retriever(llm,retriever,contextualize_qa_prompt)

# Question Answer chain
question_answer_chain = create_stuff_documents_chain(llm,qa_prompt)

# Final RAG chain
rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
        
conversational_rag_chain=RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer"
    )

if user_input:
    session_history = get_session_history(session_id)
    response = conversational_rag_chain.invoke(
        {"input":user_input},
        config={
            "configurable":{"session_id":session_id}
        },#construct a key say "abc123" in store
    )
    
    st.write('----------------------------------')
    st.success(f"Assistant: {response['answer']}")
    st.write('----------------------------------')







