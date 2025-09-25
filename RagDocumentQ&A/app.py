#cd '.\LangchainProjects\RagDocumentQ&A\'
#(D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\RagDocumentQ&A> streamlit run app.py

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
#from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import time

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#open source model from groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key = os.getenv("GROQ_API_KEY") #st.secrets["GROQ_API_KEY"]
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key) #'deepseek-r1-distill-llama-70b','llama-3.1-8b-instant','gemma2-9b-it','qwen/qwen3-32b',"groq/compound-mini"

#hugging face embeddings
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")

##prompt template
prompt = ChatPromptTemplate.from_template(
        """ 
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        <context>
        {context}
        <context>
        Question:{input}
        """
)

def create_vector_embedding():
    #save everything in session so my entire session have memory
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #embedding technique
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # data injetion
        st.session_state.docs = st.session_state.loader.load() # document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) # text splitting
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)#convert document into vectors ans save in faiss vector db


#title of the app
st.title("Rag Document Q&A chatbot with groq model and hugging face embedding")

#taking user query
user_prompt = st.text_input("Enter your query from the research paper")

# button for Generating Document Embeddings
if st.button("Generate Document Embeddings"):
    create_vector_embedding()
    st.write("FAISS Vector Database is Ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt) 
    #Creating a chain for passing a list of Documents to our model. Inside prompt's {context} entire list of documents will be passed.
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_prompt})
    print(f"Response time: {time.process_time()-start}")

    st.write(response['answer'])

    ##with streamlit expander
    with st.expander("Document's similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('----------------------------------')





