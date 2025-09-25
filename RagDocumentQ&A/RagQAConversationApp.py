#cd '.\LangchainProjects\RagDocumentQ&A\'
#(D:\study\udemy\langchain\venv) PS D:\study\udemy\langchain\LangchainProjects\RagDocumentQ&A> streamlit run RagQAConversationApp.py

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

#open source model from groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
#api_key = os.getenv("GROQ_API_KEY")
#llm = ChatGroq(model="gemma2-9b-it", groq_api_key=api_key) 

#hugging face embeddings technique
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#set up streamlit app
#title of the app
st.title("Conversational Rag Q&A chatbot with PDF uploads and chat history using groq model and hugging face embedding")
st.write("Upload Pdf's and chat with their content as context")

#input groq api key
api_key=st.text_input("Enter your Groq API key:",type="password")

#check if groq api key is provided
if api_key:
    llm = ChatGroq(model_name="gemma2-9b-it", groq_api_key=api_key) 

    #chat interface

    # statefully manage chat history
    session_id=st.text_input("Session ID",value="default_session")

    # save everything in session so my entire session have memory
    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A Pdf file",type="pdf",accept_multiple_files=True)

    # Processing of uploaded PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            #wb -  write byte mode
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
        
            loader = PyPDFLoader(temppdf) # data injetion
            docs = loader.load() # document loading
            documents.extend(docs)
        #split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500) # text splitting
        splits = text_splitter.split_documents(documents)
        #convert document into vectors and save in Chroma vector db
        vectorstore = Chroma.from_documents(documents = splits ,embedding = embeddings)
        retriever = vectorstore.as_retriever()

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

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
            )


        #taking user query
        user_input = st.text_input("Enter your question from the uploaded paper/paper's:")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{"session_id":session_id}
                },#constructs  a key "abc123" in store
            )
            #st.write(st.session_state.store)
            st.write('----------------------------------')
            st.success(f"Assistant: {response['answer']}")
            st.write('----------------------------------')
            #st.write("Chat History:",session_history.messages)
else:
    st.warning("Please enter groq api key to proceed")






