# data injetion-pdfs+ aws bedrock -> deepseek model+langchain +faiss db-> store embeddings in vector store
# question -> similarity search ->vector store->relevant chunks->prompt->llm call->answer
import json
import os
import sys
import boto3

# using titan embedding model to generate embeddings. we will call from langchain framework.

from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms.bedrock import Bedrock

## data injestion
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

#vector embedding and vector store
from langchain_community.vectorstores import FAISS

## LLM model
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import streamlit as st

## bedrock clients
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

prompt_template = """
Human: Use the following pieces of context to provide a 
concise answer to the question at the end but use at least summarize with 
250 words with detailed explanations. If you don't know the answer, 
just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

## data injetion
def data_ingestion():
    loader = PyPDFDirectoryLoader("pdfs")
    documents = loader.load()

    #character text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000,
                                                   chunk_overlap=1000)
    
    docs = text_splitter.split_documents(documents)
    return docs

# vector embedding and vector store

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index") #can be stored in any db

#check deepseek
def get_deepseek_llm():
    #Here is the JSON request body format for calling DeepSeek-R1 via Amazon Bedrock’s InvokeModel API (text completion) based on AWS docs: 
    #AWS Documentation
    #{
        #"prompt": "Your text prompt here",
        #"temperature": 0.7,
        #"top_p": 0.9,
        #"max_tokens": 512,
        #"stop": ["\n", "Human:", "Assistant:"]  // optional array of stop sequences
    #}
    # Use the Inference Profile ID or ARN (check AWS Bedrock console > Inference profiles)
    model_id = "us.deepseek.r1-v1:0"#"arn:aws:bedrock:us-east-1:123456789012:inference-profile/deepseek-r1-v1:0"

    llm = Bedrock(
        model_id=model_id,
        #model_provider="deepseek",   # REQUIRED when using ARN
        client=bedrock,
        inferenceConfig={
            #"inputText": " ",    
            "maxTokens": 512,
            "temperature": 0.7 
            # "top_p": 0.9 #optional, add more params if needed
        }
    ) #wrapper provided by langchain to invoke models present in aws amazon bedrock
    return llm

#llama 3 working
def get_llama_llm():
    model_id = "meta.llama3-8b-instruct-v1:0"#"meta.llama2-70b-chat-v1" #"us.meta.llama3-2-3b-instruct-v1:0"
    llm = Bedrock(model_id=model_id,
                  client = bedrock,
                  model_kwargs={
                                "prompt": " ",   # Bedrock expects this field,but langchain automatically fills this with prompt
                                "max_gen_len": 512,
                                "temperature": 0.7,
                                "top_p": 0.9
                                }
                  ) #wrapper provided by langchain to invoke models present in aws amazon bedrock
    return llm

def get_response_llm(llm,vectorstore_faiss,query):
    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore_faiss.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":query})
    return answer['result']


def main():
    st.set_page_config("Chat PDF")
    
    st.header("Chat with PDF using AWS Bedrock💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("Done")

    if st.button("Deepseek Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_deepseek_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

    if st.button("Llama Output"):
        with st.spinner("Processing..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings,allow_dangerous_deserialization=True)
            llm=get_llama_llm()
            
            #faiss_index = get_vector_store(docs)
            st.write(get_response_llm(llm,faiss_index,user_question))
            st.success("Done")

if __name__ == "__main__":
    main()



