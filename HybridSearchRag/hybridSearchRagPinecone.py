#does both keyword search and vector search i.e in sparse and dense vectors.
from langchain_community.retrievers import PineconeHybridSearchRetriever 
from pinecone import Pinecone,ServerlessSpec
from pinecone_text.sparse import BM25Encoder

from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
os.environ["PINECONE_API_KEY"] =  os.getenv("PINECONE_API_KEY")

index_name = "hybrid-search-langchain-pinecone"

# initialize pinecone client
pc = Pinecone(api_key = os.getenv("PINECONE_API_KEY"))
print("Connected to Pinecone successfully!")

# Create the index if it doesn’t exist
if index_name not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name = index_name,
        dimension = 384, # embedding dimension of dense vector, hugging face embedding technique creates 384 dimension vectors
        metric = 'dotproduct', # sparse values supported only for dotproduct
        spec = ServerlessSpec(cloud ='aws' ,region ='us-east-1')
    )
    print("Pinecone index ready:", index_name)

#  Connect to index
index = pc.Index(index_name)
print("Pinecone index:", index)

## vector embedding and sparse matrix
# Define embeddings (dense)
embeddings = HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2") # for dense vectors

bm25_encoder = BM25Encoder().default() # for sparse matrix
print("bm25 encoder: ",bm25_encoder)

# add sample sentences
sentences = [
    "LangChain integrates with Pinecone for vector search.",
    "Hybrid search combines keyword and semantic similarity.",
    "Pinecone is a managed vector database service."
]

# tfidf values on these sentences
bm25_encoder.fit(sentences)

# store the values injson file
bm25_encoder.dump("bm25_values.json")

# load to our bm25 encoder object,load anywhere in project where needed.
# Only use them later when you want to reuse the trained BM25 model across sessions
bm25_encoder = BM25Encoder().load("bm25_values.json")

#Add sample documents
docs = [
    {"id": "1", "text": "LangChain integrates with Pinecone for vector search."},
    {"id": "2", "text": "Hybrid search combines keyword and semantic similarity."},
    {"id": "3", "text": "Pinecone is a managed vector database service."}
]

# Create hybrid retriever (dense + sparse)
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25_encoder,  # keyword-based sparse encoder
    index=index
)

# adding all my sentences
retriever.add_texts(
    ["LangChain integrates with Pinecone for vector search.",
    "Hybrid search combines keyword and semantic similarity.",
    "Pinecone is a managed vector database service."]
    )

# invoke retriever for results
result = retriever.invoke("What is pinecone?")

# Display result of sentences
print("Sentences result:")
for i, doc in enumerate(result):
    print(f"\nResult {i+1}: {doc.page_content}")

# adding all my docs in retriever
retriever.add_texts([d["text"] for d in docs], ids=[d["id"] for d in docs])

# Query
query = "What is hybrid search?"
results = retriever.invoke(query)

# Display results of docs
print("Docs result:")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}: {doc.page_content}")


