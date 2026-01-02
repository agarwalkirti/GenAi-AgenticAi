#throwing openai quota error

#from crewai_tools import YoutubeChannelSearchTool
# Initialize the tool
#yt_tool = YoutubeChannelSearchTool()

# Initialize the tool with a specific Youtube channel handle to target our search
#yt_tool = YoutubeChannelSearchTool(youtube_channel_handle='@krishnaik06')

#handle = "@krishnaik06"
#channel_url = f"https://www.youtube.com/{handle}"
#yt_tool = YoutubeChannelSearchTool(youtube_channel_handle=channel_url)

# Add multiple YouTube channels # Now yt_tool can fetch/search across all these channels
#yt_tool.add("https://www.youtube.com/@krishnaik06")
#yt_tool.add("https://www.youtube.com/@codebasics")
#yt_tool.add("https://www.youtube.com/@freecodecamp")
#yt_tool.add("https://www.youtube.com/@sentdex")

#Why this works
#YoutubeChannelSearchTool internally stores channel transcripts in a ChromaDB vectorstore.By default, it uses OpenAIEmbeddingFunction.
#Overriding it with SentenceTransformerEmbeddingFunction makes embeddings run locally (no API calls, no quota issues).
#YoutubeChannelSearchTool internally wraps a ChromaDB adapter.By default, ChromaDB adapter → uses OpenAIEmbeddingFunction.
#Even if you pass embedding_function=..., sometimes it gets overridden inside the adapter. Creating a custom Chroma client with SentenceTransformerEmbeddingFunction ensures the tool never calls OpenAI.

from crewai_tools import YoutubeChannelSearchTool
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
import requests
import re
from chromadb import PersistentClient,Settings

# Swap out OpenAI with a local HuggingFace embedding model
#embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

#This monkey-patches ChromaDB’s default so all tools in CrewAI use HuggingFace embeddings instead of OpenAI.
#chromadb.api.models.Collection._embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Create a persistent Chroma client with local embeddings
# Proper way: use Settings class, not a dict
#chroma_client = PersistentClient(
#    path="./chroma_db",
#    settings=Settings(anonymized_telemetry=False)
#)

#chroma_client._embedding_function = embedding_fn  # force override

# Initialize YouTube tool with custom Chroma client. Pass embedding_fn when initializing the tool
#yt_tool = YoutubeChannelSearchTool(chroma_client=chroma_client, embedding_function=embedding_fn)
#from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter

# Create adapter with local embeddings
#adapter = CrewAIRagAdapter(
#    client=chroma_client,
#    embedding_function=embedding_fn
#)

# Inject adapter into the YouTube tool
#yt_tool = YoutubeChannelSearchTool(adapter=adapter)

#print("Embedding function in use:", yt_tool.adapter._client.embedding_function)

#def resolve_channel_url(url_or_handle: str) -> str:
#    """
#    Resolve a YouTube @handle, /c/, /user/, or direct channel link into a canonical /channel/UC... URL.
#    """
#    if url_or_handle.startswith("@"):
#        url_or_handle = f"https://www.youtube.com/{url_or_handle}"

#    response = requests.get(url_or_handle, allow_redirects=True)
#    final_url = response.url

    # 1. Direct channel link already
#    match = re.search(r"(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)", final_url)
#    if match:
#        return match.group(1)

#    # 2. Try extracting from page HTML (canonical link usually contains /channel/UC...)
#    html = response.text
#    match = re.search(r'"canonicalUrl":"(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)"', html)
#    if not match:
#        match = re.search(r'<link rel="canonical" href="(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)"', html)

#    if match:
#        return match.group(1)

#    raise ValueError(f"Could not resolve channel URL from: {url_or_handle}")


# Handles (auto-converted) 
#yt_tool.add(resolve_channel_url("@krishnaik06")) 
#yt_tool.add(resolve_channel_url("@codebasics"))

#PersistentClient(settings=Settings(...))  (instead of passing a raw dict).
#Forced chroma_client._embedding_function = embedding_fn (so even internal calls don’t hit OpenAI).
#Used CrewAIRagAdapter and passed both client + embedding_function.
#Now yt_tool → only uses HuggingFace embeddings locally (no OpenAI quota).
#No OpenAI is ever called – all embeddings are local. Cleaner than monkey-patching _embedding_function.
#You can reuse LocalYoutubeChannelSearchTool everywhere in your CrewAI project. Still fully compatible with yt_tool.add(...), Agent, Task, etc.


from crewai_tools import YoutubeChannelSearchTool
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from chromadb import PersistentClient, Settings
from crewai_tools.adapters.crewai_rag_adapter import CrewAIRagAdapter
import requests
import re

# ------------------------
# Local HuggingFace embedding
# ------------------------
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ------------------------
# Persistent Chroma client
# ------------------------
chroma_client = PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Force override embedding function in the client
chroma_client._embedding_function = embedding_fn

# ------------------------
# Custom Adapter using local embeddings
# ------------------------
class LocalCrewAIRagAdapter(CrewAIRagAdapter):
    def __init__(self, client=chroma_client):
        # Force client + local embeddings
        super().__init__(client=client, embedding_function=embedding_fn)

# ------------------------
# Subclass YoutubeChannelSearchTool to use local adapter
# ------------------------
class LocalYoutubeChannelSearchTool(YoutubeChannelSearchTool):
    def __init__(self):
        adapter = LocalCrewAIRagAdapter()
        super().__init__(adapter=adapter)

# ------------------------
# Instantiate the tool
# ------------------------
yt_tool = LocalYoutubeChannelSearchTool()

# ------------------------
# Utility: Resolve YouTube @handle → canonical /channel/UC... URL
# ------------------------
def resolve_channel_url(url_or_handle: str) -> str:
    if url_or_handle.startswith("@"):
        url_or_handle = f"https://www.youtube.com/{url_or_handle}"

    response = requests.get(url_or_handle, allow_redirects=True)
    final_url = response.url

    # Direct channel link already
    match = re.search(r"(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)", final_url)
    if match:
        return match.group(1)

    # Extract from HTML canonical tag
    html = response.text
    match = re.search(r'"canonicalUrl":"(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)"', html)
    if not match:
        match = re.search(r'<link rel="canonical" href="(https://www\.youtube\.com/channel/[A-Za-z0-9_-]+)"', html)

    if match:
        return match.group(1)

    raise ValueError(f"Could not resolve channel URL from: {url_or_handle}")

# ------------------------
# Add YouTube channels
# ------------------------
yt_tool.add(resolve_channel_url("@krishnaik06"))
yt_tool.add(resolve_channel_url("@codebasics"))
yt_tool.add(resolve_channel_url("@freecodecamp"))
yt_tool.add(resolve_channel_url("@sentdex"))
