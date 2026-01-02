# build a question answering application over a graph database
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.graphs import Neo4jGraph
#from langchain_neo4j import Neo4jGraph
from langchain.chains import GraphCypherQAChain
#from langchain_neo4j.chains import GraphCypherQAChain

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("SIMPLE_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# hugging face embeddings
os.environ["HUGGING_FACE_API_KEY"] = os.getenv("HUGGING_FACE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name ="all-MiniLM-L6-v2") 

#open source model from groq
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
api_key = os.getenv("GROQ_API_KEY")
# open source model from groq
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)

neo4j_uri = os.getenv("NEO4J_URI")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

graph = Neo4jGraph(url=neo4j_uri,username=neo4j_username,password=neo4j_password)
print("Connected to Neo4j graph db successfully!",graph)

## dataset- movies review
# actors,director,genres-multiple values
# id,released,title,imdb rating- single values
movie_query=""" 
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/refs/heads/main/movies/movies_small.csv' AS row

MERGE (m:Movie {id: toInteger(row.movieId)})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating = toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors, '|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))

"""
graph.query(movie_query)

graph.refresh_schema()
print("Graph Schema:",graph.schema)

chain = GraphCypherQAChain.from_llm(graph=graph,llm=llm,verbose=True,allow_dangerous_requests=True) #exclude_types=["Genre"]
print("chain :",chain)
print("\nchain graph schema:",chain.graph_schema)
print("\nchain schema:",chain.schema)

#question,schema to be provided in default prompt of GraphCypherQAChain
response = chain.invoke(
        {
            "query":"Who was the director of the movie Casino",
            "schema":graph.schema
        }
    )
print("Response: ",response)

response2 = chain.invoke(
        {
            "query":"Who acted in movie Casino",
            "schema":graph.schema
        }
    )
print("Response2: ",response2)

#prompting strategies to improve results

examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {{title: 'Casino'}})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {{title: 'Schindler\\'s List'}})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

from langchain_core.prompts import FewShotPromptTemplate,PromptTemplate

example_prompt=PromptTemplate.from_template(
    "User input:{question}\n Cypher query:{query}"
)

prompt=FewShotPromptTemplate(
    examples=examples[:5],
    example_prompt=example_prompt,
    prefix="You are a Neo4j expert. Given an input question,create a syntactically very accurate Cypher query",
    suffix="User input: {question}\nCypher query: ",
    input_variables=["question","schema"]
)
print("Prompt: ",prompt)
print("Prompt format: ",prompt.format(question="How many artists are there?", schema = graph.schema))
prompt_chain=GraphCypherQAChain.from_llm(graph=graph,llm=llm,cypher_prompt=prompt,verbose=True)
response1 = prompt_chain.invoke("How many movies Tom Hank acted in?")
print("Prompt + chain Response1: ",response1)
response = prompt_chain.invoke("In which movies Tom Hank acted in?")
print("Prompt + chain Response1: ",response)
response2 = prompt_chain.invoke("actors who acted in multiple movies")
print("Prompt + chain Response2: ",response2)










