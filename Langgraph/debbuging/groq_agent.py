from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages,BaseMessage
from langgraph.graph import StateGraph,START,END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGRAPH_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# groq api
groq_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

## Langgraph Application ###

#  Messages have the type 'list'. The 'add_messages function in the annotation defines how this state key should be updated
#  (int his case, it appends messags to the list,rather than overwriting them).

# State class inheriting TypedDict. # messages variable is of type Annotated
class State(TypedDict): 
  messages:Annotated[list[BaseMessage],add_messages] 

def make_default_graph():
    #starting graph building process
    graph_builder = StateGraph(State)

    # chatbot takes parameter as state. Chatbot is inheriting State class because state management keeps on changing.
    # chatbot function is invoking previous messages given by user with state and creating new messages which is returned.
    def chatbot(state:State):
        return {"messages":[llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot",chatbot)
    graph_builder.add_edge(START,"chatbot")
    graph_builder.add_edge("chatbot",END)

    graph = graph_builder.compile()
    return graph

agent = make_default_graph()


