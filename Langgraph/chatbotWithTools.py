from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from IPython.display import Image, display

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun

import os
from dotenv import load_dotenv
load_dotenv()

# Set environment variables for LangSmith tracking and LangChain project
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGRAPH_LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# groq api
groq_api_key = os.getenv("GROQ_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant",model_kwargs={"tool_choice": "none"})

## Working With Tools
## Arxiv And Wikipedia tools
arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=300)
arxiv_tool=ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=300)
wiki_tool=WikipediaQueryRun(api_wrapper=api_wrapper)

wiki_tool.invoke("who is Sharukh Khan")
arxiv_tool.invoke("Attention is all you need")
tools=[wiki_tool,arxiv_tool]
llm_with_tools=llm.bind_tools(tools=tools)

## Langgraph Application ###

#  Messages have the type 'list'. The 'add_messages function in the annotation defines how this state key should be updated
#  (int his case, it appends messags to the list,rather than overwriting them).

# State class inheriting TypedDict. # messages variable is of type Annotated
class State(TypedDict): 
  messages:Annotated[list,add_messages] 

#starting graph building process
graph_builder = StateGraph(State)

# chatbot takes parameter as state. Chatbot is inheriting State class because state management keeps on changing.
# chatbot function is invoking previous messages given by user with state and creating new messages which is returned.
def chatbot(state:State):
  return {"messages":[llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot) # my node name
graph_builder.add_edge(START,"chatbot") #start node connected to chatbot

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node) # tool needs to be added to our chatbot
# for bidirectional edges ,chatbot connected to tool node bidirectionally
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition, #prebuilt conditions,default condition's
)

graph_builder.add_edge("tools", "chatbot") #tool node connected to chatbot
graph_builder.add_edge("chatbot",END) # end connected to chatbot
graph = graph_builder.compile()

# to display graph in jupyter notebook or google colab
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

user_input="Hi there!, My name is Kirti"
events=graph.stream(
     {"messages": [("user", user_input)]},stream_mode="values"
)
for event in events:
  event["messages"][-1].pretty_print()

user_input = "what is RLHF."
# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [("user", user_input)]},stream_mode="values"
)
for event in events:
    event["messages"][-1].pretty_print()

while True:
   user_input = input("User: ")
   if user_input.lower() in ["quit","q"]:
      print("Good Bye!")
      break
   for event in graph.stream({'messages':("user",user_input)}):
      print(event.values())
      for value in event.values():
         print(value['messages']) # it should print user message
         print("Assistant:", value['messages'][-1].content) # it should print llm model response