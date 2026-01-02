from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode,tools_condition
from IPython.display import Image, display

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
  messages:Annotated[list,add_messages] 

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

while True:
   user_input = input("User: ")
   if user_input.lower() in ["quit","q"]:
      print("Good Bye!")
      break
   for event in graph.stream({'messages':("user",user_input)}):
      print(event.values())
      for value in event.values():
         print(value['messages']) # it should user message
         print("Assistant:", value['messages'][-1].content) # it should have llm model response

## for seeing graph

#Try online Mermaid rendering first.If that fails, fallback to local Pyppeteer rendering.
#Save the .png file and display it (if in Jupyter/IPython).
#Always save the .mmd Mermaid source for manual viewing.

from IPython.display import Image, display

def render_langgraph(graph, filename="langgraph.png"):
    """
    Render a LangGraph diagram with online -> Pyppeteer fallback,
    and save both PNG and Mermaid source.
    """
    # Save Mermaid source always
    try:
        mermaid_code = graph.get_graph().draw_mermaid()
        with open("langgraph.mmd", "w") as f:
            f.write(mermaid_code)
        print("Mermaid source saved to langgraph.mmd (view at https://mermaid.live)")
    except Exception as e:
        print("Could not save Mermaid source:", e)

    # Attempt online rendering first
    try:
        print("Attempting online Mermaid rendering...")
        png = graph.get_graph().draw_mermaid_png(max_retries=3, retry_delay=1.5)
        print("Online rendering successful!")
    except Exception as e1:
        print("Online rendering failed:", e1)
        print("Falling back to local Pyppeteer rendering...")
        try:
            png = graph.get_graph().draw_mermaid_png(draw_method="pyppeteer")
            print("Local Pyppeteer rendering successful!")
        except Exception as e2:
            print("Both rendering methods failed:", e2)
            return

    # Save PNG
    with open(filename, "wb") as f:
        f.write(png)
    print(f"🖼️ Graph saved as {filename}")

    # Display if in notebook / IPython
    try:
        display(Image(png))
    except Exception:
        pass

render_langgraph(graph)

