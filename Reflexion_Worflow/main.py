from dotenv import load_dotenv
from langchain_core.messages import BaseMessage,ToolMessage,HumanMessage
from langgraph.graph import END, MessageGraph
from chains import revisor, first_responder
from tool_executor import excute_tools
from typing import List
from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod


load_dotenv()

MAX_ITERATIONS = 2
graph = MessageGraph() 
graph.add_node("draft",first_responder)
graph.add_node("execute_tools",excute_tools)
graph.add_node("revise",revisor)
graph.add_edge("draft","execute_tools")
graph.add_edge("execute_tools","revise")


def event_loop(state:List[BaseMessage])->str:
    count_tool_visits  = sum(isinstance(item,ToolMessage) for item in state)
    num_iteration = count_tool_visits
    if num_iteration > MAX_ITERATIONS:
        return END 
    return "execute_tools"

graph.add_conditional_edges(
    "revise",
    event_loop,
    {
        "execute_tools": "execute_tools",
        END: END
    }
)
graph.set_entry_point("draft")
builder = graph.compile()

png_bytes = builder.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

with open("graph_flow.png","wb") as f:
    f.write(png_bytes)
    

if __name__ == "__main__":
    print("hello reflexion agent")
    res = builder.invoke(
        HumanMessage(content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital."))
    print(res[-1].tool_calls[0]["args"]["answer"])
    