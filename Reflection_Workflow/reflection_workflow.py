from typing import List, Sequence

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END, MessageGraph, START

from IPython.display import Image
from langchain_core.runnables.graph import MermaidDrawMethod
from chains import reflection_chain, genration_chain
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s"
)

REFLECT = 'reflect'
GENERATE = "generate"


def genration_node(state: Sequence[BaseMessage]):
    # Invoke generation chain with current messages state
    return genration_chain.invoke({"messages": state})


def reflection_node(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    # Invoke reflection chain and wrap result in a HumanMessage list
    res = reflection_chain.invoke({"messages": messages})
    return [HumanMessage(content=res.content)]


builder = MessageGraph()
builder.add_node(GENERATE, genration_node)
builder.add_node(REFLECT, reflection_node)
builder.set_entry_point(GENERATE)


def should_continue(state: List[BaseMessage]):
    # Route to END if more than 6 messages, else to REFLECT
    logging.info("\nLength of the message: "+ str(len(state)))
    if len(state) > 6:
        return END  
    return REFLECT


# Add conditional edges from GENERATE node based on should_continue routing function
builder.add_conditional_edges(
    GENERATE,
    should_continue,
    {
        REFLECT : REFLECT,
        END : END
    }
)

# Add unconditional edge from REFLECT back to GENERATE
builder.add_edge(REFLECT,GENERATE)

graph = builder.compile()

# Optional: print graph visualization
print(graph.get_graph().draw_mermaid())
graph.get_graph().print_ascii()

png_bytes = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)

with open("graph_flow.png", "wb") as f:
    f.write(png_bytes)


if __name__ == "__main__":
    inputs = HumanMessage(
        content="""Write an email to a customer apologizing for a delayed shipment and explaining the reason for the delay. Offer a discount for the inconvenience."""
    )

    response = graph.invoke(inputs)

