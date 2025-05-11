from dotenv import load_dotenv
load_dotenv() 
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import ToolNode
from schemas import AnswerQuestion,ReviseAnswer

wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)

search_tool = DuckDuckGoSearchResults(api_wrapper=wrapper)



def run_queries(search_queries: list[str], **kwargs):
    """Run the generated queries"""
    return search_tool.batch([{"query": query} for query in search_queries])


excute_tools = ToolNode([
    StructuredTool.from_function(run_queries, name=AnswerQuestion.__name__),
    StructuredTool.from_function(run_queries, name=ReviseAnswer.__name__)
    
])
