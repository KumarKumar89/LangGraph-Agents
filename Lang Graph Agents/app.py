import os
from typing import TypedDict, Annotated, List
from langchain_core.messages import (
    AnyMessage, SystemMessage, HumanMessage, ToolMessage, messages_from_dict
)
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dotenv import load_dotenv

# Optional: wrap Tavily import so it doesn't break during Studio load
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

load_dotenv()

# --- 1. Set Up Environment ---
if TavilyClient:
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
else:
    tavily = None


# --- 2. Define Tools ---
@tool
def search_tool(query: str):
    """
    Searches the web for information on a given topic using the Tavily search engine.
    Returns a list of relevant search results.
    """
    print(f"--- EXECUTING SEARCH: {query} ---")
    if not tavily:
        return [{"title": "Mock Result", "content": "Tavily client not initialized."}]
    results = tavily.search(query=query, max_results=5)
    return results["results"]


@tool
def write_report_tool(content: str, filename: str = "research_report.md"):
    """
    Writes the given content to a markdown file.
    """
    print(f"--- WRITING REPORT: {filename} ---")
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"✅ Successfully wrote report to {filename}"


tools = [search_tool, write_report_tool]


# --- 3. Define Graph State ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], lambda x, y: x + y]


# --- 4. Initialize Model ---
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)
model_with_tools = model.bind_tools(tools)


# --- 5. Helper Function ---
def normalize_messages(messages):
    """Ensure all messages are LangChain Message objects."""
    normalized = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, SystemMessage, ToolMessage)):
            normalized.append(msg)
        else:
            # Convert dict-like messages to LC message objects
            try:
                normalized.append(messages_from_dict([msg])[0])
            except Exception:
                normalized.append(HumanMessage(content=str(msg)))
    return normalized


# --- 6. Node Functions ---
def planner_node(state: AgentState):
    """Decide what to do next (search, synthesize, etc.)."""
    print("--- PLANNING ---")
    state["messages"] = normalize_messages(state["messages"])

    system_prompt = (
        "You are an expert research assistant. Use the search tool to gather information, "
        "then synthesize it into a report. Finally, save it using the report writing tool. "
        "Do not answer from your own knowledge."
    )

    messages_with_prompt = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages_with_prompt)
    return {"messages": [response]}


tool_node = ToolNode(tools)


def response_synthesizer_node(state: AgentState):
    """Synthesize the search results into a report."""
    print("--- SYNTHESIZING RESPONSE ---")
    state["messages"] = normalize_messages(state["messages"])

    # Extract content safely
    first_message = state["messages"][0]
    last_message = state["messages"][-1]
    user_request = getattr(first_message, "content", str(first_message))
    tool_output = getattr(last_message, "content", str(last_message))

    synthesis_prompt = (
        "You are a report writer. Synthesize the provided search results into a clear, "
        "well-structured research report. The user's original request was: "
        f"'{user_request}'.\n\nSearch Results:\n{tool_output}\n\n"
        "After writing, call the `write_report_tool` to save it."
    )

    response = model_with_tools.invoke(synthesis_prompt)
    return {"messages": [response]}


def router_node(state: AgentState):
    """Route to the correct next node."""
    print("--- ROUTING ---")
    state["messages"] = normalize_messages(state["messages"])
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        if last_message.name == "search_tool":
            print("➡️ Route: synthesize")
            return "synthesize"
        print("✅ Route: end")
        return "end"

    elif hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("➡️ Route: tools")
        return "tools"

    print("✅ Route: end")
    return "end"


# --- 7. Graph Creation ---
def create_research_graph() -> StateGraph:
    """Create and return the research workflow graph."""
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_node("synthesizer", response_synthesizer_node)

    graph_builder.set_entry_point("planner")

    graph_builder.add_conditional_edges(
        "planner", router_node, {"tools": "tools", "end": END}
    )
    graph_builder.add_conditional_edges(
        "tools", router_node, {"synthesize": "synthesizer", "end": END}
    )
    graph_builder.add_edge("synthesizer", "tools")

    return graph_builder


def create_compiled_graph():
    """Compile and return the research graph."""
    graph = create_research_graph()
    return graph.compile()
