"""
Graph definition for LangGraph Cloud deployment.
Exports the supervisor graph that can be deployed to LangGraph Cloud.

Note: 
- Execution traces are automatically saved by LangGraph Cloud (no store needed for traces)
- LangGraph Cloud handles store persistence automatically if needed
"""
# CRITICAL: Import config FIRST to ensure environment variables are loaded
# before importing any modules that depend on them
import config  # noqa: F401

from agents.supervisor import create_supervisor

# Create the graph
# graph = create_supervisor()

from langchain.agents import create_agent
from tools.think_tool import think

from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.graph import StateGraph, END, START
from agents.state import SupervisorState
from langchain_core.messages import AIMessage

async def test_node(state: SupervisorState):
    context = state.get("context", {})
    return {"messages": [AIMessage(content=f"Hello, world! {context}")]}

workflow = StateGraph(SupervisorState)

workflow.add_node("test", test_node)
workflow.add_node("supervisor", create_agent(
    model='gpt-5-nano',
    tools=[think],
    system_prompt="You are a supervisor. You are responsible for overseeing the execution of a task. before doing anything always use the think tool to think about the task and the tools you have available.",

))
workflow.add_edge(START, "test")
workflow.add_edge("test", "supervisor")
workflow.add_edge("supervisor", END)

workflow.compile()


graph = workflow.compile()


