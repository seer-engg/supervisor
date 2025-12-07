"""
Graph definition for LangGraph Cloud deployment.
Exports the supervisor graph that can be deployed to LangGraph Cloud.

Note: 
- Execution traces are automatically saved by LangGraph Cloud (no store needed for traces)
- LangGraph Cloud handles store persistence automatically if needed
"""
from agents.supervisor import create_supervisor

# Create the graph
graph = create_supervisor()

