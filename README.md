# Delegate Model

A baseline delegate orchestrator with explicit planning and dynamic worker spawning for multi-step integration tasks.

## Architecture Overview

The Delegate Model implements a supervisor-worker architecture where:

- **Supervisor Agent**: Creates a plan (todos list) and delegates tasks to workers
- **Generic Workers**: Dynamically spawned agents that use semantic tool discovery to complete domain-specific tasks
- **Context Isolation**: Workers execute in isolated threads, preventing context bloat
- **Plan Visibility**: Execution plan stored in state, visible and mutable

### Key Components

#### Supervisor (`agents/supervisor.py`)
- Maintains a `todos` list in state (simple list of strings)
- Groups todos by SERVICE/DOMAIN boundaries (GitHub, Asana, Gmail, etc.)
- Dynamically spawns workers via `spawn_worker()` tool
- Auto-updates todos based on worker completion status

#### Generic Worker (`agents/generic_worker.py`)
- Created on-demand with task-specific instructions
- Uses `search_tools()` and `execute_tool()` for dynamic tool discovery
- Executes in isolated context (ephemeral)
- Returns structured `WorkerResponse` with status and results

#### Tools
- **`spawn_worker`**: Dynamically creates and executes workers
- **`search_tools`**: Semantic search for available tools (via ToolHub + Composio)
- **`execute_tool`**: Executes tools with validation and error handling
- **`think`**: Explicit reasoning tool for planning and reflection
- **`write_todos`**: Updates the supervisor's todo list

## Installation

```bash
pip install -r requirements.txt
```

## Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key for LLM calls
- `COMPOSIO_USER_ID`: Composio user ID for tool access

Optional (for semantic tool search):
- `TOOL_HUB_PATH`: Path to ToolHub installation
- `TOOL_HUB_INDEX_DIR`: Path to ToolHub index directory

Optional (for secrets):
- `ASANA_WORKSPACE_ID`: Asana workspace ID
- `ASANA_PROJECT_ID`: Asana project ID

## Usage

### Basic Example

```python
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from agents.supervisor import create_supervisor

async def main():
    # Create shared memory store
    store = InMemoryStore()
    
    # Create supervisor agent
    agent = create_supervisor(store=store)
    
    # Task instruction
    task = "Sync a GitHub PR to Asana: Find the most recent merged PR from seer-engg/buggy-coder, extract details, and create/update an Asana task."
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [],
        "tool_call_counts": {"_total": 0}
    }
    
    # Execute
    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": "example-run"}
    }
    
    result = await agent.ainvoke(initial_state, config=config)
    
    # Get final output
    messages = result.get("messages", [])
    final_output = messages[-1].content if messages else ""
    print(final_output)

if __name__ == "__main__":
    asyncio.run(main())
```

### With LangFuse Tracing

```python
from langfuse.langchain import CallbackHandler
from langfuse.types import TraceContext

# Setup tracing
trace_context = TraceContext(
    session_id="my-session",
    user_id="user-123",
    tags=["delegate-model"]
)
langfuse_handler = CallbackHandler(trace_context=trace_context)

config = {
    "recursion_limit": 10,
    "callbacks": [langfuse_handler],
    "configurable": {"thread_id": "traced-run"}
}

result = await agent.ainvoke(initial_state, config=config)
```

## Architecture Details

### Planning Phase
1. Supervisor receives user task
2. If `todos` is empty, supervisor calls `write_todos()` to create plan
3. Todos are grouped by domain/service boundaries
4. Example: `["Get PR from GitHub: Find and extract details", "Sync PR to Asana: Search, create/update, and close"]`

### Execution Phase
1. For each todo, supervisor calls `spawn_worker(instruction, reasoning)`
2. Worker is created dynamically with task-specific instructions
3. Worker uses `search_tools()` to discover needed capabilities
4. Worker plans execution via `think()` tool
5. Worker executes tools via `execute_tool()`
6. Worker returns structured `WorkerResponse`
7. Supervisor auto-removes completed todos

### Context Isolation
- Each worker executes in its own isolated thread
- Worker's context is ephemeral (destroyed after completion)
- Large outputs can be saved to shared store if needed
- Supervisor maintains minimal context (todos + worker summaries)

## File Structure

```
delegate-model/
├── agents/
│   ├── __init__.py
│   ├── supervisor.py      # Main orchestrator
│   ├── generic_worker.py  # Dynamic worker implementation
│   └── state.py           # State definitions
├── tools/
│   ├── __init__.py
│   ├── spawn_worker.py    # Worker spawning tool
│   ├── composio_tools.py  # Tool discovery & execution
│   ├── think_tool.py      # Reasoning tool
│   ├── openai_retry_middleware.py  # Retry logic
│   ├── secrets_store.py   # Secrets management
│   ├── runtime_tool_store.py  # Tool schema cache
│   └── memory_tools.py    # Shared memory utilities
├── models.py              # Pydantic models
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Key Features

1. **Explicit Planning**: Dedicated planning phase before execution
2. **Dynamic Worker Spawning**: Workers created on-demand, not pre-configured
3. **Generic Workers**: Single worker type with dynamic tool discovery
4. **Plan Visibility**: Plan stored in state, visible and mutable
5. **Context Isolation**: Workers execute in isolated threads
6. **Full Observability**: LangFuse traces show complete flow

## Limitations

- ToolHub is optional but recommended for semantic tool search
- Currently optimized for Composio tool integrations
- Secrets store currently only supports Asana workspace/project IDs
- Worker auto-removal of todos uses simple heuristic (first todo removed on success)

## License

[Add your license here]

