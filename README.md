# Supervisor

An orchestrator with explicit planning and dynamic worker spawning for multi-step integration tasks.

## Architecture

- **Supervisor**: Creates plan (todos) and delegates to workers
- **Generic Workers**: Dynamically spawned agents using semantic tool discovery (ToolHub + Pinecone + Composio)
- **Context Isolation**: Workers execute in isolated threads
- **Tools**: `spawn_worker`, `search_tools`, `execute_tool`, `think`, `write_todos`

## Installation

```bash
pip install -e .
```

## Environment Variables

**Required:** `OPENAI_API_KEY`, `COMPOSIO_USER_ID`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`

**Optional:** `COMPOSIO_API_KEY`

**Note:** Supervisor no longer uses hardcoded integration-specific IDs (like `ASANA_WORKSPACE_ID`, `GITHUB_REPO_ID`, etc.). Workers discover resources dynamically using the user's connected accounts. If sandbox mode is needed, it should be passed from the frontend via context.

**Note**: Tools must be pre-computed in Pinecone before use.

## Usage

```python
import asyncio
from langchain_core.messages import HumanMessage
from langgraph.store.memory import InMemoryStore
from agents.supervisor import create_supervisor

async def main():
    agent = create_supervisor(store=InMemoryStore())
    result = await agent.ainvoke({
        "messages": [HumanMessage(content="Sync GitHub PR to Asana...")],
        "todos": [],
        "tool_call_counts": {"_total": 0}
    }, {"recursion_limit": 10, "configurable": {"thread_id": "run"}})
    print(result["messages"][-1].content)

asyncio.run(main())
```

## Deployment

### Option 1: LangGraph Cloud (Managed)

1. `pip install langgraph-cli` → `langgraph login` → `langgraph deploy`

**Notes:** Traces auto-saved (Langfuse API). Store persistence handled by Cloud. Pre-compute tools in Pinecone first.

### Option 2: Self-Hosted with LangGraph (Recommended)

Deploy on Railway, Render, AWS, or any platform using LangGraph API:

1. **Install dependencies:** `pip install -e .`
2. **Set environment variables** (see above)
3. **Run LangGraph server:** `langgraph dev --port 8000 --host 0.0.0.0 --config langgraph.json`
4. **Deploy using Docker:** Use the provided `Dockerfile` for containerized deployment

**Advantages:**
- ✅ Full control over infrastructure
- ✅ Native LangGraph API (no LangServe wrapper)
- ✅ Same API capabilities (streaming, batch, etc.)
- ✅ Consistent with Seer deployment model

## Pre-computing Tool Index

**One-time operation** - indexes Composio tools in Pinecone.

1. Create Pinecone account (https://www.pinecone.io/, free tier: 1M vectors)
2. Get API key from dashboard
3. Run: `python -m tool_hub.precompute_pinecone_index` (from `tool_hub` directory)

**Process:** Creates index → Fetches tools from Composio → Enriches with LLM metadata (dependencies, neighbors) → Stores in Pinecone with integration filtering.

**Required env vars:** `OPENAI_API_KEY`, `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `COMPOSIO_USER_ID`, `COMPOSIO_API_KEY` (optional)

**Update integrations:** Edit `tool_hub/tool_hub/precompute_pinecone_index.py` → `INTEGRATIONS` list.

**Important:** Run once before deployment. Index persists. Idempotent. Takes time (2000+ tools).

## Composio Setup

1. Create account (https://composio.dev) → Connect services via dashboard → Get `COMPOSIO_USER_ID` → Set in `.env`
2. Optional: Get `COMPOSIO_API_KEY` from settings

Credentials managed through Composio (not `.env`). Multi-user: use each user's `COMPOSIO_USER_ID`.

## Features

- Explicit planning phase
- Dynamic worker spawning (on-demand)
- Semantic tool discovery (Hub & Spoke: semantic + dependency expansion)
- Plan visibility (stored in state)
- Context isolation
- Full observability (Langfuse API)

## Limitations

- Requires pre-computed Pinecone index
- Optimized for Composio integrations
- Worker auto-removal uses simple heuristic
