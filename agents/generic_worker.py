from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from typing import Optional, List

from tools.composio_tools import search_tools, execute_tool
from tools.think_tool import think
from tools.secrets_store import _secrets_store

from .prompts import PROMPT_GENERIC_WORKER

def create_generic_worker(
    role_name: str, 
    specific_instructions: str,
    integrations: Optional[List[str]] = None
):
    """
    Creates a Generic Worker Sub-Agent.
    
    This agent is NOT hardcoded with specific tools.
    Instead, it uses `search_tools` and `execute_tool` to dynamically find what it needs.
    
    Args:
        role_name: Name of the role (e.g. "GitHub Researcher")
        specific_instructions: Task-specific system prompt additions.
        integrations: Optional list of integration names (lowercase, e.g., ["github", "asana"]).
                     If None, searches all integrations (slower but comprehensive).
                     If provided, restricts tool search to specified integrations.
    """
    
    # 1. Generic Toolset
    # Workers can search for tools and execute them. All tool outputs are visible in worker's isolated context.
    tools = [
        think,
        search_tools, 
        execute_tool, 
    ]
    
    # Check if this is an Asana-related task and inject secrets
    secrets_context = ""
    if integrations and "asana" in integrations:
        secrets = _secrets_store.get_all()
        if secrets:
            secrets_context = "\n\n**ASANA CONFIGURATION:**\n"
            secrets_context += _secrets_store.format_for_prompt()
            secrets_context += "\nUse these IDs when calling Asana tools that require workspace_gid or project_gid parameters.\n"
    
    # Add integration context to system prompt if integrations specified
    integration_context = ""
    if integrations:
        integration_names = ", ".join([i.upper() for i in integrations])
        integration_list = ", ".join([f'"{i}"' for i in integrations])
        integration_context = f"\n\n**INTEGRATION DOMAIN:** You are working with {integration_names} tools. When calling `search_tools`, use the `integration_filter` parameter to restrict searches to these integrations: `integration_filter=[{integration_list}]`. This makes searches faster and more relevant."
    
    # 2. System Prompt (PRUNED - ~500 chars vs ~1200)
    system_prompt = PROMPT_GENERIC_WORKER.format(
        ROLE_NAME=role_name,
        SPECIFIC_INSTRUCTIONS=specific_instructions,
        SECRETS_CONTEXT=secrets_context,
        INTEGRATION_CONTEXT=integration_context
    )
    
    # 3. Model and Middleware
    # We use a capable model since it needs to reason about tool discovery
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    
    # Middleware: Model retry + Tool call limits
    # ModelRetryMiddleware: Retries model calls with exponential backoff (4 total attempts: initial + 3 retries)
    # Tool call limits middleware - DOUBLED LIMITS for better worker autonomy
    middleware = [
        ModelRetryMiddleware(
            max_retries=3,  # 3 retries (4 total attempts)
            backoff_factor=2.0,  # Exponential backoff: 2s, 4s, 8s
            initial_delay=2.0,  # Initial delay of 2 seconds
        ),
        ToolCallLimitMiddleware(thread_limit=40, run_limit=16),  # Doubled global limit
        ToolCallLimitMiddleware(tool_name="search_tools", thread_limit=10, run_limit=6),  # Doubled
        ToolCallLimitMiddleware(tool_name="execute_tool", thread_limit=20, run_limit=10),  # Doubled
    ]
    
    # 4. Create agent using create_agent - it returns a compiled graph
    # create_agent handles the entire agent loop internally and accepts system_prompt directly
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        middleware=middleware
    )
    
    return agent

