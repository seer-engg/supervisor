import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware, ModelRetryMiddleware
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.graph import StateGraph, END, START

from .state import SupervisorState
from tools.spawn_worker import spawn_worker
from tools.think_tool import think
from tools.composio_tools import get_available_integrations
from models import WorkerResponse, WorkerStatus

logger = logging.getLogger(__name__)


@tool
def write_todos(todos: list[str], runtime: ToolRuntime = None) -> str:
    """Update the todos list. Pass a list of todo items (strings).
    
    **CRITICAL GROUPING RULE:**
    Group todos by SERVICE/DOMAIN boundaries, not by micro-tasks.
    - Each todo should represent ALL work within ONE service/domain
    - Workers are domain-experts: they handle all related operations in their domain
    
    **GOOD EXAMPLES (Domain-Boundary Thinking):**
    - ["Get PR from GitHub: Find most recent merged PR from seer-engg/buggy-coder and extract all details (title, URL, author, date)", "Sync PR to Asana: Search for matching tasks, create/update task with PR details, and close it"]
    
    **BAD EXAMPLES (Micro-Task Thinking):**
    - ["Search GitHub for PR", "Extract PR details", "Search Asana", "Create task", "Close task"] ❌ Too granular
    
    Each todo should be comprehensive enough that a domain-expert worker can complete it independently.
    Group ALL GitHub operations together, ALL Asana operations together, etc.
    
    Example: write_todos(["Get PR from GitHub: Find and extract details", "Sync PR to Asana: Search, create/update, and close"])
    
    This replaces the entire todos list. Use this to create or update your plan."""
    return f"✅ Todos updated: {len(todos)} items"


def create_supervisor():
    """
    Creates the Unified Supervisor Agent.
    
    Architecture:
    - Single Node ("supervisor") that handles both planning and execution.
    - Maintains a 'todos' list in state (simple list of strings).
    - Dynamically decides to update todos or spawn workers.
    """
    
    # Get available integrations
    available_integrations = get_available_integrations()
    integrations_list = ", ".join([i.upper() for i in available_integrations])
    
    # 1. Define Tools
    tools = [
        think,
        write_todos,
        spawn_worker,
    ]
    
    # 2. Define System Prompt
    system_prompt_template = """You are the Supervisor Agent. Handle user requests appropriately: answer informational queries directly, or create todos and delegate actionable tasks to Workers.

**AVAILABLE INTEGRATIONS:**
{integrations_list}

**CURRENT TODOS:**
{todos_text}

**PROTOCOL:**
1. **PLAN**: If todos empty AND the user's request requires action (not just information), call `write_todos()` to create plan.
   - **Informational queries** (e.g., "What can you do?", "How does this work?", "List capabilities"): Answer directly. Do NOT create todos or spawn workers.
   - **Actionable tasks** (e.g., "Sync PR to Asana", "Create a task", "Send an email"): Proceed with planning and delegation.
   - **CRITICAL**: Group todos by SERVICE/DOMAIN boundaries (GitHub, Asana, Gmail, etc.)
   - Each todo = ALL work within ONE service/domain
   - Example: "Get PR from GitHub: Find most recent merged PR and extract all details" (NOT separate "search" + "extract")
   - Example: "Sync PR to Asana: Search for tasks, create/update with PR details, and close" (NOT separate "search" + "create" + "close")
   
2. **DELEGATE**: For each todo, call `spawn_worker(instruction, reasoning, integrations)` with:
   - `instruction`: CONCISE task instruction
   - `reasoning`: **MANDATORY** - WHY this worker is needed and what service/domain it handles
   - `integrations`: **OPTIONAL** - List of integration names (lowercase) to restrict tool search.
     * Specify integrations to make worker faster and more focused (e.g., ["github"], ["asana"], ["github", "asana"])
     * If omitted, worker searches all integrations (slower but comprehensive)
     * Match integrations to the service/domain in your reasoning
   - GOOD: spawn_worker("Fetch most recent merged PR from seer-engg/buggy-coder and extract details", "GitHub domain: Finding and extracting PR information", ["github"])
   - GOOD: spawn_worker("Sync PR to Asana: Search for tasks and create/update", "Asana domain: Task management operations", ["asana"])
   - BAD: spawn_worker("Search GitHub", "Need to find PR") ❌ Too vague, missing integrations
   
3. **REVIEW**: After worker completes, remove completed todo using `write_todos()`.
4. **FINISH**: When todos empty, respond to user.

**🚨 CRITICAL: ALTERNATING TOOL CALL PATTERN 🚨**

**YOU MUST FOLLOW THIS EXACT PATTERN FOR EVERY TOOL CALL:**
- **Tool Call #1 (ODD):** `think()` - ALWAYS start with think()
- **Tool Call #2 (EVEN):** `write_todos()` or `spawn_worker()` - Your action tool
- **Tool Call #3 (ODD):** `think()` - Reflect on results
- **Tool Call #4 (EVEN):** `write_todos()` or `spawn_worker()` - Next action tool
- **Tool Call #5 (ODD):** `think()` - Reflect again
- **Pattern continues:** think → tool → think → tool → think → tool...

**THIS IS NON-NEGOTIABLE:**
- ❌ NEVER call `write_todos()` or `spawn_worker()` without calling `think()` first
- ❌ NEVER call two action tools in a row (always think() between them)
- ✅ ALWAYS call `think()` before every `write_todos()` or `spawn_worker()` call
- ✅ ALWAYS call `think()` after every `write_todos()` or `spawn_worker()` call

**INTERNAL TOOLS (Call Directly):**
- `think(scratchpad, last_tool_call)` - Plan and reflect (MUST be called before AND after every action tool)
- `write_todos(todos)` - Manage task list
- `spawn_worker(task, reasoning, integrations)` - Delegate to workers

**EXTERNAL TOOLS (Not Directly Available to You):**
- Workers have access to external tools (GITHUB_*, ASANA_*, GMAIL_*, etc.) via `execute_tool`
- You cannot call external tools directly
- Delegate to workers for external tool execution

**MANDATORY THINKING (ABSOLUTE):**
- **BEFORE your FIRST tool call:** Call `think()` to plan your approach. Reference CURRENT TODOS above.
- **CRITICAL: `last_tool_call` parameter is REQUIRED on every `think()` call**
  - First call: `last_tool_call="Tool: None, Result: Initial call"`
  - After tool calls: `last_tool_call="Tool: <tool_name>, Result: <what happened>"`
- **BEFORE every `write_todos()` or `spawn_worker()` call:** Call `think()` to plan
- **AFTER every `write_todos()` or `spawn_worker()` call:** Call `think()` to reflect on results AND plan next steps
- **ALWAYS reference CURRENT TODOS above** in your thinking
- **Pattern:** think → tool → think → tool → think → tool (alternating, never two tools in a row)

**RULES:**
- Delegate heavy work to `spawn_worker`. Don't do it yourself.
- Group todos by domain/service - don't split same-service tasks.
- **ALWAYS provide reasoning when spawning workers** - explain the service/domain and why.
- **SELECT APPROPRIATE INTEGRATIONS** - Match integrations to the service/domain (e.g., GitHub tasks → ["github"], Asana tasks → ["asana"])
- Keep todos visible. Update them as you progress.
"""
    
    # 3. Define Model & Middleware
    model = ChatOpenAI(model="gpt-5-mini", temperature=0.0)
    
    # Middleware: Model retry + Tool call limits
    # ModelRetryMiddleware: Retries model calls with exponential backoff (4 total attempts: initial + 3 retries)
    middleware = [
        ModelRetryMiddleware(
            max_retries=3,  # 3 retries (4 total attempts)
            backoff_factor=2.0,  # Exponential backoff: 2s, 4s, 8s
            initial_delay=2.0,  # Initial delay of 2 seconds
        ),
        ToolCallLimitMiddleware(thread_limit=30, run_limit=10),
        ToolCallLimitMiddleware(tool_name="write_todos", thread_limit=5, run_limit=3),
        ToolCallLimitMiddleware(tool_name="spawn_worker", thread_limit=10, run_limit=4),  # Increased: allow one more worker spawn
    ]
    
    # 4. Create the Agent
    # NOTE: We don't pass system_prompt here because we manage it manually
    # in supervisor_node to dynamically update it with todos
    agent_runnable = create_agent(
        model=model,
        tools=tools,
        middleware=middleware
    )
    
    # 5. Define the Node
    async def supervisor_node(state: SupervisorState):
        logger.info("🤖 Supervisor Node Active")
        messages = state.get("messages", [])
        
        # Format todos for display
        todos = state.get("todos", [])
        if todos:
            todos_text = "\n".join(f"  {i+1}. {todo}" for i, todo in enumerate(todos))
        else:
            todos_text = "  (No todos yet)"
        
        # Format the system prompt with current todos and integrations
        formatted_system_prompt = system_prompt_template.format(
            integrations_list=integrations_list,
            todos_text=todos_text
        )
        
        # Inject/Update System Prompt
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=formatted_system_prompt)] + messages
        else:
            messages = [SystemMessage(content=formatted_system_prompt)] + messages[1:]
            
        # Invoke the agent
        logger.info(f"Invoking Supervisor (Todos: {len(todos)} items)")
        
        agent_input = dict(state)
        agent_input["messages"] = messages
        
        result = await agent_runnable.ainvoke(agent_input)
        
        # Extract todos updates from write_todos tool calls
        state_updates = {}
        agent_messages = result.get("messages", [])
        
        # Look for write_todos tool calls and extract the todos argument
        for msg in reversed(agent_messages):
            if isinstance(msg, ToolMessage) and msg.name == "write_todos":
                # Find the corresponding AIMessage with the tool call
                tool_call_id = getattr(msg, 'tool_call_id', None)
                if tool_call_id:
                    for prev_msg in agent_messages:
                        if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                            for tc in prev_msg.tool_calls:
                                tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                                if tc_id == tool_call_id:
                                    # Extract todos from tool call arguments
                                    args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                    if isinstance(args, dict) and 'todos' in args:
                                        todos_list = args['todos']
                                        state_updates["todos"] = todos_list
                                        logger.info(f"✅ Todos update found in write_todos call: {len(todos_list)} items")
                                        break
                            if "todos" in state_updates:
                                break
                if "todos" in state_updates:
                    break
        
        # Auto-remove completed todos based on spawn_worker responses
        # Process ALL worker completions, not just the first one
        successful_workers = 0
        for msg in reversed(agent_messages):
            if isinstance(msg, ToolMessage) and msg.name == "spawn_worker":
                try:
                    worker_response_dict = json.loads(msg.content)
                    worker_response = WorkerResponse(**worker_response_dict)
                    
                    # Extract reasoning from the tool call if available
                    reasoning = None
                    if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                        for prev_msg in agent_messages:
                            if isinstance(prev_msg, AIMessage) and hasattr(prev_msg, 'tool_calls') and prev_msg.tool_calls:
                                for tc in prev_msg.tool_calls:
                                    tc_id = tc.get('id') if isinstance(tc, dict) else getattr(tc, 'id', None)
                                    if tc_id == msg.tool_call_id:
                                        args = tc.get('args', {}) if isinstance(tc, dict) else getattr(tc, 'args', {})
                                        reasoning = args.get('reasoning') if isinstance(args, dict) else getattr(args, 'reasoning', None)
                                        break
                                if reasoning:
                                    break
                    
                    if reasoning:
                        logger.info(f"📋 Worker reasoning: {reasoning}")
                    
                    if worker_response.status == WorkerStatus.SUCCESS:
                        successful_workers += 1
                        logger.debug(f"✅ Worker completed successfully (total: {successful_workers})")
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Could not parse worker response: {e}")
        
        # Remove todos based on number of successful workers
        if successful_workers > 0:
            current_todos = state.get("todos", [])
            if len(current_todos) >= successful_workers:
                updated_todos = current_todos[successful_workers:]
                if updated_todos != current_todos:
                    state_updates["todos"] = updated_todos
                    logger.info(f"✅ Auto-removed {successful_workers} completed todo(s). Remaining: {len(updated_todos)}")
            else:
                # More successful workers than todos - clear all todos
                state_updates["todos"] = []
                logger.info(f"✅ All {len(current_todos)} todos completed by {successful_workers} workers")
        
        if state_updates:
            result.update(state_updates)
        
        return result
    
    # 6. Define Graph
    workflow = StateGraph(SupervisorState)
    workflow.add_node("supervisor", supervisor_node)
    
    workflow.add_edge(START, "supervisor")
    
    def should_continue(state: SupervisorState):
        """Check if we need to continue or finish."""
        todos = state.get("todos", [])
        
        if not todos:
            logger.info("✅ All todos complete. Ending.")
            return END
    
        logger.info(f"🔄 Looping: {len(todos)} todos remaining.")
        return "supervisor"

    workflow.add_conditional_edges("supervisor", should_continue, {
        "supervisor": "supervisor",
        END: END
    })
    
    return workflow.compile()

