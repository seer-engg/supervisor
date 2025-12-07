"""
Example usage of the Supervisor.
Demonstrates how to create and run a supervisor agent with a task.
"""
import asyncio
import logging
from langchain_core.messages import HumanMessage
from agents.supervisor import create_supervisor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Run a simple example task."""
    
    # Create supervisor agent
    logger.info("Creating supervisor agent...")
    agent = create_supervisor()
    logger.info("✓ Supervisor agent created")
    
    # Example task: GitHub ↔ Asana integration
    task = """ACTION REQUIRED: Sync a GitHub PR to Asana.

EXECUTE THESE STEPS IN ORDER:
1. Search GitHub for the most recently merged/closed PR in repository 'seer-engg/buggy-coder'.
2. Extract PR details: title, URL, author, merge/close date.
3. OPTIONALLY search Asana for tasks matching the PR title or keywords (if search fails or finds nothing, proceed to step 4).
4. DECISION:
   - IF task EXISTS (from step 3): Update the task with PR details (add comment with PR URL, author, date).
   - IF task DOES NOT EXIST (search found nothing or search was skipped): Create a new Asana task with PR title and details.
5. Close the Asana task (whether it was updated or newly created)."""
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=task)],
        "todos": [],
        "tool_call_counts": {"_total": 0}
    }
    
    # Execute
    logger.info("🚀 Starting agent execution...")
    config = {
        "recursion_limit": 10,
        "configurable": {"thread_id": "example-run"}
    }
    
    try:
        result = await agent.ainvoke(initial_state, config=config)
        
        # Get final output
        messages = result.get("messages", [])
        final_output = messages[-1].content if messages else ""
        
        logger.info("✅ Agent execution completed")
        print("\n" + "="*60)
        print("FINAL OUTPUT:")
        print("="*60)
        print(final_output)
        
        # Show todos status
        todos = result.get("todos", [])
        if todos:
            print(f"\nRemaining todos: {len(todos)}")
            for i, todo in enumerate(todos, 1):
                print(f"  {i}. {todo}")
        else:
            print("\n✅ All todos completed!")
        
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())

