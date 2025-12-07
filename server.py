#!/usr/bin/env python
"""
LangServe server for deploying the Supervisor Agent.

This wraps the LangGraph supervisor agent in a FastAPI server using LangServe,
providing the same streaming and API capabilities as LangGraph Cloud, but with
full control over the infrastructure.

Usage:
    python server.py

Or with uvicorn:
    uvicorn server:app --host 0.0.0.0 --port 8000

Environment Variables:
    OPENAI_API_KEY: Required - OpenAI API key
    COMPOSIO_USER_ID: Required - Composio user ID
    PINECONE_API_KEY: Required - Pinecone API key
    PINECONE_INDEX_NAME: Required - Pinecone index name
    COMPOSIO_API_KEY: Optional - Composio API key
    PORT: Optional - Server port (default: 8000)
"""
# CRITICAL: Import config FIRST to load and validate environment variables
# This must happen before any other imports that depend on env vars
import config  # noqa: F401

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the graph
try:
    from graph import graph
    logger.info("✅ Successfully imported supervisor graph")
except Exception as e:
    logger.error(f"❌ Failed to import graph: {e}")
    raise

# Initialize FastAPI app
app = FastAPI(
    title="Supervisor Agent API",
    version="1.0.0",
    description="Supervisor-Worker Architecture for Multi-Step Integration Tasks",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add LangServe routes
# This automatically creates endpoints:
# - POST /agent/invoke - Single invocation
# - POST /agent/stream - Streaming invocation
# - POST /agent/batch - Batch invocation
# - GET /agent/playground - Interactive playground (if enabled)
add_routes(
    app,
    graph,
    path="/agent",
    # Enable playground for testing (disable in production)
    playground_type="chat",
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint with environment variable validation."""
    env_summary = config.get_env_summary()
    
    all_required_set = all(env_summary["required"].values())
    status = "healthy" if all_required_set else "unhealthy"
    
    response = {
        "status": status,
        "service": "supervisor-agent",
        "version": "1.0.0",
        "environment": env_summary
    }
    
    if not all_required_set:
        missing = [var for var, is_set in env_summary["required"].items() if not is_set]
        response["missing_variables"] = missing
        response["error"] = f"Missing required environment variables: {', '.join(missing)}"
    
    return response

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Supervisor Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "agent_endpoint": "/agent",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get port and host from environment (Railway sets PORT automatically)
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"🚀 Starting Supervisor Agent API server on {host}:{port}")
    logger.info(f"📚 API docs available at http://{host}:{port}/docs")
    logger.info(f"🤖 Agent endpoint: http://{host}:{port}/agent")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )

