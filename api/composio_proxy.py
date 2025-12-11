"""
Composio API Proxy for Frontend
Handles CORS and proxies Composio API requests server-side
"""
import os
import asyncio
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)
from pydantic import BaseModel
from composio import Composio
from composio_langchain import LangchainProvider

app = FastAPI()

# CORS configuration - allow frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://app.getseer.dev",
        "http://localhost:5173",  # Vite dev server (default)
        "http://localhost:3000",  # Alternative dev port
        "http://127.0.0.1:5173",  # Alternative localhost format
        "http://127.0.0.1:3000",  # Alternative localhost format
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Composio client
COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY")
if COMPOSIO_API_KEY:
    composio_client = Composio(api_key=COMPOSIO_API_KEY, provider=LangchainProvider())
else:
    composio_client = Composio(provider=LangchainProvider())


class ConnectRequest(BaseModel):
    """Request to initiate OAuth connection"""
    user_id: str
    auth_config_id: str
    callback_url: Optional[str] = None


class WaitForConnectionRequest(BaseModel):
    """Request to wait for connection"""
    connection_id: str
    timeout_ms: Optional[int] = 120000  # 2 minutes default


@app.get("/api/composio/connected-accounts")
async def list_connected_accounts(
    user_ids: Optional[List[str]] = Query(None),
    toolkit_slugs: Optional[List[str]] = Query(None),
    auth_config_ids: Optional[List[str]] = Query(None),
):
    """
    List connected accounts for a user.
    Proxies Composio's connected_accounts.list() API.
    """
    try:
        # Composio Python SDK uses snake_case parameters
        # Wrap in asyncio.to_thread to avoid blocking the event loop
        result = await asyncio.to_thread(
            composio_client.connected_accounts.list,
            user_ids=user_ids,
            toolkit_slugs=toolkit_slugs,
            auth_config_ids=auth_config_ids,
        )
        
        # Convert to dict format for JSON response
        items = []
        if hasattr(result, 'items'):
            for item in result.items:
                try:
                    toolkit_slug = ""
                    if hasattr(item, 'toolkit'):
                        if hasattr(item.toolkit, 'slug'):
                            toolkit_slug = item.toolkit.slug
                        elif isinstance(item.toolkit, dict):
                            toolkit_slug = item.toolkit.get('slug', '')
                    elif isinstance(item, dict) and 'toolkit' in item:
                        toolkit_slug = item['toolkit'].get('slug', '') if isinstance(item['toolkit'], dict) else str(item['toolkit'])
                    
                    # Extract user_id if available (to verify format)
                    user_id = None
                    if hasattr(item, 'user_id'):
                        user_id = item.user_id
                    elif isinstance(item, dict) and 'user_id' in item:
                        user_id = item.get('user_id')
                    
                    items.append({
                        "id": item.id if hasattr(item, 'id') else (item.get('id', '') if isinstance(item, dict) else ''),
                        "status": item.status if hasattr(item, 'status') else (item.get('status', 'UNKNOWN') if isinstance(item, dict) else 'UNKNOWN'),
                        "user_id": user_id,  # Include user_id to verify format
                        "toolkit": {
                            "slug": toolkit_slug,
                        },
                    })
                except Exception as e:
                    logger.warning(f"Error parsing connected account item: {e}")
                    continue
        
        return {
            "items": items,
            "total": len(items),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list connected accounts: {str(e)}")


@app.post("/api/composio/connect")
async def initiate_connection(request: ConnectRequest):
    """
    Initiate OAuth connection flow.
    Proxies Composio's connected_accounts.link() API.
    """
    try:
        # Composio Python SDK uses snake_case parameters
        # Wrap in asyncio.to_thread to avoid blocking the event loop
        connection_request = await asyncio.to_thread(
            composio_client.connected_accounts.link,
            user_id=request.user_id,
            auth_config_id=request.auth_config_id,
            callback_url=request.callback_url,
        )
        
        # ConnectionRequest is an object with attributes, not a dict
        redirect_url = getattr(connection_request, 'redirect_url', None) or getattr(connection_request, 'redirectUrl', None)
        connection_id = getattr(connection_request, 'id', None)
        
        if not redirect_url:
            raise HTTPException(status_code=500, detail="Composio did not return a redirect URL")
        
        return {
            "redirectUrl": str(redirect_url),
            "connectionId": str(connection_id) if connection_id else "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate connection: {str(e)}")


@app.post("/api/composio/wait-for-connection")
async def wait_for_connection(request: WaitForConnectionRequest):
    """
    Wait for OAuth connection to be established.
    Proxies Composio's connected_accounts.waitForConnection() API.
    """
    try:
        # Composio Python SDK: wait_for_connection takes connection_id as positional argument
        # Wrap in asyncio.to_thread to avoid blocking the event loop
        # Signature: wait_for_connection(connection_id, timeout=None)
        result = await asyncio.to_thread(
            composio_client.connected_accounts.wait_for_connection,
            request.connection_id,  # positional argument
            request.timeout_ms,  # positional argument for timeout
        )
        
        # Result is a ConnectedAccount object with attributes, not a dict
        status = getattr(result, 'status', None) or 'UNKNOWN'
        connected_account_id = getattr(result, 'id', None)
        
        return {
            "status": str(status),
            "connectedAccountId": str(connected_account_id) if connected_account_id else "",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to wait for connection: {str(e)}")


@app.delete("/api/composio/connected-accounts/{account_id}")
async def delete_connected_account(account_id: str):
    """
    Delete a connected account.
    Proxies Composio's connected_accounts.delete() API.
    """
    try:
        # Wrap in asyncio.to_thread to avoid blocking the event loop
        await asyncio.to_thread(
            composio_client.connected_accounts.delete,
            account_id
        )
        return {"success": True, "message": f"Connected account {account_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete connected account: {str(e)}")


@app.post("/api/composio/tools/execute/{tool_slug}")
async def execute_tool(tool_slug: str, request: dict):
    """
    Execute a Composio tool.
    Proxies Composio's tools API using the same pattern as composio_tools.py.
    """
    try:
        user_id = request.get("user_id")
        connected_account_id = request.get("connected_account_id")
        arguments = request.get("arguments", {})
        
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        # Execute tool using Composio SDK's execute method
        # This method accepts connected_account_id as a parameter
        # Wrap in asyncio.to_thread to avoid blocking the event loop
        def _execute_tool():
            # Use tools.execute() which accepts connected_account_id
            # Set dangerously_skip_version_check=True to avoid version requirement
            # (matches frontend behavior where dangerouslySkipVersionCheck was used)
            result = composio_client.tools.execute(
                tool_slug,
                user_id=user_id,
                connected_account_id=connected_account_id,
                arguments=arguments,
                dangerously_skip_version_check=True,
            )
            return result
        
        result = await asyncio.to_thread(_execute_tool)
        
        # Return the result in the format expected by the frontend
        # The frontend expects response.data, so wrap the result accordingly
        if isinstance(result, dict):
            # If result is already a dict, check if it has a 'data' key
            if "data" in result:
                return {"data": result["data"], "success": True}
            else:
                return {"data": result, "success": True}
        else:
            # If result is not a dict, wrap it
            return {"data": result, "success": True}
    except Exception as e:
        logger.error(f"Error executing tool {tool_slug}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to execute tool: {str(e)}")


@app.get("/api/composio/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "composio_configured": bool(COMPOSIO_API_KEY),
    }

