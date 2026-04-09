from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from agentmem_os.storage.store import ConversationStore
from agentmem_os.storage.manager import StorageManager
from agentmem_os.llm.adapters import UniversalAdapter

app = FastAPI(title="MemNAI Context & Storage API")

# Setup CORS for the local web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared instances
store = ConversationStore()
adapter = UniversalAdapter()
manager = StorageManager()

# Models
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = "groq/llama-3.1-8b-instant" # Default local, accepts claude/gpt as alternate

class BranchRequest(BaseModel):
    parent_id: str
    branch_name: str

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Ensure session exists
    store.get_or_create_session(req.session_id, name=req.session_id, model=req.model)
    
    # Save user turn
    store.save_turn(req.session_id, "user", req.message)
    
    # Generate
    try:
        response_text = adapter.send_message(req.session_id, req.message, model=req.model)
        store.save_turn(req.session_id, "assistant", response_text)
        return {"session_id": req.session_id, "reply": response_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    turns = store.get_history(session_id, last_n=limit)
    return {"turns": turns}

@app.get("/branch/list/{session_id}")
async def list_branches(session_id: str):
    branches = store.list_branches(session_id)
    return [{"id": b.session_id, "parent": b.parent_session_id, "name": b.name, "tokens": b.total_tokens} for b in branches]

@app.post("/branch/create")
async def create_branch(req: BranchRequest):
    try:
        child = store.create_branch(req.parent_id, req.branch_name)
        return {"status": "success", "new_session_id": child.session_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/storage/status")
async def storage_status():
    return {
        "active_path": manager.active_path,
        "is_fallback": manager.is_fallback_active()
    }

if __name__ == "__main__":
    uvicorn.run("agentmem_os.api.app:app", host="0.0.0.0", port=8000, reload=True)
