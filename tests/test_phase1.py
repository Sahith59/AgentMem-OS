import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
import uuid
from memnai.storage.manager import StorageManager
from memnai.storage.store import ConversationStore

def test_storage_manager_fallback():
    dummy_config = {
        "storage": {
            "base_path": "/dummy_volume/memnai/",
            "fallback_path": "~/.memnai_fallback/",
            "warn_on_fallback": False
        }
    }
    
    import yaml
    with open("dummy_config.yaml", "w") as f:
        yaml.dump(dummy_config, f)
        
    sm = StorageManager(config_path="dummy_config.yaml")
    
    assert sm.is_fallback_active() is True
    assert sm.active_path == os.path.expanduser("~/.memnai_fallback/")
    os.remove("dummy_config.yaml")

def test_branch_creation():
    store = ConversationStore()
    sess_id = f"sess-1-{uuid.uuid4().hex[:6]}"
    root = store.get_or_create_session(sess_id, name="root-session")
    store.save_turn(root.session_id, "user", "Hello world!")
    store.save_turn(root.session_id, "assistant", "Hi there!")
    
    child = store.create_branch(root.session_id, "approach-b")
    
    assert child.parent_session_id == sess_id
    assert child.branch_point_turn > 0
    assert child.branch_type == "hard"
    
    branches = store.list_branches(sess_id)
    assert len(branches) == 2 
