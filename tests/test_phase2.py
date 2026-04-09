import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pytest
from agentmem_os.storage.manager import StorageManager
from agentmem_os.storage.store import ConversationStore
from agentmem_os.llm.summarizer import SummarizationEngine
import uuid

from unittest.mock import patch

@patch('agentmem_os.llm.summarizer.SummarizationEngine._get_llm')
def test_entity_preservation(mock_get_llm):
    from langchain_core.outputs import Generation, LLMResult
    
    # Mock LLM chain response for entity extraction
    mock_llm = mock_get_llm.return_value
    mock_llm.generate.return_value = LLMResult(generations=[[Generation(text="Alice, Bob, Project Apollo")]])
    mock_llm.invoke.return_value = {"text": "Alice, Bob, Project Apollo"}
    
    engine = SummarizationEngine()
    test_text = "Alice and Bob had a meeting on January 15th 2026 to discuss Project Apollo."
    entities = engine.extract_entities(test_text)
    
    assert "Alice" in entities or "Bob" in entities or "Project Apollo" in entities

@patch('agentmem_os.llm.summarizer.SummarizationEngine.compress')
def test_branch_snapshot_generation(mock_compress):
    # Mock compress to return a fake compressed string and entities
    mock_compress.return_value = ("[Context]\nMocked summary via patch.\n[Key Entities]\nMocked", [])
    
    store = ConversationStore()
    sess_id = f"sess-p2-{uuid.uuid4().hex[:6]}"
    root = store.get_or_create_session(sess_id, name="root-session2")
    
    for i in range(5):
        store.save_turn(root.session_id, "user", f"Here is dummy payload {i} for testing compression strings over token limits.")
        store.save_turn(root.session_id, "assistant", f"I received the dataset. Acknowledging loop {i} complete.")
        
    child = store.create_branch(root.session_id, "test-branch")
    
    # Assert snapshot occurred
    assert child.inherited_context is not None
    assert child.inherited_context != "Snapshot Pending"
    assert "[Context]" in child.inherited_context
    assert "[Key Entities]" in child.inherited_context
if __name__ == '__main__':
    print('Running tests...')
    test_entity_preservation()
    print('Entity test passed')
    test_branch_snapshot_generation()
    print('Snapshot test passed')
