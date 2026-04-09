import os
import chromadb
from chromadb.config import Settings
from agentmem_os.storage.manager import StorageManager
from agentmem_os.llm.summarizer import get_embedder

class ChromaManager:
    """
    Manages semantic memory using ChromaDB.
    Strictly isolated namespaces per session.
    """
    def __init__(self):
        sm = StorageManager()
        self.vector_dir = sm.get_path("vectors")
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(
            path=self.vector_dir,
            settings=Settings(anonymized_telemetry=False)
        )

    def _get_embedding_fn(self):
        # Wrap langchain embeddings for ChromaDB
        class ChromaEmbedder:
            def __init__(self, embedder):
                self.embedder = embedder
            def __call__(self, input):
                return self.embedder.embed_documents(input)
            def embed_query(self, input):
                return self.embedder.embed_documents(input)
            def embed_documents(self, input):
                return self.embedder.embed_documents(input)
            def name(self):
                return "langchain-custom-embedder"
                
        return ChromaEmbedder(get_embedder())

    def get_or_create_collection(self, session_id: str):
        """One collection per session for semantic isolation."""
        return self.client.get_or_create_collection(
            name=session_id,
            embedding_function=self._get_embedding_fn()
        )

    def add_summary_chunk(self, session_id: str, chunk_id: str, content: str, metadata: dict):
        col = self.get_or_create_collection(session_id)
        col.add(
            ids=[chunk_id],
            documents=[content],
            metadatas=[metadata]
        )

    def add_turn_chunk(self, session_id: str, chunk_id: str, content: str, metadata: dict):
        # We can also index important individual turns, but typically we index the summaries.
        self.add_summary_chunk(session_id, chunk_id, content, metadata)

    def search(self, session_id: str, query: str, top_k: int = 5) -> list[str]:
        col = self.get_or_create_collection(session_id)
        
        if col.count() == 0:
            return []
            
        fetch_k = min(top_k * 3, col.count())
        if fetch_k == 0:
            return []
            
        res = col.query(
            query_texts=[query],
            n_results=fetch_k,
            include=["documents", "embeddings"]
        )
        
        if not res['documents']:
            return []
            
        docs = res["documents"][0]
        embeddings = res["embeddings"][0]
        
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
        query_emb = self._get_embedding_fn()([query])[0]
        q_emb = np.array(query_emb)
        doc_embs = [np.array(e) for e in embeddings]
        
        selected = []
        candidates = list(range(len(docs)))
        
        while len(selected) < top_k and candidates:
            best_score = -float('inf')
            best_idx = -1
            
            for i in candidates:
                sim_to_query = cosine_similarity(q_emb, doc_embs[i])
                
                if not selected:
                    max_sim_to_selected = 0
                else:
                    max_sim_to_selected = max(cosine_similarity(doc_embs[i], doc_embs[s]) for s in selected)
                
                # MMR Equation (lambda = 0.5)
                mmr_score = 0.5 * sim_to_query - 0.5 * max_sim_to_selected
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
                    
            if best_idx != -1:
                selected.append(best_idx)
                candidates.remove(best_idx)
                
        return [docs[idx] for idx in selected]
