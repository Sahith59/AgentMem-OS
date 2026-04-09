import logging
from typing import List, Dict, Tuple
from agentmem_os.storage.manager import StorageManager

# Lazy-load to prevent import halting
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            _embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        except ImportError:
            # Fallback for environments without sentence-transformers
            from langchain_community.embeddings import FakeEmbeddings
            _embedder = FakeEmbeddings(size=384)
    return _embedder

class SummarizationEngine:
    def __init__(self, model_name="llama3.1", threshold=0.92):
        self.model_name = model_name
        self.threshold = threshold
        self.llm = None  # Init only when needed

    def _get_llm(self):
        if self.llm is None:
            import os
            from langchain_community.chat_models import ChatLiteLLM
            # Use local Qwen 2.5 as default for summarization, configurable later
            model_name = os.environ.get("MEMNAI_SUMMARIZER_MODEL", "ollama/qwen2.5:14b")
            self.llm = ChatLiteLLM(model=model_name, temperature=0.1)
        return self.llm

    def extract_entities(self, text: str) -> List[str]:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import os
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
            
        doc = nlp(text)
        entities = set()
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT", "WORK_OF_ART", "EVENT"]:
                entities.add(ent.text)
                
        return list(entities)[:15]

    def is_duplicate(self, text: str, existing_summaries: List[str]) -> bool:
        if not existing_summaries:
            return False
            
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        embedder = get_embedder()
        
        # LangChain embeddings require .embed_query() for single and .embed_documents() for lists
        new_emb = embedder.embed_query(text)
        existing_embs = embedder.embed_documents(existing_summaries)
        
        similarities = cosine_similarity([new_emb], existing_embs)[0]
        return np.max(similarities) > self.threshold

    def compress(self, turns: List[Dict]) -> Tuple[str, List[str]]:
        raw_text = "\n".join([f"{t['role']}: {t['content']}" for t in turns])
        entities = self.extract_entities(raw_text)
        
        from langchain.chains.summarize import load_summarize_chain
        from langchain.docstore.document import Document
        
        chunk_size = 4000
        docs = [Document(page_content=raw_text[i:i+chunk_size]) for i in range(0, len(raw_text), chunk_size)]
        
        chain = load_summarize_chain(self._get_llm(), chain_type="map_reduce")
        res = chain.invoke(docs)
        summary = res["output_text"].strip()
        
        missing_entities = [ent for ent in entities if ent.lower() not in summary.lower()]
        if missing_entities:
            summary += "\n\n[Key Entities] (Enforced)\n" + ", ".join(missing_entities)
            
        return summary, entities
