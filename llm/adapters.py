import litellm
from memnai.db.database import get_session
from memnai.db.models import CostLog
from memnai.llm.context_assembler import ContextAssembler

class UniversalAdapter:
    def __init__(self):
        self.assembler = ContextAssembler()

    def send_message(self, session_id: str, query: str, model: str = "ollama/qwen2.5:14b"):
        """
        Assembles context and routes to any LLM via LiteLLM.
        Logs token usage and cost automatically.
        """
        # 1. Assemble context within token bounds
        context_str = self.assembler.assemble(session_id, query)
        
        # 2. Prepare message stack
        messages = [
            {"role": "system", "content": context_str},
            {"role": "user", "content": query}
        ]

        # 3. Apply explicit prompt caching if Claude
        # Anthropic requires cache_control INSIDE the content block, not as a top-level
        # message key. LiteLLM passes it through correctly only when content is a list
        # of typed content blocks — a plain string content field is NOT patchable.
        if "claude" in model.lower():
            messages[0]["content"] = [
                {
                    "type": "text",
                    "text": context_str,
                    "cache_control": {"type": "ephemeral"},
                }
            ]

        # 4. Trigger universal LLM call
        import litellm
        
        # Setup automatic failover strategy based on primary model requested.
        # Only add fallbacks when the corresponding API key is actually present —
        # LiteLLM raises on the fallback attempt itself when keys are missing.
        fallbacks = []
        model_str = model.lower()
        import os
        if "groq" in model_str and os.environ.get("GROQ_API_KEY"):
            fallbacks = ["ollama/qwen2.5:14b"]
        elif "claude" in model_str and os.environ.get("GROQ_API_KEY"):
            fallbacks = ["groq/llama-3.1-8b-instant"]
            
        litellm.suppress_debug_info = True
        import logging
        logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
            
        response = litellm.completion(
            model=model,
            messages=messages,
            fallbacks=fallbacks,
            temperature=0.7
        )
        
        content = response.choices[0].message.content
        
        # 5. Extract telemetry and calculate cost
        usage = response.get("usage", {})
        input_toks = getattr(usage, "prompt_tokens", 0)
        output_toks = getattr(usage, "completion_tokens", 0)
        cache_toks = getattr(usage, "cache_read_input_tokens", 0)
        
        try:
            total_cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            total_cost = 0.0
            
        # 6. Log telemetry to database
        db = get_session()
        try:
            log = CostLog(
                session_id=session_id,
                model=model,
                input_tokens=input_toks,
                output_tokens=output_toks,
                cached_tokens=cache_toks,
                cost_usd=total_cost
            )
            db.add(log)
            db.commit()
        finally:
            db.close()

        return content
