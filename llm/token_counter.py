import tiktoken

class TokenCounter:
    """
    Utility for counting the number of tokens in user or assistant content
    before appending them into the SQLite database.
    MemNAI defaults to openai's tiktoken parser for tokenization.
    """
    def __init__(self, model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.encoder = self._get_encoder()

    def _get_encoder(self):
        try:
            return tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        return len(self.encoder.encode(text))
