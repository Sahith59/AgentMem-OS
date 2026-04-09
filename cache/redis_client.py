import json
import redis
import logging
from typing import List, Dict
from memnai.storage.manager import StorageManager

class RedisCache:
    """
    Hot Cache storing the last 10 turns per active session in Redis memory.
    """
    def __init__(self, host='localhost', port=6379, db=0):
        try:
            self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.client.ping()
        except redis.ConnectionError:
            logging.warning("Redis not running. Hot cache disabled for this session.")
            self.client = None

        self.storage_manager = StorageManager()
        self.max_turns = 10

    def push_turn(self, session_id: str, turn: Dict):
        if not self.client:
            return
        
        key = f"memnai:session:{session_id}:turns"
        self.client.lpush(key, json.dumps(turn))
        self.client.ltrim(key, 0, self.max_turns - 1)

    def get_history(self, session_id: str) -> List[Dict]:
        if not self.client:
            return []
            
        key = f"memnai:session:{session_id}:turns"
        turns = self.client.lrange(key, 0, -1)
        return [json.loads(turn) for turn in reversed(turns)]

    def persist(self):
        if self.client:
            try:
                self.client.bgsave()
            except redis.exceptions.ResponseError as e:
                pass
