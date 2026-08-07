import json
import os
import redis
import logging
from typing import List, Dict
from agentmem_os.storage.manager import StorageManager

class RedisCache:
    """
    Hot Cache storing the last `max_turns` turns per active session in
    Redis memory (20 — must cover the deepest product read, the context
    assembler's get_history(last_n=20)).

    AGENTMEM_OS_DISABLE_REDIS=1 disables the cache entirely (Stage 5
    finding: keys are session-id-only — no DB identity in the key — so a
    scratch-DB test or benchmark against a live localhost Redis reads
    GHOST turns from earlier runs and writes its own. Isolation pins on
    the SQLite path alone cannot cover this channel; this switch does.)
    """
    def __init__(self, host='localhost', port=6379, db=0):
        if os.environ.get("AGENTMEM_OS_DISABLE_REDIS") == "1":
            logging.info("Redis hot cache disabled via AGENTMEM_OS_DISABLE_REDIS.")
            self.client = None
        else:
            try:
                self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
                self.client.ping()
            except redis.ConnectionError:
                logging.warning("Redis not running. Hot cache disabled for this session.")
                self.client = None

        self.storage_manager = StorageManager()
        # 20, not 10 (Stage 5 G3 R1 B2): the product's deepest hot read
        # is the context assembler's get_history(last_n=20) — a cache
        # shallower than its main caller can NEVER hit once the
        # depth-contract check exists, degrading L1 to a pure write
        # amplifier. The cache must be at least as deep as the deepest
        # read it claims to serve.
        self.max_turns = 20

    def push_turn(self, session_id: str, turn: Dict):
        if not self.client:
            return

        key = f"agentmem_os:session:{session_id}:turns"
        self.client.lpush(key, json.dumps(turn))
        self.client.ltrim(key, 0, self.max_turns - 1)

    def replace_history(self, session_id: str, turns: List[Dict]):
        """Atomically REPLACE the cached list (oldest-first input, same
        shape get_history returns). Repopulating a warm cache with
        push_turn appended duplicates (Stage 5 G3 R1 B2 — measured
        ['t1'..'t5','t1'..'t5'] after one fall-through); replacement is
        the only repopulation that cannot corrupt. Disclosed window
        (G3 R2 n2): a save_turn landing between the caller's SQLite
        read and this pipeline is wiped from the CACHE until the next
        save_turn re-pushes it — SQLite stays authoritative and
        correct throughout; the cache self-heals."""
        if not self.client:
            return
        key = f"agentmem_os:session:{session_id}:turns"
        pipe = self.client.pipeline()
        pipe.delete(key)
        for turn in turns:
            pipe.lpush(key, json.dumps(turn))
        pipe.ltrim(key, 0, self.max_turns - 1)
        pipe.execute()

    def get_history(self, session_id: str) -> List[Dict]:
        if not self.client:
            return []

        key = f"agentmem_os:session:{session_id}:turns"
        turns = self.client.lrange(key, 0, -1)
        return [json.loads(turn) for turn in reversed(turns)]

    def persist(self):
        if self.client:
            try:
                self.client.bgsave()
            except redis.exceptions.ResponseError as e:
                pass
