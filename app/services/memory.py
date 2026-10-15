"""Redis-backed conversation memory (one capped, expiring list per session)."""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from redis.asyncio import Redis


class StoredMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class RedisChatMemory:
    """Stores the last ``max_turns`` user/assistant exchanges per session, with a TTL."""

    def __init__(self, redis: Redis, max_turns: int, ttl_seconds: int) -> None:
        self._redis = redis
        self._max_messages = max_turns * 2
        self._ttl = ttl_seconds

    @staticmethod
    def _key(session_id: str) -> str:
        return f"chat:{session_id}:messages"

    async def get_history(self, session_id: str) -> list[StoredMessage]:
        raw: list[Any] = await self._redis.lrange(self._key(session_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def append_exchange(self, session_id: str, user: str, assistant: str) -> None:
        key = self._key(session_id)
        messages: list[StoredMessage] = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.rpush(key, *(json.dumps(m) for m in messages))
            pipe.ltrim(key, -self._max_messages, -1)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    async def close(self) -> None:
        await self._redis.aclose()
