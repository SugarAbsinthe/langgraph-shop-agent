"""Dual-layer user profile store for shopping guide Agent.

Structured layer (SQLite): key-value constraints with confidence scores.
Semantic layer (ChromaDB): embedded user utterances for fuzzy memory recall.
Time decay: confidence decays exponentially; stale entries auto-pruned.
"""
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _days_since(iso_ts: str) -> float:
    """Return days elapsed since the given ISO timestamp."""
    try:
        ts = datetime.fromisoformat(iso_ts)
        delta = datetime.now(timezone.utc) - ts.replace(tzinfo=timezone.utc)
        return max(0, delta.total_seconds() / 86400.0)
    except Exception:
        return 0


class ProfileStore:
    """Manages structured and semantic user profiles across multiple conversations.

    Structured: SQLite table with (conv_id, key, value, confidence, timestamp, source).
    Semantic:  ChromaDB collection with embedded user utterances.
    """

    def __init__(
        self,
        db_path: str,
        chroma_dir: str,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        decay_lambda: float = 0.05,
    ):
        self.db_path = db_path
        self.decay_lambda = decay_lambda
        self.model = SentenceTransformer(embedding_model)

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

        self.client = chromadb.PersistentClient(path=chroma_dir)
        try:
            self.memory_col = self.client.get_collection("user_memory")
        except Exception:
            self.memory_col = self.client.create_collection("user_memory")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                profile_key TEXT NOT NULL,
                profile_value TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 1.0,
                last_updated TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'explicit',
                UNIQUE(conv_id, profile_key)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_profiles_conv
            ON user_profiles(conv_id)
        """)
        conn.commit()
        conn.close()

    # ---- Structured Profile ----

    def update(self, conv_id: str, key: str, value: str,
               confidence: float = 1.0, source: str = "explicit") -> None:
        """Upsert a profile key-value pair."""
        conn = sqlite3.connect(self.db_path)
        now = _now_iso()
        conn.execute("""
            INSERT INTO user_profiles (conv_id, profile_key, profile_value, confidence, last_updated, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(conv_id, profile_key) DO UPDATE SET
                profile_value = excluded.profile_value,
                confidence = excluded.confidence,
                last_updated = excluded.last_updated,
                source = excluded.source
        """, (conv_id, key, value, confidence, now, source))
        conn.commit()
        conn.close()

    def get_structured(self, conv_id: str) -> dict:
        """Return the current effective profile for a conversation.

        Applies time decay: effective_confidence = confidence * exp(-λ * days).
        Entries with effective confidence below 0.15 are excluded.
        """
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT profile_key, profile_value, confidence, last_updated, source "
            "FROM user_profiles WHERE conv_id = ?", (conv_id,)
        ).fetchall()
        conn.close()

        profile = {}
        for key, value, conf, ts, src in rows:
            days = _days_since(ts)
            effective = conf * math.exp(-self.decay_lambda * days)
            if effective >= 0.15:
                profile[key] = {
                    "value": value,
                    "confidence": round(effective, 3),
                    "source": src,
                }
        return profile

    def clear_conv(self, conv_id: str) -> None:
        """Remove all profile entries for a conversation."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM user_profiles WHERE conv_id = ?", (conv_id,))
        conn.commit()
        conn.close()

    # ---- Semantic Memory ----

    def add_memory(self, conv_id: str, utterance: str,
                   topic: str = "", metadata: Optional[dict] = None) -> None:
        """Embed and store a user utterance for later semantic recall."""
        embedding = self.model.encode(utterance).tolist()
        ts = _now_iso()
        meta = {
            "conv_id": conv_id,
            "topic": topic,
            "timestamp": ts,
        }
        if metadata:
            meta.update(metadata)

        mem_id = f"mem_{conv_id}_{int(time.time() * 1000)}"
        self.memory_col.add(
            ids=[mem_id],
            documents=[utterance],
            embeddings=[embedding],
            metadatas=[meta],
        )

    def search_semantic(self, conv_id: str, query: str,
                        top_k: int = 5) -> list[str]:
        """Search user memory for semantically similar past utterances."""
        embedding = self.model.encode(query).tolist()
        try:
            results = self.memory_col.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where={"conv_id": conv_id},
                include=["documents", "distances"],
            )
        except Exception:
            return []

        if not results["ids"] or not results["ids"][0]:
            return []

        memories = []
        for i, doc in enumerate(results["documents"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0
            memories.append(f"[dist={distance:.3f}] {doc}")
        return memories

    # ---- Utility ----

    def serialize_profile(self, conv_id: str) -> str:
        """Format the structured profile as a concise prompt string."""
        profile = self.get_structured(conv_id)
        if not profile:
            return "(暂无画像)"

        lines = []
        for key, info in profile.items():
            conf_pct = int(info["confidence"] * 100)
            lines.append(f"- {key}: {info['value']} (置信度 {conf_pct}%)")
        return "\n".join(lines)

    def serialize_memories(self, conv_id: str, query: str) -> str:
        """Retrieve and format relevant user memories for the prompt."""
        memories = self.search_semantic(conv_id, query, top_k=3)
        if not memories:
            return ""
        return "## 用户历史偏好记忆\n" + "\n".join(f"- {m}" for m in memories)
