"""Product retriever for shopping guide Agent.

Three-level retrieval pipeline:
  1. Query product_descriptions for overall product matches
  2. Query product_specs for attribute-level matching
  3. Query product_reviews for relevant user feedback

The result is formatted as a structured context block for the LLM prompt.
"""
import json
import sqlite3
from typing import Optional

from sentence_transformers import SentenceTransformer
import chromadb


class ProductRetriever:
    """Hybrid product retrieval across descriptions, specs, and reviews."""

    def __init__(self, chroma_dir: str, embedding_model: str = "BAAI/bge-small-zh-v1.5",
                 catalog_db: str = None, cache=None):
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.desc_col = self.client.get_collection("product_descriptions")
        self.spec_col = self.client.get_collection("product_specs")
        self.review_col = self.client.get_collection("product_reviews")
        self.catalog_db = catalog_db
        self._cache = cache

    def retrieve(self, query: str, top_k: int = 5,
                 filters: Optional[dict] = None) -> str:
        """Retrieve and format product context for the LLM prompt."""
        # Check RAG cache first (Redis miss returns None → fall through)
        if self._cache is not None and filters is None:
            cached = self._cache.get(query, top_k)
            if cached is not None:
                return cached

        where = None
        if filters:
            where = {}
            for k, v in filters.items():
                if isinstance(v, list):
                    where[k] = {"$in": v}
                else:
                    where[k] = v

        query_embedding = self.model.encode(query).tolist()

        # 1. Product descriptions
        desc_results = self.desc_col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        seen_ids = set()
        products = []
        if desc_results["ids"] and desc_results["ids"][0]:
            for i, _pid in enumerate(desc_results["ids"][0]):
                meta = desc_results["metadatas"][0][i]
                prod_id = meta["product_id"]
                if prod_id not in seen_ids:
                    seen_ids.add(prod_id)
                    products.append(meta)

        # 2. Broaden search via specs
        if len(products) < top_k:
            spec_results = self.spec_col.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,
                where=where,
                include=["metadatas"],
            )
            if spec_results["ids"] and spec_results["ids"][0]:
                for i, _sid in enumerate(spec_results["ids"][0]):
                    meta = spec_results["metadatas"][0][i]
                    prod_id = meta["product_id"]
                    if prod_id not in seen_ids:
                        seen_ids.add(prod_id)
                        products.append({
                            "product_id": prod_id,
                            "name": meta["product_name"],
                            "brand": meta["brand"],
                            "category": meta["category"],
                            "subcategory": meta["subcategory"],
                        })
                    if len(products) >= top_k:
                        break

        # 3. Retrieve relevant reviews
        matched_pids = [p["product_id"] for p in products]
        reviews_by_pid = {}
        if matched_pids:
            review_results = self.review_col.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 3,
                where={"product_id": {"$in": matched_pids[:10]}},
                include=["documents", "metadatas"],
            )
            if review_results["ids"] and review_results["ids"][0]:
                for i, _rid in enumerate(review_results["ids"][0]):
                    meta = review_results["metadatas"][0][i]
                    rev_pid = meta["product_id"]
                    text = review_results["documents"][0][i]
                    if rev_pid not in reviews_by_pid:
                        reviews_by_pid[rev_pid] = []
                    reviews_by_pid[rev_pid].append({
                        "aspect": meta["aspect"],
                        "sentiment": meta["sentiment"],
                        "content": text,
                    })

        result = self._format_for_prompt(products, reviews_by_pid)

        # Store in cache for subsequent identical queries
        if self._cache is not None and filters is None:
            self._cache.set(query, top_k, result)

        return result

    def _format_for_prompt(self, products: list[dict],
                           reviews_by_pid: dict) -> str:
        """Format retrieved products, with full specs from catalog."""
        lines = ["## 相关产品推荐\n"]

        # Pre-load specs from SQLite catalog for rich display
        specs_by_pid = {}
        desc_by_pid = {}
        if self.catalog_db:
            try:
                conn = sqlite3.connect(self.catalog_db)
                conn.row_factory = sqlite3.Row
                pids = [p["product_id"] for p in products if "product_id" in p]
                if pids:
                    placeholders = ",".join("?" * len(pids))
                    rows = conn.execute(
                        f"SELECT product_id, specs, description FROM products WHERE product_id IN ({placeholders})",
                        pids
                    ).fetchall()
                    for row in rows:
                        specs_by_pid[row["product_id"]] = row["specs"]
                        desc_by_pid[row["product_id"]] = row["description"]
                conn.close()
            except Exception:
                pass

        for i, p in enumerate(products):
            pid = p.get("product_id")

            lines.append(f"### {i+1}. {p.get('name', '?')}")
            lines.append(f"- 品牌: {p.get('brand', '?')} | 品类: {p.get('category', '?')} / {p.get('subcategory', '?')}")
            lines.append(f"- 价格: RMB{p.get('price', '?')} | 评分: {p.get('rating', '?')} 分 | 销量: {p.get('sales_count', 0)}")
            lines.append(f"- 发布日期: {p.get('release_date', '?')} | 产品ID: {pid}")

            # Rich spec details from catalog
            desc = desc_by_pid.get(pid, "")
            if desc:
                lines.append(f"- 产品简介: {desc}")

            spec_json = specs_by_pid.get(pid, "")
            if spec_json:
                try:
                    specs = json.loads(spec_json) if isinstance(spec_json, str) else spec_json
                    spec_lines = []
                    for k, v in specs.items():
                        spec_lines.append(f"  {k}: {v}")
                    if spec_lines:
                        lines.append("- 规格参数:")
                        lines.extend(spec_lines)
                except Exception:
                    pass

            # Review summary
            if pid in reviews_by_pid:
                lines.append("- 用户评价摘要:")
                for rev in reviews_by_pid[pid][:3]:
                    emoji = "+" if rev["sentiment"] == "positive" else "-"
                    lines.append(f"  [{emoji}] {rev['aspect']}: {rev['content']}")
            lines.append("")

        return "\n".join(lines)
