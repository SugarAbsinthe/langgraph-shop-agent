"""Generate product embeddings and store in ChromaDB for shopping guide RAG.

Three collections:
  product_descriptions — product-level natural language descriptions
  product_specs         — individual spec fields extracted from JSON
  product_reviews     — review snippets by aspect
"""

import json
import sqlite3
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def build_product_embeddings(
    products_db: str, reviews_db: str, persist_dir: str, embedding_model_name: str
) -> None:
    print(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=persist_dir)

    for name in ["product_descriptions", "product_specs", "product_reviews"]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    desc_col = client.create_collection("product_descriptions")
    spec_col = client.create_collection("product_specs")
    review_col = client.create_collection("product_reviews")

    # Load products
    conn = sqlite3.connect(products_db)
    conn.row_factory = sqlite3.Row
    products = [dict(r) for r in conn.execute("SELECT * FROM products ORDER BY product_id")]
    conn.close()

    # Level 1: Product descriptions
    print(f"Embedding {len(products)} product descriptions...")
    for p in products:
        desc = p["description"]
        embedding = model.encode(desc).tolist()
        desc_col.add(
            ids=[f"product_{p['product_id']}"],
            documents=[desc],
            embeddings=[embedding],
            metadatas=[{
                "product_id": p["product_id"],
                "name": p["name"],
                "brand": p["brand"],
                "category": p["category"],
                "subcategory": p["subcategory"],
                "price": p["price"],
                "rating": p["rating"],
                "sales_count": p["sales_count"],
                "release_date": p["release_date"],
            }],
        )

    # Level 2: Individual specs
    print("Embedding spec-level descriptions...")
    for p in products:
        specs = json.loads(p["specs"])
        for attr, value in specs.items():
            spec_text = f"{p['name']}（{p['category']}）的{attr}: {value}"
            embedding = model.encode(spec_text).tolist()
            spec_col.add(
                ids=[f"spec_{p['product_id']}_{attr}"],
                documents=[spec_text],
                embeddings=[embedding],
                metadatas=[{
                    "product_id": p["product_id"],
                    "product_name": p["name"],
                    "brand": p["brand"],
                    "category": p["category"],
                    "subcategory": p["subcategory"],
                    "attribute": attr,
                    "value": str(value),
                }],
            )

    # Level 3: Review snippets
    conn = sqlite3.connect(reviews_db)
    conn.row_factory = sqlite3.Row
    reviews = [dict(r) for r in conn.execute("SELECT * FROM product_reviews ORDER BY review_id")]
    conn.close()

    print(f"Embedding {len(reviews)} review snippets...")
    for r in reviews:
        rtext = f"[{r['sentiment']}] {r['aspect']}: {r['content']}"
        embedding = model.encode(rtext).tolist()
        review_col.add(
            ids=[f"review_{r['review_id']}"],
            documents=[rtext],
            embeddings=[embedding],
            metadatas=[{
                "product_id": r["product_id"],
                "aspect": r["aspect"],
                "sentiment": r["sentiment"],
            }],
        )

    print(f"Done! Embeddings stored in {persist_dir}")
    print(f"  product_descriptions: {desc_col.count()} entries")
    print(f"  product_specs: {spec_col.count()} entries")
    print(f"  product_reviews: {review_col.count()} entries")


def main():
    base_dir = Path(__file__).resolve().parent.parent.parent
    products_db = str(base_dir / "data" / "products.db")
    reviews_db = str(base_dir / "data" / "product_reviews.db")
    persist_dir = str(base_dir / "data" / "product_chroma_db")
    embedding_model = "BAAI/bge-small-zh-v1.5"
    build_product_embeddings(products_db, reviews_db, persist_dir, embedding_model)


if __name__ == "__main__":
    main()
