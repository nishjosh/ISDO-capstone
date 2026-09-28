"""
Build a ChromaDB knowledge base from markdown articles and test it.

Steps:
  1. Read all .md files in data/kb/
  2. Split each file into chunks at '## ' headings
  3. Store every chunk in the ChromaDB collection 'isdo_kb'
  4. Run 4 sample queries and print the best-matching article + confidence

Requires: pip install chromadb
(The first run downloads ChromaDB's default embedding model, ~80 MB.)
"""

import re
from pathlib import Path

import chromadb

KB_DIR = Path("data/kb")
DB_DIR = "data/chroma_db"
COLLECTION_NAME = "isdo_kb"


def split_into_chunks(text: str) -> list[tuple[str, str]]:
    """Split markdown text at '## ' headings.

    Returns a list of (section_heading, chunk_text) tuples. Any text before
    the first '## ' heading (e.g. the '# Title' and intro) becomes its own
    chunk labelled 'Introduction'.
    """
    # Split just before every line that starts with '## ' (but not '### ')
    parts = re.split(r"(?m)^(?=## )", text)
    chunks = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        first_line = part.splitlines()[0]
        if first_line.startswith("## "):
            heading = first_line[3:].strip()
        else:
            heading = "Introduction"
        chunks.append((heading, part))
    return chunks


def load_articles() -> tuple[list[str], list[str], list[dict]]:
    """Read every .md file and return ids, documents and metadatas."""
    ids, documents, metadatas = [], [], []

    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR.resolve()}")

    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(text)
        print(f"  {md_file.name}: {len(chunks)} chunks")

        for i, (heading, chunk_text) in enumerate(chunks):
            ids.append(f"{md_file.stem}_chunk{i}")
            # Prefix the article name so each chunk keeps its context
            documents.append(f"Article: {md_file.stem}\n\n{chunk_text}")
            metadatas.append(
                {"article": md_file.name, "section": heading, "chunk_index": i}
            )

    return ids, documents, metadatas


def build_collection(client: chromadb.ClientAPI):
    """(Re)create the collection so reruns don't leave stale chunks."""
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    # Cosine distance -> confidence = 1 - distance (range roughly 0..1)
    return client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def run_test_queries(collection) -> None:
    sample_queries = [
        "I forgot my password and I'm locked out of my account",
        "My laptop won't connect to the office Wi-Fi",
        "The printer shows jobs in the queue but nothing prints",
        "VPN keeps disconnecting when I work from home",
    ]

    print("\n" + "=" * 70)
    print("TEST QUERIES")
    print("=" * 70)

    for query in sample_queries:
        results = collection.query(query_texts=[query], n_results=3)

        best_meta = results["metadatas"][0][0]
        best_distance = results["distances"][0][0]
        confidence = 1 - best_distance

        print(f"\nQuery:      {query}")
        print(f"Best match: {best_meta['article']}  (section: {best_meta['section']})")
        print(f"Confidence: {confidence:.2%}")

        # Show runners-up for context
        for meta, dist in zip(results["metadatas"][0][1:], results["distances"][0][1:]):
            print(f"   also: {meta['article']} / {meta['section']}  ({1 - dist:.2%})")


def main() -> None:
    print(f"Reading articles from {KB_DIR}/ ...")
    ids, documents, metadatas = load_articles()

    client = chromadb.PersistentClient(path=DB_DIR)
    collection = build_collection(client)
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"\nStored {collection.count()} chunks in collection '{COLLECTION_NAME}'")

    run_test_queries(collection)


if __name__ == "__main__":
    main()