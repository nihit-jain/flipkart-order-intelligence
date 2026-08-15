import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from agent.knowledge_base import get_policy_documents


MODEL_NAME = "all-MiniLM-L6-v2"
# Minimum cosine similarity required for a policy answer.
# This will be calibrated against our retrieval tests.
POLICY_SIMILARITY_THRESHOLD = 0.45
INDEX_DIR = Path("data") / "policy_index"
INDEX_PATH = INDEX_DIR / "policy.faiss"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"


def chunk_document(document: dict) -> list[dict]:
    """Split one policy document into sentence-level chunks."""

    sentences = re.split(r"(?<=[.!?])\s+", document["text"].strip())

    chunks = []

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if not sentence:
            continue

        chunks.append(
            {
                "chunk_id": f'{document["id"]}_chunk_{index}',
                "document_id": document["id"],
                "title": document["title"],
                "text": sentence,
            }
        )

    return chunks


def build_chunks() -> list[dict]:
    """Convert all policy documents into sentence-level chunks."""

    documents = get_policy_documents()

    chunks = []

    for document in documents:
        chunks.extend(chunk_document(document))

    return chunks


def build_index():
    """Create and save the FAISS policy index."""

    chunks = build_chunks()

    texts = [chunk["text"] for chunk in chunks]

    print(f"Policy documents: {len(get_policy_documents())}")
    print(f"Policy chunks: {len(chunks)}")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    import numpy as np

    np.save(EMBEDDINGS_PATH, embeddings)

    metadata_path = INDEX_DIR / "chunks.npy"
    np.save(
        metadata_path,
        np.array(chunks, dtype=object),
        allow_pickle=True,
    )

    print("FAISS index saved successfully.")
    print(f"Index: {INDEX_PATH}")
    print(f"Metadata: {metadata_path}")

    return index, chunks, model


def load_index():
    """Load the saved FAISS index, chunks, and embedding model."""

    import numpy as np

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "Policy index not found. Run build_index() first."
        )

    metadata_path = INDEX_DIR / "chunks.npy"

    index = faiss.read_index(str(INDEX_PATH))

    chunks = np.load(
        metadata_path,
        allow_pickle=True,
    ).tolist()

    model = SentenceTransformer(MODEL_NAME)

    return index, chunks, model


def search_policies(
    query: str,
    top_k: int = 3,
):
    """Return the most relevant policy chunks for a query."""

    index, chunks, model = load_index()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for score, index_position in zip(
        scores[0],
        indices[0],
    ):
        if index_position < 0:
            continue

        chunk = chunks[index_position].copy()
        chunk["score"] = float(score)

        results.append(chunk)

    return results

def check_groundedness(
    results: list[dict],
    threshold: float = POLICY_SIMILARITY_THRESHOLD,
) -> dict:
    """
    Check whether retrieved policy evidence is strong enough
    to support a grounded answer.
    """

    if not results:
        return {
            "grounded": False,
            "best_score": 0.0,
            "threshold": threshold,
        }

    best_score = max(
        float(result["score"])
        for result in results
    )

    return {
        "grounded": best_score >= threshold,
        "best_score": round(best_score, 4),
        "threshold": threshold,
    }

if __name__ == "__main__":
    build_index()

    print("\nTesting retrieval...\n")

    results = search_policies(
        "How long can I return an apparel item?",
        top_k=3,
    )

    for result in results:
        print(
            f"[{result['score']:.4f}] "
            f"{result['title']}: "
            f"{result['text']}"
        )