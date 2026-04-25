import numpy as np

try:
    import faiss
except ImportError:  # pragma: no cover - handled at runtime for missing dependency
    faiss = None


def chunk_text(text, chunk_size=100):
    sentences = text.split(".")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + "."
        else:
            chunks.append(current)
            current = sentence + "."

    if current:
        chunks.append(current)

    return chunks

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def retrieve_similar_documents(query_embedding, doc_embeddings, docs, k=5):
    """Retrieve the most similar documents using a FAISS cosine index."""
    if faiss is None:
        raise ImportError(
            "faiss is not installed. Install it with `pip install faiss-cpu`."
        )

    if not doc_embeddings or not docs:
        return []

    if len(doc_embeddings) != len(docs):
        raise ValueError("doc_embeddings and docs must have the same length.")

    embeddings = np.asarray(doc_embeddings, dtype="float32").copy()
    query = np.asarray([query_embedding], dtype="float32").copy()

    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    k = min(k, len(docs))
    distances, indices = index.search(query, k)

    return [
        {"document": docs[index], "score": float(score)}
        for index, score in zip(indices[0], distances[0])
        if index != -1
    ]


def retrieve_top_5_similar_documents(query_embedding, doc_embeddings, docs):
    """Retrieve the top 5 most similar documents using FAISS."""
    return retrieve_similar_documents(query_embedding, doc_embeddings, docs, k=5)


def retrieve_top_k(query_embedding, doc_embeddings, docs, k=3):
    return [
        result["document"]
        for result in retrieve_similar_documents(
            query_embedding,
            doc_embeddings,
            docs,
            k=k,
        )[:k]
    ]
