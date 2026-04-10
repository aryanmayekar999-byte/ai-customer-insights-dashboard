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
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(query_embedding, doc_embeddings, docs, k=3):
    similarities = [
        cosine_similarity(query_embedding, emb)
        for emb in doc_embeddings
    ]

    top_k_idx = np.argsort(similarities)[-k:][::-1]

    return [docs[i] for i in top_k_idx]