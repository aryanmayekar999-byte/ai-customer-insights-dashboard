import re

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


def _tokenize(text):
    return set(re.findall(r"\b\w+\b", str(text).lower()))


def _lexical_overlap_score(query_text, document):
    query_tokens = _tokenize(query_text)
    if not query_tokens:
        return 0.0

    document_tokens = _tokenize(document)
    if not document_tokens:
        return 0.0

    return len(query_tokens & document_tokens) / len(query_tokens)


def rerank_documents(results, query_text, semantic_weight=0.8):
    """Rerank retrieved results with a simple semantic + lexical score."""
    if not query_text:
        return results

    semantic_weight = min(max(semantic_weight, 0.0), 1.0)
    lexical_weight = 1.0 - semantic_weight

    reranked = []
    for result in results:
        semantic_score = result["score"]
        lexical_score = _lexical_overlap_score(query_text, result["document"])
        rerank_score = (semantic_weight * semantic_score) + (
            lexical_weight * lexical_score
        )

        updated_result = {
            **result,
            "semantic_score": semantic_score,
            "lexical_score": lexical_score,
            "rerank_score": rerank_score,
        }
        reranked.append(updated_result)

    return sorted(reranked, key=lambda result: result["rerank_score"], reverse=True)


def _metadata_matches(metadata, metadata_filter):
    if metadata_filter is None:
        return True

    metadata = metadata or {}

    if callable(metadata_filter):
        return metadata_filter(metadata)

    if not isinstance(metadata_filter, dict):
        raise TypeError("metadata_filter must be a dict, callable, or None.")

    return all(metadata.get(key) == value for key, value in metadata_filter.items())


def _filter_by_metadata(doc_embeddings, docs, metadatas, metadata_filter):
    if metadatas is None:
        metadatas = [{} for _ in docs]

    if len(metadatas) != len(docs):
        raise ValueError("metadatas and docs must have the same length.")

    filtered = []
    for embedding, doc, metadata in zip(doc_embeddings, docs, metadatas):
        metadata = metadata or {}
        if _metadata_matches(metadata, metadata_filter):
            filtered.append((embedding, doc, metadata))

    if not filtered:
        return [], [], []

    filtered_embeddings, filtered_docs, filtered_metadatas = zip(*filtered)
    return list(filtered_embeddings), list(filtered_docs), list(filtered_metadatas)


def retrieve_similar_documents(
    query_embedding,
    doc_embeddings,
    docs,
    k=5,
    metadatas=None,
    metadata_filter=None,
    query_text=None,
    rerank=False,
    rerank_candidate_multiplier=3,
    semantic_weight=0.8,
):
    """Retrieve documents with FAISS, optional metadata filtering, and reranking."""
    if faiss is None:
        raise ImportError(
            "faiss is not installed. Install it with `pip install faiss-cpu`."
        )

    if len(doc_embeddings) == 0 or len(docs) == 0:
        return []

    if len(doc_embeddings) != len(docs):
        raise ValueError("doc_embeddings and docs must have the same length.")

    if k <= 0:
        return []

    doc_embeddings, docs, metadatas = _filter_by_metadata(
        doc_embeddings,
        docs,
        metadatas,
        metadata_filter,
    )

    if not docs:
        return []

    embeddings = np.asarray(doc_embeddings, dtype="float32").copy()
    query = np.asarray([query_embedding], dtype="float32").copy()

    faiss.normalize_L2(embeddings)
    faiss.normalize_L2(query)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    candidate_k = min(k, len(docs))
    if rerank and query_text:
        candidate_k = min(len(docs), k * max(1, rerank_candidate_multiplier))

    distances, indices = index.search(query, candidate_k)

    results = [
        {
            "document": docs[index],
            "metadata": metadatas[index],
            "score": float(score),
        }
        for index, score in zip(indices[0], distances[0])
        if index != -1
    ]

    if rerank:
        results = rerank_documents(
            results,
            query_text,
            semantic_weight=semantic_weight,
        )

    return results[:k]


def retrieve_top_5_similar_documents(
    query_embedding,
    doc_embeddings,
    docs,
    metadatas=None,
    metadata_filter=None,
    query_text=None,
    rerank=False,
    rerank_candidate_multiplier=3,
    semantic_weight=0.8,
):
    """Retrieve the top 5 most similar documents using FAISS and metadata."""
    return retrieve_similar_documents(
        query_embedding,
        doc_embeddings,
        docs,
        k=5,
        metadatas=metadatas,
        metadata_filter=metadata_filter,
        query_text=query_text,
        rerank=rerank,
        rerank_candidate_multiplier=rerank_candidate_multiplier,
        semantic_weight=semantic_weight,
    )


def retrieve_top_k(
    query_embedding,
    doc_embeddings,
    docs,
    k=3,
    metadatas=None,
    metadata_filter=None,
    query_text=None,
    rerank=False,
    rerank_candidate_multiplier=3,
    semantic_weight=0.8,
):
    return [
        result["document"]
        for result in retrieve_similar_documents(
            query_embedding,
            doc_embeddings,
            docs,
            k=k,
            metadatas=metadatas,
            metadata_filter=metadata_filter,
            query_text=query_text,
            rerank=rerank,
            rerank_candidate_multiplier=rerank_candidate_multiplier,
            semantic_weight=semantic_weight,
        )[:k]
    ]
