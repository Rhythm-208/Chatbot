#Combines sparse lexical search (BM25) with dense vector search (Chroma),

import re
from collections import defaultdict
from rank_bm25 import BM25Okapi

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    return _reranker

def _tokenize(text: str) ->list[str]:
    return re.findall(r"\w+", text.lower())

def _fetch_candidate_docs(chunks_store,source_filter,limit = 1000):
    result = chunks_store.get(where = source_filter , include = ["documents","metadatas"] , limit = limit)

    return result.get("documents", []) , result.get("metadatas", [])

def hybrid_retrieval(chunks_store,query: str , source_filter = None ,  k: int = 4, vector_k: int = 10, bm25_k: int = 10,
                     rrf_k: int = 60, use_reranker: bool = True):
    docs , metadatas = _fetch_candidate_docs(chunks_store,source_filter)

    if not docs:
        return []

    # ---- Sparse: BM25 ----
    tokenized_corpus = [_tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(_tokenize(query))
    bm25_ranked_idx = sorted(range(len(docs)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]

    # ---- Dense: vector similarity via Chroma ----
    vector_results = chunks_store.similarity_search(query, k=vector_k, filter=source_filter)
    content_to_idx = {d: i for i, d in enumerate(docs)}
    vector_ranked_idx = [content_to_idx[d.page_content] for d in vector_results if d.page_content in content_to_idx]

    #---- Reciprocal Rank Fusion: combine both rankings into one score ----
    rrf_scores = defaultdict(float)
    for rank , idx in enumerate(bm25_ranked_idx):
        rrf_scores[idx] += 1.0/(rrf_k + rank + 1)
    for rank , idx in enumerate(vector_ranked_idx):
        rrf_scores[idx] += 1.0/(rrf_k + rank + 1)

    fused_idx = sorted(range(len(docs)), key=lambda i: rrf_scores[i], reverse=True)
    candidates = [(docs[i], metadatas[i]) for i in fused_idx]

    if not use_reranker or len(candidates) <= k:
        return candidates[:k]

    reranker = _get_reranker()
    pairs = [[query, content] for content, _ in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [c for c, _ in reranked[:k]]






