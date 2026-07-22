import json
import os
from hybrid_retrieval import hybrid_retrieval

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

K =5
QA_FILE  = os.path.join(SCRIPT_DIR, "eval_qa_pairs.json")

#ADD QA_PAIRS
QA_PAIRS = [
    # {
    #     "question": "What is the BLEU score reported for the base model?",
    #     "source": "rag_experiment_document.pdf",
    #     "expected_pages": [4],
    # },
]


def load_qa_pairs() ->list[dict]:
    if os.path.exists(QA_FILE):
        with open(QA_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return QA_PAIRS

def reciprocal_rank(retrieved_pages: list[int] , expected_pages : set[int],k:int) -> float:
    for rank, page in enumerate(retrieved_pages[:k], start=1):
        if page in expected_pages:
            return 1.0 / rank
    return 0.0

def hit_at_k(retrieved_pages: list[int], expected_pages: set[int], k: int) -> bool:
    return any(page in expected_pages for page in retrieved_pages[:k])

def run_hybrid(chunks_store, question: str, source: str, k: int) -> list[int]:

    results = hybrid_retrieval(chunks_store, question, source_filter={"source": source}, k=k)
    return [meta.get("page") for content, meta in results]


def run_vector_only(chunks_store, question: str, source: str, k: int) -> list[int]:
    docs = chunks_store.similarity_search(question, k=k, filter={"source": source})
    return [d.metadata.get("page") for d in docs]

def main():
    qa_pairs = load_qa_pairs()
    if not qa_pairs:
        print("No Q&A pairs found. Fill in QA_PAIRS in this script, or create "
              f"'{QA_FILE}' (see eval_qa_pairs_template.json for the format), then re-run.")
        return

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    chunks_store = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )
    hybrid_rrs, hybrid_hits = [], []
    vector_rrs, vector_hits = [], []
    print(f"Evaluating {len(qa_pairs)} questions at k={K}...\n")
    print(f"{'Question':<55} {'Hybrid rank':<12} {'Vector rank':<12}")
    print("-" * 80)

    for qa in qa_pairs:
        question = qa["question"]
        source = qa["source"]
        expected = set(qa["expected_pages"])

        hybrid_pages = run_hybrid(chunks_store, question, source, K)
        vector_pages = run_vector_only(chunks_store, question, source, K)

        h_rr = reciprocal_rank(hybrid_pages, expected, K)
        v_rr = reciprocal_rank(vector_pages, expected, K)
        hybrid_rrs.append(h_rr)
        vector_rrs.append(v_rr)
        hybrid_hits.append(hit_at_k(hybrid_pages, expected, K))
        vector_hits.append(hit_at_k(vector_pages, expected, K))

        h_rank_display = f"1/{int(round(1 / h_rr))}" if h_rr > 0 else "not found"
        v_rank_display = f"1/{int(round(1 / v_rr))}" if v_rr > 0 else "not found"
        short_q = (question[:52] + "...") if len(question) > 55 else question
        print(f"{short_q:<55} {h_rank_display:<12} {v_rank_display:<12}")

    hybrid_mrr = sum(hybrid_rrs) / len(hybrid_rrs)
    vector_mrr = sum(vector_rrs) / len(vector_rrs)
    hybrid_hit_rate = sum(hybrid_hits) / len(hybrid_hits)
    vector_hit_rate = sum(vector_hits) / len(vector_hits)

    print("\n" + "=" * 50)
    print(f"{'Metric':<20} {'Hybrid':<15} {'Vector-only':<15}")
    print("-" * 50)
    print(f"{'MRR@' + str(K):<20} {hybrid_mrr:<15.3f} {vector_mrr:<15.3f}")
    print(f"{'Hit-rate@' + str(K):<20} {hybrid_hit_rate:<15.1%} {vector_hit_rate:<15.1%}")
    print("=" * 50)

    if hybrid_mrr > vector_mrr:
        improvement = ((hybrid_mrr - vector_mrr) / vector_mrr * 100) if vector_mrr > 0 else float('inf')
        print(f"\nHybrid retrieval improved MRR@{K} by {improvement:.0f}% over vector-only search.")
    elif hybrid_mrr < vector_mrr:
        print(f"\nVector-only actually scored higher on this Q&A set — "
              f"worth looking at which questions hybrid did worse on, and why.")
    else:
        print(f"\nBoth methods scored identically on this Q&A set.")


if __name__ == "__main__":
    main()





