import pickle
import json
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter

# BM25 Scoring Function
def bm25_score(query_toks: List[str], doc_toks: List[str], doc_len: int, index_pkg: Dict,
               k1: float = 1.5, b: float = 0.75) -> float:
    """
    Compute BM25 score for a single document given a tokenized query.

    Args:
        query_toks: List of query tokens (words)
        doc_toks: List of document tokens (words)
        doc_len: Length of the document in tokens
        index_pkg: Dictionary containing BM25 meta information:
            - N: total number of documents
            - avgdl: average document length
            - df: document frequency per term
        k1, b: BM25 parameters (defaults recommended)

    Returns:
        BM25 relevance score (float)
    """
    score = 0.0

    # Check if BM25 meta exists; compute if missing
    if 'N' in index_pkg and 'avgdl' in index_pkg and 'df' in index_pkg:
        N = index_pkg['N']
        avgdl = index_pkg['avgdl']
        df = index_pkg['df']
    else:
        # Compute meta from candidate docs
        N = 1 if doc_len == 0 else 1  # single doc fallback
        avgdl = doc_len
        df = Counter(doc_toks)  # only this doc as fallback

    for term in query_toks:
        if term not in df:
            continue
        n_t = df.get(term, 0) or 1
        f_td = doc_toks.count(term)
        idf = np.log(1 + (N - n_t + 0.5) / (n_t + 0.5))
        score += idf * (f_td * (k1 + 1)) / (f_td + k1 * (1 - b + b * doc_len / avgdl))
    return score


# Main Ranking Function
def rank_documents(query_toks, candidate_docs, doc_ids, inverted_index_path=None, method="default"):
    """
    Rank candidate documents for a given tokenized query.

    Supports BM25, TF-IDF, and hybrid ranking methods.

    Args:
        query_toks: List of query tokens
        candidate_docs: List of tokenized candidate documents
        doc_ids: List of document IDs corresponding to candidate_docs
        inverted_index_path: Optional path to a pickled index package (can be dict)
        method: "bm25", "tfidf", "hybrid", or "default" ("bm25" default)

    Returns:
        ranked_doc_ids: List of doc IDs sorted by descending relevance
        ranked_scores: List of corresponding relevance scores
    """
    import gzip
    from utils.tfidf import TfidfRanker  # Import TF-IDF ranker for scoring
    import numpy as np

    # Load index package if provided as a file path
    if isinstance(inverted_index_path, str):
        with gzip.open(inverted_index_path, "rb") as f:
            index_pkg = pickle.load(f)
    elif isinstance(inverted_index_path, dict):
        index_pkg = inverted_index_path  # Use provided dictionary
    else:
        index_pkg = {}  # No index, only TF-IDF scoring

    # Compute BM25 meta data if missing and BM25 is requested
    if method in ("bm25", "hybrid"):
        if 'N' not in index_pkg or 'avgdl' not in index_pkg or 'df' not in index_pkg:
            N = len(candidate_docs)  # Total number of candidate docs
            doc_lens = [len(doc) for doc in candidate_docs]  # Length of each doc
            avgdl = sum(doc_lens) / N if N > 0 else 1.0  # Compute average doc length
            df = {}  # Document frequency dictionary
            for doc in candidate_docs:
                for term in set(doc):
                    df[term] = df.get(term, 0) + 1
            # Store computed meta in index package
            index_pkg['N'] = N
            index_pkg['avgdl'] = avgdl
            index_pkg['df'] = df

    # Initialize score list
    scores = []

    # Map "default" method to BM25
    if method == "default":
        method = "bm25"

    # BM25 Scoring
    if method == "bm25":
        for doc in candidate_docs:
            scores.append(bm25_score(query_toks, doc, len(doc), index_pkg))

    # TF-IDF Scoring
    elif method == "tfidf":
        ranker = TfidfRanker(candidate_docs)
        for doc in candidate_docs:
            scores.append(ranker.compute_score(query_toks, doc))

    # Hybrid: BM25 + top-20 TF-IDF rerank
    elif method == "hybrid":
        # Compute BM25 scores
        bm25_scores = [bm25_score(query_toks, d, len(d), index_pkg) for d in candidate_docs]
        # Identify top-20 BM25 documents
        top20_idx = np.argsort(bm25_scores)[-20:][::-1]
        scores = bm25_scores.copy()
        # Compute TF-IDF for top-20 and combine
        ranker = TfidfRanker(candidate_docs)
        for i in top20_idx:
            scores[i] += 0.3 * ranker.compute_score(query_toks, candidate_docs[i])
    else:
        raise ValueError(f"Unknown method: {method}")

    # Sort documents by descending score; tie-break using ascending doc_id
    ranked = sorted(zip(doc_ids, scores), key=lambda x: (-x[1], x[0]))
    ranked_doc_ids, ranked_scores = zip(*ranked) if ranked else ([], [])
    return list(ranked_doc_ids), list(ranked_scores)

# Pearson Correlation Function
def pearson_corr(y_pred: List[float], y_true: List[float]) -> float:
    """
    Compute Pearson correlation coefficient between predicted scores and true relevance.

    Args:
        y_pred: Predicted scores
        y_true: True relevance scores

    Returns:
        Pearson r value, or 0.0 if insufficient data or zero variance
    """
    if len(y_pred) < 2 or np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    return float(np.corrcoef(y_pred, y_true)[0, 1])

# Development Set Evaluation
def evaluate_devset(index_file: str):
    """
    Evaluate ranking performance on development set using Pearson correlation.

    Loads index package, development queries, documents, and relevance judgments.
    Computes Pearson correlation for BM25, TF-IDF, and hybrid methods.

    Args:
        index_file: Path to the pickled index package
    """
    # Load index package
    with open(index_file, "rb") as f:
        index_pkg = pickle.load(f)

    # Load tokenized queries
    with open("data/dev/queries.json") as f:
        queries = json.load(f)

    # Load documents
    id2doc = {}
    with open("data/dev/documents.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            id2doc[obj["id"]] = obj["tokens"]

    # Load relevance judgments (qrels)
    with open("data/dev/relevance_judge.json") as f:
        relevance = json.load(f)

    methods = ["bm25", "tfidf", "hybrid"]
    results = {m: [] for m in methods}

    # Evaluate each query
    for qid, qdata in queries.items():
        q_tokens = qdata["tokens"]  # Query tokens
        candidates = list(relevance[qid].keys())  # Candidate doc IDs
        candidate_docs = [id2doc[d] for d in candidates]  # Candidate token lists
        doc_ids = candidates

        for m in methods:
            # Rank documents
            ranked_ids, scores = rank_documents(q_tokens, candidate_docs, doc_ids, index_pkg, method=m)
            y_pred = np.array(scores, dtype=float)
            y_true = np.array([relevance[qid][d] for d in ranked_ids], dtype=float)
            pearson = pearson_corr(y_pred, y_true)  # Compute Pearson correlation
            results[m].append(pearson)

    # Print averaged results for all methods
    print("==== Development Set Results ====")
    for m in methods:
        avg_score = np.mean(results[m])
        print(f"{m:10s}  Pearson r = {avg_score:.3f}")

# Execute Evaluation if Run as Script
if __name__ == "__main__":
    evaluate_devset("index_pkg.pkl")  # Provide path to pickled index
