"""
tfidf.py - Simple TF–IDF scoring module for IR assignment (Task 3)
Computes cosine similarity between a query and a document using TF–IDF.
"""

import math
from collections import Counter
from typing import List, Dict

# TF-IDF Ranker Class
class TfidfRanker:
    def __init__(self, docs: List[List[str]]):
        """
        Initialize the TF–IDF ranker.

        Args:
            docs: List of tokenized documents
        """
        self.docs = docs
        self.N = len(docs)  # total number of documents
        self.df = self._compute_df(docs)  # document frequency for each term
        # Smoothed IDF formula: log((N+1)/(df+1)) + 1
        self.idf = {term: math.log((self.N + 1) / (df + 1)) + 1 for term, df in self.df.items()}

    # Document frequency computation
    def _compute_df(self, docs: List[List[str]]) -> Dict[str, int]:
        """
        Compute document frequency (DF) for each term in the corpus.

        Args:
            docs: list of tokenized documents
        Returns:
            Dictionary: term -> number of documents containing term
        """
        df = {}
        for doc in docs:
            for term in set(doc):  # only count once per document
                df[term] = df.get(term, 0) + 1
        return df

    # Cosine similarity scoring
    def compute_score(self, query: List[str], doc: List[str]) -> float:
        """
        Compute cosine similarity between query and document using TF-IDF vectors.

        Args:
            query: tokenized query
            doc: tokenized document
        Returns:
            float score: TF-IDF cosine similarity
        """
        tf_doc = Counter(doc)       # term frequencies in document
        tf_query = Counter(query)   # term frequencies in query

        # Build TF-IDF weighted vectors for document
        doc_vec = {}
        for term, freq in tf_doc.items():
            if term in self.idf:
                doc_vec[term] = freq * self.idf[term]  # TF * IDF

        # Build TF-IDF weighted vectors for query
        query_vec = {}
        for term, freq in tf_query.items():
            if term in self.idf:  # ignore out-of-vocabulary terms
                query_vec[term] = freq * self.idf[term]

        # Compute dot product between query and document
        dot = sum(doc_vec.get(t, 0.0) * w for t, w in query_vec.items())
        # Compute L2 norms
        norm_doc = math.sqrt(sum(w * w for w in doc_vec.values()))
        norm_query = math.sqrt(sum(w * w for w in query_vec.values()))

        # Avoid division by zero
        if norm_doc == 0 or norm_query == 0:
            return 0.0
        # Cosine similarity formula
        return dot / (norm_doc + 1e-9) / (norm_query + 1e-9)
