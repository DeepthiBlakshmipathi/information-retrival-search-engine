from typing import List, Dict
import numpy as np

def semantic_vector(docs: List[List[str]],
                    embeddings: Dict[str, np.ndarray] = None,
                    method: str = 'mean') -> np.ndarray:
    """
    Aggregate word embeddings into document vectors.
    
    Parameters:
        docs (List[List[str]]): List of tokenized documents (list of words).
        embeddings (Dict[str, np.ndarray], optional): Preloaded word embeddings (word → vector).
        method (str): Aggregation method to compute document vector. Options:
            - 'mean': average of all word vectors
            - 'sum': sum of all word vectors
            - 'max': max pooling over all word vectors
            - 'tfidf_weighted': TF-IDF weighted average of word vectors
            - 'meanmax': concatenation of mean and max pooled vectors

    Returns:
        np.ndarray: Document vectors, shape [num_docs, embedding_dim] or
                    [num_docs, 2 * embedding_dim] for 'meanmax'.
    """
    
    # Load default embeddings if not provided
    if embeddings is None:
        from embeddings import _WORD_VEC
        embeddings = _WORD_VEC

    # Validate input type: must be list of lists
    if not isinstance(docs, list) or not all(isinstance(doc, list) for doc in docs):
        raise TypeError("docs must be a list of list of strings.")

    # Check if the chosen aggregation method is valid
    valid_methods = {"mean", "sum", "max", "tfidf_weighted", "meanmax"}
    if method not in valid_methods:
        raise ValueError(f"Unsupported method {method}")

    # Ensure '<unk>' token exists in embeddings for OOV words
    if "<unk>" not in embeddings:
        raise KeyError("<unk> token required in embeddings")

    # Determine embedding dimension from '<unk>' vector
    embedding_dim = embeddings['<unk>'].shape[0]
    num_docs = len(docs)

    # Pre-allocate array to store document vectors
    # For 'meanmax', we double the dimension to store both mean and max
    if method == "meanmax":
        doc_vectors = np.zeros((num_docs, embedding_dim * 2), dtype=np.float32)
    else:
        doc_vectors = np.zeros((num_docs, embedding_dim), dtype=np.float32)

    # Prepare TF-IDF weighting if requested
    if method == "tfidf_weighted":
        from sklearn.feature_extraction.text import TfidfVectorizer
        # Convert each document into a space-separated string
        docs_strings = [" ".join([w if w in embeddings else "<unk>" for w in doc]) for doc in docs]
        # Fit TF-IDF vectorizer on all documents
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_strings)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        vocab = {w: i for i, w in enumerate(feature_names)}  # map words to column indices

    # Process each document individually
    for i, doc in enumerate(docs):
        if len(doc) == 0:
            continue  # leave zero vector for empty document

        # Get word embeddings for all tokens, use '<unk>' for OOV words
        token_vecs = np.array([embeddings.get(w, embeddings['<unk>']) for w in doc])

        # Aggregate word vectors according to selected method
        if method == "mean":
            doc_vectors[i] = token_vecs.mean(axis=0)  # average vector
        elif method == "sum":
            doc_vectors[i] = token_vecs.sum(axis=0)   # sum of vectors
        elif method == "max":
            doc_vectors[i] = token_vecs.max(axis=0)   # max pooling
        elif method == "tfidf_weighted":
            # Compute TF-IDF weighted vector
            weights = np.array([tfidf_matrix[i, vocab[w]] if w in vocab else 0.0 for w in doc])
            total_weight = weights.sum()
            if total_weight > 0:
                doc_vectors[i] = (weights[:, None] * token_vecs).sum(axis=0) / total_weight
            else:
                # fallback to mean if all weights are zero
                doc_vectors[i] = token_vecs.mean(axis=0)
        elif method == "meanmax":
            # Concatenate mean and max pooled vectors
            mean_vec = token_vecs.mean(axis=0)
            max_vec = token_vecs.max(axis=0)
            doc_vectors[i] = np.concatenate([mean_vec, max_vec])

    return doc_vectors



### Summary


"""
This code defines the `semantic_vector` function, which converts a list of tokenized documents into numerical document vectors by aggregating preloaded word embeddings. 
It supports multiple aggregation methods, including simple averages (`mean`), sums (`sum`), maximum pooling (`max`), TF-IDF weighted averages (`tfidf_weighted`), and 
concatenation of mean and max vectors (`meanmax`). The function handles out-of-vocabulary words by using a special `<unk>` embedding and ensures that empty documents 
result in zero vectors. For TF-IDF weighting, it transforms each document into a string and computes word importance before combining the word vectors. The output is a 
NumPy array of document vectors with dimensions dependent on the chosen aggregation method, making this function suitable for generating dense vector representations of 
textual data for further NLP or machine learning tasks.

"""
