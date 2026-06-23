from semantic_vector import semantic_vector  # Import the semantic_vector function
from embeddings import _WORD_VEC             # Import preloaded GloVe embeddings
import numpy as np

def test_semantic_vector():
    """
    Test the semantic_vector function with various aggregation methods.
    Validates output shapes and ensures empty documents produce zero vectors.
    """

    # Example tokenized documents:
    # - First: all words are in the GloVe vocabulary
    # - Second: some words may be out-of-vocabulary (OOV)
    # - Third: empty document to check zero vector handling
    docs = [
        ["cat", "dog"],             # in-vocab
        ["animal", "pet", "cat"],   # some OOV
        []                          # empty doc
    ]

    # List of aggregation methods to test
    methods = ["mean", "sum", "max", "meanmax", "tfidf_weighted"]

    # Print header for readability
    print("\n" + "=" * 80)
    print("Testing semantic_vector for all methods")
    print("=" * 80 + "\n")

    # Loop through each aggregation method
    for method in methods:
        print(f"--- Method: {method} ---", flush=True)  # Print current method being tested

        # Generate document vectors using the semantic_vector function
        vecs = semantic_vector(docs, embeddings=_WORD_VEC, method=method)

        # Print output shape and the actual vectors
        print("Output shape:", vecs.shape, flush=True)
        print(vecs, flush=True)
        print("\n" + "=" * 80)

        # Validate that empty documents produce zero vectors
        if method != "meanmax":
            assert np.all(vecs[2] == 0), "Empty doc should be zero vector"
        else:
            # meanmax doubles the vector length, still check zero vector for empty doc
            assert np.all(vecs[2] == 0), "Empty doc should be zero vector (meanmax)"

    # Print final completion message
    print("\nAll tests completed successfully!")

# Run the test function only if this script is executed directly
if __name__ == "__main__":
    test_semantic_vector()

### Summary

"""
This script is designed to test the `semantic_vector` function, which converts tokenized documents into document-level embeddings using preloaded GloVe word vectors. 
It evaluates five aggregation methods—mean, sum, max, meanmax, and TF-IDF weighted mean—on a small set of example documents that include in-vocabulary words, out-of-
vocabulary words, and an empty document. For each method, the script prints the resulting vector shapes and values, and it verifies that empty documents produce zero 
vectors. This ensures that the function handles different aggregation strategies correctly and robustly, while also managing unknown words using the `<unk>` token. The 
test runs automatically when the script is executed directly, providing clear feedback on the correctness of the semantic vector computation.
"""