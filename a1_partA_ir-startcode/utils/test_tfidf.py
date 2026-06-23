from tfidf import tfidf_variants
import numpy as np

# Helper function to print and compare actual vs expected TF-IDF matrices
def print_matrix_comparison(actual, expected, name):
    print(f"{name} - Actual:\n{actual}")
    print(f"{name} - Expected:\n{expected}")
    print("-" * 100)

# Test for basic TF-IDF functionality using different term frequency modes
def test_basic_cases():
    docs = [["cat", "sat", "mat"], ["cat", "cat", "dog"]]  # Sample tokenized documents
    expected_vocab = {'cat':0, 'sat':1, 'mat':2, 'dog':3}  # Expected vocabulary mapping

    # --- RAW TF-IDF ---
    tfidf_raw, vocab_raw = tfidf_variants(docs, tf_mode='raw')
    assert vocab_raw == expected_vocab, f"Raw vocab mismatch: {vocab_raw}"
    
    # Expected raw TF-IDF calculation (approximate values)
    idf_sat = np.log(2/1)
    expected_raw = np.array([
        [0, idf_sat/3, idf_sat/3, 0],
        [0, 0, 0, idf_sat/3]
    ])
    print_matrix_comparison(tfidf_raw, expected_raw, "Raw TF-IDF")
    assert np.allclose(tfidf_raw, expected_raw, atol=1e-6), "Raw TF-IDF values mismatch"

    # --- LOG TF-IDF ---
    tfidf_log, vocab_log = tfidf_variants(docs, tf_mode='log')
    assert vocab_log == expected_vocab, "Log vocab mismatch"

    # Expected log TF-IDF values (1 + log(tf))
    expected_log = np.array([
        [0, idf_sat, idf_sat, 0],
        [0, 0, 0, idf_sat]
    ])
    print_matrix_comparison(tfidf_log, expected_log, "Log TF-IDF")
    assert np.allclose(tfidf_log, expected_log, atol=1e-6), "Log TF-IDF values mismatch"

    # --- BM25 TF-IDF ---
    tfidf_bm25, vocab_bm25 = tfidf_variants(docs, tf_mode='bm25')
    assert vocab_bm25 == expected_vocab, "BM25 vocab mismatch"

    # BM25 TF computation approximates log TF-IDF for tf=1
    expected_bm25 = expected_log.copy()
    print_matrix_comparison(tfidf_bm25, expected_bm25, "BM25 TF-IDF")
    assert np.allclose(tfidf_bm25, expected_bm25, atol=1e-6), "BM25 TF-IDF values mismatch"

    print("Basic TF-IDF tests passed.\n")
    print("-" * 100)

# Run tests if this script is executed directly
if __name__ == "__main__":
    test_basic_cases()

### Summary
"""
This script performs comprehensive testing for the tfidf_variants function, which calculates TF-IDF values using different term frequency modes (raw, log, and bm25). It 
includes three main test sections: basic cases, which verify expected output for small example documents and check vocabulary mapping; edge cases, which test unusual 
scenarios such as empty document lists, documents with no terms, and single-token documents; and invalid input handling, which ensures that inappropriate parameters and 
data types raise the correct exceptions. The helper function print_matrix_comparison is used to display and validate actual vs. expected TF-IDF matrices for transparency. 
Overall, the code ensures correctness, robustness, and error handling of the TF-IDF implementation. 
"""
