from positions import make_positions

def test_make_positions():
    # Test dataset for position mapping
    tokens = [" the", " quick ", " brown ", "fox ", " jumps ", " over ", "the ", "dog "]
    
    # ---------- Test Case 1: Unigrams ----------
    # Generate position mapping for unigrams (n=1)
    result_unigram = make_positions(tokens)
    # Expected output: each token mapped to its starting index in the list
    expected_unigram = {
        " the": [0],
        " quick ": [1],
        " brown ": [2],
        "fox ": [3],
        " jumps ": [4],
        " over ": [5],
        "the ": [6],
        "dog ": [7],
    }
    print("Unigram result:", result_unigram)
    print("Unigram expected:", expected_unigram)
    print("-" * 50)  # Separator line for readability
    assert result_unigram == expected_unigram, f"Unigram test failed: {result_unigram}"

    # ---------- Test Case 2: Bigrams ----------
    # Generate position mapping for bigrams (n=2)
    result_bigram = make_positions(tokens, n=2)
    # Expected output: each consecutive 2-token tuple mapped to its starting index
    expected_bigram = {
        (" the", " quick "): [0],
        (" quick ", " brown "): [1],
        (" brown ", "fox "): [2],
        ("fox ", " jumps "): [3],
        (" jumps ", " over "): [4],
        (" over ", "the "): [5],
        ("the ", "dog "): [6],
    }
    print("Bigram result:", result_bigram)
    print("Bigram expected:", expected_bigram)
    print("-" * 50)
    assert result_bigram == expected_bigram, f"Bigram test failed: {result_bigram}"

    # ---------- Edge Case 1: Empty token list ----------
    result_empty = make_positions([], n=1)
    assert result_empty == {}, f"Empty tokens test failed: {result_empty}"

    # ---------- Edge Case 2: n larger than token list length ----------
    result_large_n = make_positions(["only", "two"], n=3)
    assert result_large_n == {}, f"Large n test failed: {result_large_n}"

    # ---------- Edge Case 3: Single token ----------
    result_single = make_positions(["single"], n=1)
    expected_single = {"single": [0]}
    assert result_single == expected_single, f"Single token test failed: {result_single}"

    print("All position tests passed!")

if __name__ == "__main__":
    test_make_positions()


### Summary

"""
This script tests the make_positions function from positions.py to ensure it correctly maps tokens or n-grams to their starting positions in a given list of tokens. It 
verifies unigrams and bigrams against expected outputs, handles special cases like an empty token list, n larger than the number of tokens, and a single-token input. 
Clear print statements and assertion checks confirm that the function behaves as intended for both normal and edge scenarios, ensuring robust and predictable performance.
"""