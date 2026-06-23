from ngram import make_ngrams_tokens, make_ngrams_chars

# Sample tokens for testing make_ngrams_tokens function
tokens = [" the", " quick ", " brown "]
print("make_ngrams_tokens output:", make_ngrams_tokens(tokens, 2))
print("-" * 40)

# Sample text for testing make_ngrams_chars function
text = " hello "
print("make_ngrams_chars output:", make_ngrams_chars(text, 3))
print("-" * 40)

def test_make_ngrams_tokens():
    """
    Test the make_ngrams_tokens function for:
    - Correct bigram and trigram generation with <s> and </s> markers
    - Handling invalid n values (0, negative)
    """
    # Test with bigrams (n=2)
    result = make_ngrams_tokens(tokens, 2)
    expected = [
        ("<s>", " the"), 
        (" the", " quick "), 
        (" quick ", " brown "), 
        (" brown ", "</s>")
    ]
    assert result == expected, f"Expected {expected}, got {result}"

    # Test with trigrams (n=3)
    result_tri = make_ngrams_tokens(tokens, 3)
    expected_tri = [
        ("<s>", "<s>", " the"),
        ("<s>", " the", " quick "),
        (" the", " quick ", " brown "),
        (" quick ", " brown ", "</s>"),
        (" brown ", "</s>", "</s>")
    ]
    assert result_tri == expected_tri, f"Expected {expected_tri}, got {result_tri}"

    # Test invalid n values
    try:
        make_ngrams_tokens(tokens, 0)
    except ValueError as e:
        assert str(e) == "n must be a positive integer"
    else:
        assert False, "Expected ValueError for n=0"

    try:
        make_ngrams_tokens(tokens, -1)
    except ValueError as e:
        assert str(e) == "n must be a positive integer"
    else:
        assert False, "Expected ValueError for n=-1"

def test_make_ngrams_chars():
    """
    Test the make_ngrams_chars function for:
    - Correct generation of character n-grams with start/end markers ($)
    - Different n values
    - Edge cases where n is larger than text length
    - Handling invalid n values and non-string input
    """
    # Test with trigrams (n=3)
    result = make_ngrams_chars(text, 3)
    expected = ['$he', 'hel', 'ell', 'llo', 'lo$']
    assert result == expected, f"Expected {expected}, got {result}"

    # Test with bigrams (n=2)
    result_2 = make_ngrams_chars(text, 2)
    expected_2 = ['$h', 'he', 'el', 'll', 'lo', 'o$']
    assert result_2 == expected_2, f"Expected {expected_2}, got {result_2}"

    # Edge case: n larger than string length
    short_text = "hi"
    result_short = make_ngrams_chars(short_text, 5)
    expected_short = []  # No n-grams possible
    assert result_short == expected_short, f"Expected {expected_short}, got {result_short}"

    # Test invalid n values
    try:
        make_ngrams_chars(text, 0)
    except ValueError as e:
        assert str(e) == "n must be a positive integer"
    else:
        assert False, "Expected ValueError for n=0"

    try:
        make_ngrams_chars(text, -2)
    except ValueError as e:
        assert str(e) == "n must be a positive integer"
    else:
        assert False, "Expected ValueError for n=-2"

    # Test invalid text type
    try:
        make_ngrams_chars(None, 3)
    except ValueError as e:
        assert str(e) == "text must be a string"
    else:
        assert False, "Expected ValueError for text=None"

if __name__ == "__main__":
    test_make_ngrams_tokens()
    test_make_ngrams_chars()
    print("All tests passed!")


### Summary
"""
This script tests two functions—make_ngrams_tokens and make_ngrams_chars—that generate token-based and character-based n-grams, respectively. For make_ngrams_tokens,
it verifies correct bigram and trigram creation, proper insertion of start (<s>) and end (</s>) markers, and error handling for invalid n values. For make_ngrams_chars,
it checks correct formation of character n-grams with start/end markers ($), handles different n sizes, manages edge cases when n exceeds text length, and ensures proper 
exceptions are raised for invalid inputs or non-string values. If all assertions pass, the script prints “All tests passed!.
"""