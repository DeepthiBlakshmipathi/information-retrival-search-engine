"""Word- and char-level n-gram helpers (Task A-2)."""
from typing import List, Tuple

__all__ = ["make_ngrams_tokens", "make_ngrams_chars"]

def make_ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    """
    Generate word-level n-grams with start/end padding.

    Args:
        tokens: List of string tokens representing a document.
        n: Integer size of n-grams (e.g., 2 for bigrams, 3 for trigrams).

    Returns:
        List of n-grams as tuples of strings, padded with <s> at the start
        and </s> at the end (each repeated n-1 times).
    
    Raises:
        ValueError: If n is not a positive integer.
    """
    # Validate that n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Create start and end padding tokens, repeated (n-1) times
    start_pad = ["<s>"] * (n - 1)
    end_pad = ["</s>"] * (n - 1)

    # Add padding to the token list
    padded_tokens = start_pad + tokens + end_pad

    ngrams = []
    # Generate n-grams by sliding a window of size n across the padded list
    for i in range(len(padded_tokens) - n + 1):
        ngram = tuple(padded_tokens[i:i + n])  # Convert each n-gram to a tuple 
        ngrams.append(ngram)
    return ngrams


def make_ngrams_chars(text: str, n: int) -> List[str]:
    """
    Generate character-level n-grams from a string with word boundary markers.

    Args:
        text: Input string from which to create character n-grams.
        n: Integer size of character n-grams.

    Returns:
        List of character n-gram strings with '$' padding at word boundaries.

    Raises:
        ValueError: If n is not a positive integer.
    """
    # Validate that n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    # Ensure input is a string
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    # Add boundary markers '$' at start and end after stripping whitespace
    padded_text = "$" + text.strip() + "$"

    ngrams = []
    # Generate substrings of length n by sliding across the padded string
    for i in range(len(padded_text) - n + 1):
        ngram = padded_text[i:i + n]
        ngrams.append(ngram)
    return ngrams

### Summary
"""
This code defines two helper functions for generating n-grams. The make_ngrams_tokens function creates word-level n-grams from a list of tokens, adding <s> padding at
the start and </s> padding at the end to handle sentence boundaries. The make_ngrams_chars function creates character-level n-grams from a given text string, adding $ 
markers to indicate word boundaries. Both functions validate that n is a positive integer, and make_ngrams_chars also ensures the input is a string. These utilities are 
useful for feature extraction in natural language processing tasks such as text classification, language modeling, and search indexing.
"""