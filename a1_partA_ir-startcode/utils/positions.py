"""Token-position mapping (Task A-2)."""
from collections import defaultdict
from typing import Dict, List, Union, Tuple

def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:
    """
    Returns a dictionary mapping each unique n-gram to a list of its starting positions (0-indexed).
    Unigrams are returned as strings; n-grams (n > 1) as tuples of strings.

    Parameters:
    - tokens: List[str] -- List of tokens representing the document.
    - n: int -- Size of n-grams to index (default 1 for unigrams).

    Returns:
    - Dict[Union[str, Tuple[str, ...]], List[int]] -- Mapping from token or n-gram to list of positions.

    Raises:
    - TypeError: If tokens is not a list of strings or n is not an int.
    - ValueError: If n < 1.
    """
    # Validate that tokens is a list of strings
    if not isinstance(tokens, list) or not all(isinstance(t, str) for t in tokens):
        raise TypeError("tokens must be a list of strings")

    # Validate that n is an integer
    if not isinstance(n, int):
        raise TypeError("n must be an integer")

    # Ensure n is positive
    if n < 1:
        raise ValueError("n must be a positive integer")

    # If n is greater than total tokens, no n-grams can be formed
    if n > len(tokens):
        return {}

    # Use defaultdict to store positions of each token/ngram
    positions = defaultdict(list)

    # Slide a window of size n over tokens to extract n-grams
    for i in range(len(tokens) - n + 1):
        # Key is a single token for unigrams, tuple for n-grams
        key = tokens[i] if n == 1 else tuple(tokens[i:i + n])
        positions[key].append(i)  # Store the starting index

    # Convert defaultdict to regular dict before returning
    return dict(positions)

### Summary
"""
The make_positions function creates a mapping between each unique n-gram in a document and all the positions (indices) where it appears. It first validates that the 
input is a list of strings and that n is a positive integer. For unigrams (n=1), each token is stored as a string key; for higher-order n-grams, the keys are tuples 
of tokens. The function slides a window of size n across the token list, recording the starting position of each n-gram in a dictionary, where values are lists of 
indices. If n is greater than the total number of tokens, it returns an empty dictionary. This mapping is useful for tasks like concordance building, keyword indexing,
and information retrieval.
"""