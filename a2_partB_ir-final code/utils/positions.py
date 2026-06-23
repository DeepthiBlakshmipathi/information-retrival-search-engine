"""Token-position mapping (Task A-2)."""
from collections import defaultdict
from typing import Dict, List, Union, Tuple

def make_positions(tokens: List[str], n: int = 1) -> Dict[Union[str, Tuple[str, ...]], List[int]]:
    """
    Generate a mapping from n-grams to their positions in a token list.

    Args:
        tokens: List of tokens in a document
        n: Length of the n-grams (default 1 for unigrams)

    Returns:
        Dictionary where:
            - Keys: unique n-grams (string for unigrams, tuple for n > 1)
            - Values: list of starting indices (0-indexed) where the n-gram occurs
    """
    positions = defaultdict(list)  # Use defaultdict to accumulate positions easily

    for i in range(len(tokens) - n + 1):  # Slide window of size n
        if n == 1:
            key = tokens[i]  # For unigrams, use the token itself
        else:
            key = tuple(tokens[i:i+n])  # For n-grams, use a tuple of tokens
        positions[key].append(i)  # Append the starting index to the list

    return dict(positions)  # Convert defaultdict to regular dict for output

