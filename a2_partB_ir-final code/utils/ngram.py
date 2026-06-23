"""
Word- and char-level n-gram helpers (Task A-2)
"""
from typing import List, Tuple

# Specify what will be exported when using 'from ngram import *'
__all__ = ["make_ngrams_tokens", "make_ngrams_chars"]

def make_ngrams_tokens(tokens: List[str], n: int) -> List[Tuple[Tuple[str, ...], List[int]]]:
    """
    Generate token-level n-grams with their positions in the document.
    
    Args:
        tokens: List of tokens in a document
        n: Length of the n-grams
    
    Returns:
        List of tuples: (ngram_tuple, positions_list)
        positions_list contains starting indices of the n-gram in the tokens list
    """
    ngrams_with_pos = []
    for i in range(len(tokens) - n + 1):  # Slide window of size n over tokens
        ngram = tuple(tokens[i:i+n])  # Create n-gram tuple
        ngrams_with_pos.append((ngram, [i]))  # Store n-gram and its starting position
    return ngrams_with_pos

def make_ngrams_chars(token: str, max_len: int) -> List[str]:
    """
    Generate character-level n-grams for a single token with '$' as boundaries.
    
    Args:
        token: Single token string
        max_len: Maximum length of n-grams to generate
    
    Returns:
        List of character n-grams (excluding lone '$')
    """
    text = f"${token}$"  # Add boundary markers to capture start/end n-grams
    ngrams = []
    L = len(text)
    for n in range(2, max_len + 1):  # Start from n=2 to skip single '$'
        for i in range(L - n + 1):  # Slide window of size n
            gram = text[i:i+n]  # Extract substring of length n
            if gram != "$":  # Skip isolated boundary symbols
                ngrams.append(gram)
    return ngrams
