"""Robust HTML→tokens cleaning pipeline (Task A-1)."""
import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize

__all__ = ["preprocess"]

def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Convert noisy HTML documents into token lists.
    raw_html_list: List[str] a list of raw html documents, see data/dev/documents.jsonl and queries.json to see possible noisy html documents and inputs
    return: List[List[str]] a list of token lists, each token list is a list of tokens
    Hints:
    - strips tags & entities
    - keeps 1 punctuation
    - no stemming/stop-word removal (Part A spec)
    """
    # List to hold tokens for all documents
    all_tokens = [] 

    for raw_html in raw_html_list:
        try:
            # Skip invalid inputs: None or empty strings
            if not isinstance(raw_html, str) or raw_html.strip() == "":
                all_tokens.append([]) # Append empty token list for robustness
                continue # continue inside try block after appending empty list
            
            # 1. Strip HTML tags to extract visible text only
            soup = BeautifulSoup(raw_html, "html.parser")
            text = soup.get_text(separator=" ") # Use space to separate elements

            # 2. Decode HTML entities like &amp;, &lt; into normal characters
            text = html.unescape(text)

            # 3. Normalize Unicode characters to NFKC form
            #    This helps unify characters that look the same but have different code points
            text = unicodedata.normalize("NFKC", text)

            # 4. Tokenize the cleaned text into words and punctuation
            #    Using nltk's word_tokenize which preserves case and punctuation
            tokens = word_tokenize(text)
            
            # Append token list for this document
            all_tokens.append(tokens)

        except Exception as e:
            # If any error occurs, append empty list to keep the function robust
            all_tokens.append([])

    return all_tokens


### Summary

"""
This preprocess function takes a list of raw HTML documents and converts each document into a list of tokens for further text processing. It robustly handles noisy 
HTML by first stripping away all HTML tags to extract visible text, then decoding HTML entities (such as &amp; or &lt;) into their corresponding characters. The text 
is normalized using Unicode normalization (NFKC) to ensure consistent representation of characters. After that, it tokenizes the cleaned text using NLTK’s word_tokenize, 
which preserves case and punctuation without performing stemming or stop-word removal, as specified. The function is designed to be fault-tolerant: it safely skips empty 
or invalid inputs and returns an empty token list if any error occurs during processing, ensuring that the pipeline continues smoothly without interruption.
"""
