"""Robust HTML→tokens cleaning pipeline (Task A-1)."""
import re, html, unicodedata
from typing import List
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def normalize_tokens(tokens: List[str]) -> List[str]:
    """
    Lowercase + strip punctuation. Removes empty tokens.
    """
    return [re.sub(r'\W+', '', t.lower()) for t in tokens if t.strip()]

def preprocess(raw_html_list: List[str]) -> List[List[str]]:
    """Convert noisy HTML documents into token lists.

    Steps:
    - Remove HTML tags & entities
    - Normalize Unicode
    - Tokenize text
    - Normalize tokens (via normalize_tokens)
    """
    docs = []
    for raw_html in raw_html_list:
        # 1. Strip HTML
        text = BeautifulSoup(raw_html, "html.parser").get_text()
        # 2. Decode HTML entities
        text = html.unescape(text)
        # 3. Normalize Unicode
        text = unicodedata.normalize("NFKC", text)
        # 4. Tokenize
        tokens = word_tokenize(text)
        # 5. Clean tokens
        tokens = normalize_tokens(tokens)
        docs.append(tokens)

    return docs

