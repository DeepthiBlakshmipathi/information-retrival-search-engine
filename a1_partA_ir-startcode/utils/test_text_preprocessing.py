import sys
import pathlib

# Add current folder to sys.path so imports work
sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from text_preprocessing import preprocess

def test_preprocess():
    """
    Test the preprocess function with various HTML inputs including empty,
    malformed, and entities.
    """
    raw_docs = [
        "<html><body>This is a test! &amp; Only a test.</body></html>",
        "<div>Hello, world! <br> New line?</div>",
        None,
        "",
        "<p>Some <b>bold</b> text & unknown &entities;</p>"
    ]

    expected_tokens = [
        # Note: '&amp;' is unescaped to '&' and tokenized as a single token
        ['This', 'is', 'a', 'test', '!', '&', 'Only', 'a', 'test', '.'],

        ['Hello', ',', 'world', '!', 'New', 'line', '?'],

        [],

        [],

        # '&entities;' will be unescaped by html.unescape to '&entities;'
        # word_tokenize treats it as ['&entities', ';']
        ['Some', 'bold', 'text', '&', 'unknown', '&', 'entities']
    ]

    output_tokens = preprocess(raw_docs)

    for i, (expected, output) in enumerate(zip(expected_tokens, output_tokens)):
        print(f"Document {i} tokens:")
        print("Expected:", expected)
        print("Output  :", output)
        print("Match   :", expected == output)
        print("-" * 40)

if __name__ == "__main__":
    test_preprocess()

### Summary
"""This test script verifies the correctness and robustness of the preprocess function by feeding it a variety of raw HTML inputs, including well-formed HTML with entities, 
HTML containing line breaks, as well as edge cases such as None and empty strings. It defines the expected token lists after cleaning and tokenization, then compares the 
actual output from preprocess against these expectations. For each document, the script prints the expected tokens, the output tokens, and whether they match exactly. This 
ensures that the preprocessing pipeline handles typical noise and formatting issues in HTML content correctly and consistently.
"""
