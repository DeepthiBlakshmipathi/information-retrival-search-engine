from typing import Set
from index.access import get_posting_list, _load_package
import re

def process_boolean_query(query: str, index_path: str) -> Set[int]:
    """
    Process Boolean queries with AND/OR/NOT, parentheses, and quoted phrases.
    Returns a set of document IDs that match the query.

    Args:
        query (str): The Boolean query string.
        index_path (str): Path to the index package (unified index).

    Returns:
        Set[int]: Set of document IDs satisfying the Boolean query.
    """
    
    # Handle empty queries
    if query is None or query.strip() == "":
        # An empty string is treated as a boolean query with no results
        return set()

    # Load index and construct universe of all documents
    package = _load_package(index_path)          # Load index package from disk
    unified_index = package.get('unified', {})   # Get the unified inverted index
    universe = set()                             # Universe contains all doc IDs
    for postings in unified_index.values():
        universe.update(postings)                # Add all docs from all terms

    # Extract quoted phrases and replace with temporary keys
    phrase_map = {}  # Mapping of temporary keys -> tuple of tokens
    def replace_phrase(match):
        key = f"__PHRASE_{len(phrase_map)}__"  # Create unique key
        phrase_map[key] = tuple(match.group(1).split())  # Store phrase tokens
        return key

    # Replace all quoted phrases in query with temporary keys
    query_safe = re.sub(r'"([^"]+)"', replace_phrase, query)

    # Tokenize the query into operators, parentheses, and terms/phrases
    tokens = re.findall(r'\(|\)|AND|OR|NOT|__PHRASE_\d+__|\S+', query_safe)

    # Define operator precedence for Boolean evaluation
    precedence = {'NOT': 3, 'AND': 2, 'OR': 1}

    # Convert infix query to postfix (Reverse Polish Notation)
    # Using Shunting Yard algorithm
    output = []  # Postfix output
    ops = []     # Operator stack

    for token in tokens:
        if token in ("AND", "OR", "NOT"):
            # Pop operators of higher or equal precedence
            while ops and ops[-1] != '(' and precedence.get(ops[-1], 0) >= precedence[token]:
                output.append(ops.pop())
            ops.append(token)
        elif token == '(':
            ops.append(token)  # Push '(' onto stack
        elif token == ')':
            # Pop until matching '('
            while ops and ops[-1] != '(':
                output.append(ops.pop())
            if ops and ops[-1] == '(':
                ops.pop()  # Remove '('
        else:
            output.append(token)  # Terms or phrases go directly to output

    # Pop any remaining operators
    while ops:
        output.append(ops.pop())

    # Evaluate the postfix (RPN) expression
    stack = []

    # Helper function to get posting list for a token or phrase
    def get_posting(tok):
        if tok.startswith("__PHRASE_"):
            # Lookup phrase in index using tuple of tokens
            return set(get_posting_list(phrase_map[tok], index_path))
        else:
            # Lookup single term in index
            return set(get_posting_list((tok,), index_path))

    for tok in output:
        if tok == "NOT":
            # Unary NOT: complement relative to universe
            a = stack.pop() if stack else set()
            stack.append(universe - a)
        elif tok == "AND":
            # Binary AND: intersection of top two sets
            b = stack.pop() if stack else set()
            a = stack.pop() if stack else set()
            stack.append(a & b)
        elif tok == "OR":
            # Binary OR: union of top two sets
            b = stack.pop() if stack else set()
            a = stack.pop() if stack else set()
            stack.append(a | b)
        else:
            # Term or phrase: push its posting set onto stack
            stack.append(get_posting(tok))

    # Final result is the top of the stack
    return stack[0] if stack else set()
