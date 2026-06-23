#!/usr/bin/env python3
"""
Interactive Verification Script for IR System
Select tasks to run: 
1 = Task 1, 2 = Task 2, 3 = Task 3, 4 = Task 4, 5 = All, 6 = Exit
"""

import os, sys, subprocess

# Add project directories to the Python path so imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "utils"))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ranking"))

# ---------------------------
# Helper function for printing headers
# ---------------------------
def print_header(title):
    print("\n" + "="*100)
    print(f"{title}")
    print("="*100 + "\n")

# ---------------------------
# Task 1: Index Verification
# ---------------------------
def run_task1():
    """Runs verification tests for the index: posting lists, wildcards, proximity, and meta data"""
    print_header("TASK 1: Index Test Results")

    from index.builders import create_all_indexes
    from index.access import get_posting_list, find_wildcard_matches, get_term_positions, _load_package

    # Minimal tokenized docs for testing
    tokenized_docs = [
        ["climate", "change", "affects", "global", "policy"],   # doc_id 191
        ["machine", "learning", "algorithms", "improve"],       # doc_id 279
        ["climate", "science", "research", "methods"],          # doc_id 305
    ]
    doc_ids = [191, 279, 305]
    index_file = "index_pkg_test.pkl"

    # Build the index
    create_all_indexes(tokenized_docs, index_file, doc_ids=doc_ids)

    # Helper to print lists nicely
    def format_list(lst):
        return ", ".join(map(str, lst)) if lst else "(none)"

    # Print basic outputs
    print(f"Posting list for ('climate','change'): {format_list(get_posting_list(('climate','change'), index_file))}")
    print(f"Wildcard matches for '$cl': {format_list(find_wildcard_matches('$cl', index_file))}")
    print(f"Positions of 'climate' in doc 191: {format_list(get_term_positions(('climate',), 191, index_file))}")

    # Verification checks
    results = []
    posting_climate = get_posting_list(("climate",), index_file)
    posting_climate_change = get_posting_list(("climate","change"), index_file)
    results.append(("Unified index 'climate'", posting_climate == [191,305]))
    results.append(("Unified index ('climate','change')", posting_climate_change == [191]))
    results.append(("Wildcard $cl", find_wildcard_matches("$cl", index_file) == ["climate"]))
    results.append(("Proximity positions of 'climate' in doc 191", get_term_positions(("climate",),191,index_file) == [0]))

    # Check meta information
    pkg = _load_package(index_file)
    meta = pkg.get("__META__", {})
    results.append(("Meta N", meta.get("N") == 3))
    results.append(("Meta doc_lengths", meta.get("doc_lengths") == {191:5,279:4,305:4}))
    results.append(("Meta avgdl", abs(meta.get("avgdl")-(5+4+4)/3) < 1e-6))

    # Print verification results
    print("\n========= Task 1 Verification Results =========")
    for name, passed in results:
        print(f"{name:<45} {'PASS' if passed else 'FAIL'}")
    print("Task 1 Verification Complete\n")

# ---------------------------
# Task 2: Query Processing Verification
# ---------------------------
def run_task2():
    """Runs tests for query type detection, natural language conversion, and query results"""
    print_header("TASK 2: Query Processing Tests")

    from query_processing.query_process import process_query, detect_query_type, convert_natural_language

    # Test query type detection
    detection_tests = {
        "climate AND change": "boolean",
        '"machine learning"': "boolean",
        "climat*": "wildcard",
        "climate NEAR/3 change": "proximity",
        "effects climate change": "natural_language",
    }
    print("Query Type Detection Tests:")
    for query, expected in detection_tests.items():
        result = detect_query_type(query)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"{query:<40} -> {expected:<15} {status}")

    # Test natural language conversion
    nl_tests = {
        "effects climate change": "effects OR climate OR change",
        "machine learning algorithms work": "machine OR learning OR algorithms OR work",
    }
    print("\nNatural Language Conversion Tests:")
    for query, expected in nl_tests.items():
        result = convert_natural_language(query)
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"{query:<40} -> {expected:<20} {status}")

    # Test query processing output
    query_tests = {
        'climate AND change': {191},
        '"climate change"': {191},
        'climate OR machine': {191, 279, 305},
        'climate AND NOT machine': {191, 305},
        'climat*': {191,305},
        'climate NEAR/3 change': {191},
        '"machine learning" NEAR/1 algorithms': {279},
        'effects climate change': {191,305},
    }
    print("\nQuery Processing Tests:")
    for query, expected in query_tests.items():
        result = process_query(query, "index_pkg_test.pkl")
        status = "PASS" if result == expected else f"FAIL (got {result})"
        print(f"{query:<40} -> {str(expected):<20} {status}")

    print("Task 2 Verification Complete\n")


# ---------------------------
# Task 3: Ranking Verification
# ---------------------------
def run_task3():
    """Tests ranking methods (tfidf, bm25, hybrid) and correlation with relevance scores"""
    print_header("===========TASK 3: Ranking Tests================")

    from ranking.rankers import rank_documents, pearson_corr

    # Minimal example documents
    id2doc = {
        1: ["climate", "change", "effects"],
        2: ["global", "warming", "climate"],
        3: ["machine", "learning", "algorithms"],
        4: ["climate", "policy", "change"],
    }
    doc_ids = list(id2doc.keys())
    candidate_docs = [id2doc[d] for d in doc_ids]

    # Example queries
    queries = [
        ("climate change", ["climate", "change"]),
        ("machine learning", ["machine", "learning"]),
    ]
    methods = ["tfidf", "bm25", "hybrid"]

    # Example relevance scores
    relevance_map = {1:3,2:2,3:5,4:4}

    # Build minimal in-memory BM25 index
    from collections import defaultdict
    total_len = 0
    df = defaultdict(int)
    for doc in candidate_docs:
        total_len += len(doc)
        for term in set(doc):
            df[term] += 1
    N = len(candidate_docs)
    avgdl = total_len / N
    index_pkg = {"N": N, "avgdl": avgdl, "df": df}

    # Ranking tests
    for q_text, q_tokens in queries:
        print(f"Query: '{q_text}'")
        for m in methods:
            ranked_ids, scores = rank_documents(
                q_tokens,
                candidate_docs,
                doc_ids,
                inverted_index_path=index_pkg,
                method=m
            )
            y_true = [relevance_map[rid] for rid in ranked_ids]
            corr = pearson_corr(scores, y_true)
            correct_order = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
            print(f"Method: {m:<6} | Correct order: {'PASS' if correct_order else 'FAIL'} | Pearson r = {corr:.3f}")
            for rid, score in zip(ranked_ids, scores):
                print(f"  Doc {rid}: score={score:.4f}")
        print("-"*50)

    print("Task 3 Verification Complete\n")

# ---------------------------
# Task 4: Search System + MAP Evaluation
# ---------------------------
import os
import subprocess

def run_task4():
    print_header("===========Task 4: Search System + MAP Evaluation Results================")
    """
    Runs Task 4: Search & MAP Evaluation
    Produces run_default.json, run_cutoff.json, run_stopwords.json, run_sanity.json
    """
    queries = "data/dev/queries.json"
    documents = "data/dev/documents.jsonl"
    runs_dir = "runs"
    os.makedirs(runs_dir, exist_ok=True)

    # List of run configurations: (filename, option)
    run_configs = [
        ("run_default.json", None),
        ("run_cutoff.json", "cutoff"),
        ("run_stopwords.json", "stopwords"),
        ("run_sanity.json", "sanity")
    ]

    for filename, option in run_configs:
        run_path = os.path.join(runs_dir, filename)
        print(f"\nRunning search system for {filename}...")

        cmd = ["python", "system/search_system.py", queries, documents, run_path]
        if option:
            cmd.append(option)  # add optional argument

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"Error running {filename}. Check the script or option: {option}")

    print("\nTASK 4 Verification Complete")
    print("="*100 + "\n")

# ---------------------------
# Interactive Menu
# ---------------------------
def main():
    """Interactive CLI menu to run tasks individually or all at once"""
    while True:
        print("\nSelect Task to Run:")
        print("1: Task 1 - Index Verification")
        print("2: Task 2 - Query Processing Verification")
        print("3: Task 3 - Ranking Verification")
        print("4: Task 4 - Search & MAP Evaluation")
        print("5: Run All Tasks")
        print("6: Exit")

        choice = input("Enter your choice (1-6): ").strip()
        
        if choice == "1":
            run_task1()
        elif choice == "2":
            run_task2()
        elif choice == "3":
            run_task3()
        elif choice == "4":
            run_task4()
        elif choice == "5":
            run_task1()
            run_task2()
            run_task3()
            run_task4()
        elif choice == "6":
            print("\nAll Verification Complete")
            print("Exiting............. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number from 1 to 6.")

# Run the interactive menu if script is executed directly
if __name__ == "__main__":
    main()


