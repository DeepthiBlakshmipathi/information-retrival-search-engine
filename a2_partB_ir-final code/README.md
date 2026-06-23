# Information Retrieval Systems — Part B (Starter Skeleton)

This repository is a clean starter for Part B of the IR assignment. It includes module stubs and entry points that match the required interfaces. Many functions are intentionally left unimplemented so students can complete Tasks 1–4.

## Setup

- Python 3.8+
- Optional: virtual environment

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Download minimal NLTK data used by utils/text_preprocessing.py
python - <<'PY'
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
```

## Repository layout

```
index/
  __init__.py
  access.py          # Task 1 access (O(1) in-memory lookups after load)
  builders.py        # Task 1 single builder: create_all_indexes(...)
  io.py              # gzip+pickle I/O helpers
metrics/
  eval_map.py        # Task 4: compute MAP over ./runs/ vs data/dev/relevance_judge.json
query_processing/
  boolean.py         # Task 2: process_boolean_query(...)
  detection.py       # Task 2: detect_query_type(...)
  proximity.py       # Task 2: process_proximity_query(...)
  query_process.py   # Task 2: convert_natural_language(...), process_query(...)
  wildcard.py        # Task 2: process_wildcard_query(...)
ranking/
  rankers.py         # Task 3: rank_documents(...); main prints Pearson table on dev set
system/
  search_system.py   # Task 4: batch CLI
utils/
  embeddings.py      # Part A: semantic_vector(...)
  ngram.py           # Part A: make_ngrams_tokens(...), make_ngrams_chars(...)
  positions.py       # Part A: make_positions(...)
  text_preprocessing.py  # Part A: preprocess(...)
  tfidf.py           # Part A: tfidf_variants(...)
data/
  dev/
    documents.jsonl
    queries.json
    relevance_judge.json
runs/                # Your output runs (*.json)
cache/               # Transient index package file written by the CLI
test.py              # Convenience script to run all tasks and generate outputs
```

# Sanity Check (Smoke Tests)

This folder provides **minimal smoke tests** to ensure your code imports and
runs without crashing. It does **not** check correctness.

## What it checks

- **Task 1 (index)**
  - `index.builders.create_all_indexes(...)` writes a single package
  - `index.access.get_posting_list(...)` -> `List[int]`
  - `index.access.find_wildcard_matches(...)` -> `List[str]`
  - `index.access.get_term_positions(...)` -> `List[int]`

- **Task 2 (query processing)**
  - `query_processing.detection.detect_query_type`
  - `query_processing.query_process.convert_natural_language`
  - `query_processing.boolean.process_boolean_query`
  - `query_processing.wildcard.process_wildcard_query`
  - `query_processing.proximity.process_proximity_query`
  - `query_processing.query_process.process_query`

- **Task 3 (ranking)**
  - `ranking.rankers.rank_documents(...)` -> `(List[int], List[float])`

- **Task 4 (CLI)**
  - Runs `python system/search_system.py data/dev/queries.json data/dev/documents.jsonl runs/run_sanity.json`
  - Validates output JSON schema (`qid`, `doc_ids` list)

## Run

From the repository root:

```bash
python test_sanity/check_submission.py
```

Exit code:

- 0 – all smoke tests passed
- 1 – at least one check failed (see messages)

## Notes

- Uses only the Python standard library.

- Builds a tiny temporary index in `test_sanity/_tmp/index_pkg.pkl` for Task 1/2/3 smoke.

- For Task 4, it uses your dev data at `data/dev/`.

- Creates `runs/` and `cache/` if missing.

## Testing / Demo

A test.py file is included for convenience. It exercises all implemented tasks and produces the required outputs. Run it as follows:

```bash
python -m test
```

This script will:
Build the indexes (Task 1)
Process sample queries (Task 2)
Rank documents and compute Pearson correlation (Task 3)
Run the batch search pipeline and produce outputs in runs/ (Task 4)
Outputs will be printed to the console and/or saved to the appropriate folders.


Steps to Run All Verification Tasks in Your IR Assignment:
1. Open the test.py file and in the terminal navigate to the project folder
2. Run the test module:
```bash
python -m test
```
3. Select tasks to run in the terminal:
   Select Task to Run:
1: Task 1 - Index Verification
2: Task 2 - Query Processing Verification
3: Task 3 - Ranking Verification
4: Task 4 - Search & MAP Evaluation
5: Run All Tasks
6: Exit
Enter your choice (1-6):

To run all tasks, type 5 and press Enter.
To run specific tasks, type the number corresponding to that task.

4. 
- Testing/ Sanity Check
```bash
python .\test_sanity\check_submission.py 
```
or

```bash
python -m test_sanity\check_submission.py 
```
or

```bash
python ./test_sanity/check_submission.py
```
produces a full sanity check. Example output shows all tasks passing:

Task 1: Index building and term/posting lookups
Task 2: Query type detection, boolean, wildcard, proximity, and natural language processing
Task 3: Document ranking
Task 4: CLI batch search and output generation

Summary: All 12 tests passed with 0 failures, confirming that the implemented functions produce the expected outputs.

5. View the results:
The terminal will display the output for the selected tasks, including PASS/FAIL results, posting lists, ranking scores, and MAP evaluation.


- Academic Integrity Declaration: 
I used ChatGPT to draft and/or edit parts of this write-up. All experiments, code, and results are my own work.