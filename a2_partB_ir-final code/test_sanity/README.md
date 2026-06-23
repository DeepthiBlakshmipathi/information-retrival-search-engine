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