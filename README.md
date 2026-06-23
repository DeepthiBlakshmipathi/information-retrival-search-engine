# Information Retrieval Search Engine

An end-to-end information retrieval system built from scratch in Python — covering foundational text-processing utilities (Part A) and a full search engine with ranking and evaluation (Part B).

## Overview

This project implements a complete IR pipeline without relying on high-level search libraries:

- **Part A — Core Utilities** (`a1_partA_ir-startcode/`): text preprocessing, n-gram generation, positional indexing, TF-IDF variants (raw, log, BM25-style), and semantic vector embeddings (GloVe-based).
- **Part B — Search System** (`a2_partB_ir-final code/`): inverted indexing, query type detection, boolean/wildcard/proximity query processing, natural-language query conversion, document ranking (TF-IDF, BM25, and a hybrid method), and MAP-based evaluation against relevance judgments.

## Tech Stack

Python · NumPy · NLTK · BeautifulSoup4 · custom inverted-index and ranking implementations

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python - <<'PY'
import nltk
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(pkg)
PY
```

> **Note:** Part A's embeddings module expects a local GloVe file (`glove.6B.100d.txt`), which is too large to include in this repo. Download it from [nlp.stanford.edu/projects/glove](https://nlp.stanford.edu/projects/glove/) and place it at `a1_partA_ir-startcode/utils/glove.6B/glove.6B.100d.txt`.

## Running the System

Run all tasks interactively:
```bash
python -m test
```
You'll get a menu:
1: Task 1 - Index Verification
2: Task 2 - Query Processing Verification
3: Task 3 - Ranking Verification
4: Task 4 - Search & MAP Evaluation
5: Run All Tasks

Run the smoke-test suite (12 sanity checks across all tasks):
```bash
python test_sanity/check_submission.py
```

## Results

**Ranking method comparison (Pearson correlation vs. gold ranking):**

| Query | TF-IDF | BM25 | Hybrid |
|---|---|---|---|
| "machine learning" | 0.775 | 0.775 | 0.775 |
| "climate change" | -0.284 | -0.264 | -0.267 |

BM25 was selected as the default ranker — robust across query types and document lengths, with a simpler implementation than the hybrid approach for comparable performance.

**System ablation (MAP@100, baseline = 0.6282):**

| Variant | MAP | Effect |
|---|---|---|
| Baseline (full system) | 0.6282 | — |
| Stopword removal | 0.0180 | Drastic decrease — removed contextually important terms |
| Aggressive candidate cutoff (500 docs/query) | 0.0180 | Drastic decrease — pruned relevant documents |

These ablations show that naive filtering/pruning for efficiency can severely hurt retrieval effectiveness despite the expected speed gains.

## Project Structure
a1_partA_ir-startcode/   # Part A: preprocessing, n-grams, TF-IDF, embeddings
a2_partB_ir-final code/  # Part B: indexing, query processing, ranking, evaluation

## Academic Context

Built as a two-part assignment for an Information Retrieval course (RMIT, Master of Data Science). All code, experiments, and results are original work; ChatGPT was used to help draft/edit parts of the written report.
