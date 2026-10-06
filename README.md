# Somajji

AI helper for community mental health volunteers in the Chittagong Hill Tracts.
"Somajji" is a Chakma word for "companion".

**Demo only. Not for real patients.**

Status: in development. This is a portfolio project, built one phase at a time. The detailed
plan is kept private. This README will be finished when the project is deployed.

## Phase 1: the knowledge base (done)

Somajji never talks to the person in distress. A volunteer does, and Somajji suggests next steps
**only from three WHO guides for non-specialists, with the page cited**. Phase 1 builds the part that
finds the right page.

```
download PDFs -> read clean pages (printed page number) -> cut chunks -> embed (Voyage voyage-4)
              -> store (Pinecone) -> search 20 candidates -> rerank (Voyage rerank-3-lite) -> best 5 + source + page
```

### The sources

The guides belong to the World Health Organization. They are **not** in this repository: a script
downloads them from WHO's site and checks each file's hash.

| Short name | Guide | Notes |
| --- | --- | --- |
| `pfa` | Psychological first aid: Guide for field workers (WHO, 2011) | The WHO page names no open licence, only WHO's permissions policy |
| `mhgap` | mhGAP Intervention Guide, **Version 2.0** (WHO, 2016) | Same licence note as above. A newer version could not be confirmed on WHO's site, so 2.0 is used |
| `pmplus` | Problem Management Plus (PM+), individual version (WHO, 2018) | CC BY-NC-SA 3.0 IGO. The cover reads "WHO generic field-trial version 1.1, 2018" |

This is a non-commercial demo. Check each guide's licence page before reusing anything.

### Run it

You need [uv](https://docs.astral.sh/uv/), a `.env` copied from `.env.example`, a Voyage key and a
Pinecone key (both have free plans; the Pinecone free plan allows AWS us-east-1 only).

```bash
uv sync
uv run python scripts/download_sources.py        # the 3 PDFs, hashes checked
uv run python -m scripts.preflight               # free checks: keys set? Voyage and rerank work? which indexes exist?
uv run python -m scripts.create_index            # shows what it WOULD create; nothing happens
uv run python -m scripts.create_index --create   # really creates the index (an online resource: note it for teardown)
uv run python -m app.rag.ingest --dry-run        # chunks and token count, no keys needed
uv run python -m app.rag.ingest                  # embed and store (about 156,000 tokens; free on voyage-4)
uv run python -m eval.retrieval_eval --check     # validate the question file against the PDFs
uv run python -m eval.retrieval_eval             # score it
```

If a command says "unrecognized arguments", the two hyphens were probably turned into a long dash by the
app you copied from. The tools accept that, but typing the hyphens by hand always works.

### Tests

```bash
uv run pytest                  # unit tests: fast, free, no keys needed
uv run pytest -m integration   # live tests against Voyage and Pinecone; skipped when keys are missing
```

### Results, and how far to trust them

Right page of the right guide in the top 5, on 20 situations written for hill communities
(jhum farming, the headman, a landslide, a fire in the para, land worries):

| | Search only | Search + reranker (current) |
| --- | --- | --- |
| English, 20 questions (target: 16) | 15/20 | **19/20** |
| Bangla, 16 questions (informational) | 11/16 | 12/16 |

What was tried and measured: hybrid keyword + meaning search made results **worse** (12/20), adding the page's
module name to the embedded text changed nothing for English, and a 500-token chunk size scored the same as 800.

Please read these numbers with these caveats:

- **The question set was drafted by an AI** (Claude) from the guides, not written by hand. Every question
  carries a quote copied word for word from its answer page, and the checker proves the quote is on that
  page. A person spot-checked a sample. The questions are in `data/`.
- **I tuned on the same questions** (chunk size, the reranker, two reranker models compared), so 19/20 is
  somewhat optimistic. A fresh question set would be a fairer test.
- The guides are in English. Bangla questions find English pages reasonably well, but the Bangla sample is
  small and Banglish (Bangla typed in English letters) is the weakest.
- Local languages such as Chakma are not supported. The volunteer is the bridge.

**Demo only. Not for real patients. Made-up situations only.**
