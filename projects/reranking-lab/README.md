# reranking-lab: retrieval finds candidates. Reranking finds relevance

**The idea in simple words.** A RAG pipeline has two jobs, and most systems
only do the first one well. Retrieval is the fast, blunt pass: it embeds
the question and each chunk separately and compares vectors, so the
question and the chunk never actually meet. That is what makes it fast
enough to search millions of chunks, and it is also what makes it
imprecise. Similarity is not relevance.

Reranking is the careful second pass over a small candidate set: it reads
the question and each candidate together, as a pair, and scores true
relevance. Same corpus, same question. The only change is the second pass.

This project runs one question through both passes over the same 8 chunks,
so you can watch the answer move:

- **Question:** "How do I reset a user's API key?"
- **Chunk 1** holds the real reset steps. The others are keyword-stuffed
  pages, near-misses (create a key, reset a password, rotate a service
  key), or unrelated. Retrieval scores them in a tight band and the
  keyword-stuffer wins. Reranking spreads them out: the real answer
  jumps to the top and the stuffing sinks.

## Run it

```bash
pip install -r requirements.txt
python main.py
```

The install pulls `sentence-transformers` and a CPU torch build, then the
first run downloads the cross-encoder model
(`cross-encoder/ms-marco-MiniLM-L-6-v2`, about 80 MB). After that the run
is offline. No API key is needed anywhere. If you skip the install, the
program still runs: it falls back to a labelled stand-in scorer (see
"What the reranker is") and tells you so at the top of its output.
There is no real vector database here: retrieval is TF-IDF cosine
similarity, written out in plain Python so you can see exactly what it
does (and exactly what it cannot do: it never reads the question and a
chunk together).

## What to notice in the output

- After retrieval alone, the two keyword-stuffed chunks are #1 (66.21)
  and #2 (59.25). The chunk with the actual reset steps is #3 (44.12).
  The model reads the top 5, so it would answer from the wrong material
  with full confidence, and nothing in the pipeline would report an
  error.
- After reranking with the real cross-encoder, the reset-steps chunk is
  #1 with a logit of 10.91. The model now reads the right chunk first.
  That move, #3 to #1, is the whole point of the second pass.
- The honest caveat is in the same table: the two stuffed chunks stay
  #2 (7.64) and #3 (5.98). They mention the right words, so the
  cross-encoder scores them too. Reranking decides which chunk the model
  reads first; it does not punish repetition by itself. Better chunking
  is the fix for stuffing, not a stronger reranker.
- The password-reset near-miss drops out of the top 5 entirely, and the
  authentication-docs chunk falls to -0.90. The cross-encoder's raw
  logits are shown as-is (higher = more relevant); they are not
  percentages, and they are not comparable with the retrieval scores or
  with the stand-in's 0-100 scale.

## Sample output

Real output of `python main.py` with the cross-encoder installed,
captured in `output.txt`:

```
reranking-lab: the same question, two passes over 8 chunks.
Question: How do I reset a user's API key?
Corpus: 8 chunks. The model reads the top 5 after each pass.
Reranker: REAL cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2).

PASS 1 — RETRIEVAL ONLY (TF-IDF cosine, question and chunk scored separately; scores shown 0-100)
  1.  66.21  2. API keys overview (keyword-stuffed)
  2.  59.25  6. Resetting things (keyword-stuffed)
  3.  44.12  1. Reset a user's API key (the real answer)
  4.  18.00  3. Create a new API key (near-miss)
  5.   9.68  8. API authentication docs (near-miss)

PASS 2 — AFTER RERANKING (real cross-encoder; raw logits, higher = more relevant)
  1.  10.91  1. Reset a user's API key (the real answer)
  2.   7.64  6. Resetting things (keyword-stuffed)
  3.   5.98  2. API keys overview (keyword-stuffed)
  4.   4.86  3. Create a new API key (near-miss)
  5.  -0.90  8. API authentication docs (near-miss)

The chunk with the reset steps (1. Reset a user's API key (the real answer)) moved from #3 after retrieval to #1 after reranking.
The stuffed chunks stay near the top: they mention the right words, so the cross-encoder scores them too. Reranking decides which chunk the model reads first; it does not punish repetition by itself. Better chunking is the fix for stuffing, not a stronger reranker.

Done. Same corpus, same question. The second pass is the difference.
```

## What the reranker is

The run above used the **real cross-encoder**
`cross-encoder/ms-marco-MiniLM-L-6-v2`. It reads each (question, chunk)
pair together and scores relevance; the program prints the raw logits, so
the numbers are the model's own, not a rescaled version of them.

If `sentence-transformers` is not installed, the program falls back to a
**stand-in pairwise scorer** and says so at the top of its output. The
stand-in reads the question and the chunk together with plain term rules:
the exact phrase ("API key"), the ordered how-to sequence (reset before
key), imperative step language (open, go, click, select), and a penalty
when one keyword is repeated until it is a large share of the chunk. It
is a stand-in for a hosted cross-encoder, **not** a cross-encoder. Its
numbers must never be presented as cross-encoder results, and its 0-100
scale is not comparable with the cross-encoder's logits. Force it with
`RERANKER=standin python main.py` to compare the two rankings.

## What this demo is and is not

- It **is** a mechanism demo on 8 hand-written chunks: it shows why a
  second pass changes which chunk the model reads.
- It **is not** a production benchmark. Eight chunks are not an index,
  TF-IDF is not a vector database, and the stand-in scorer is not a
  learned relevance model. The numbers prove the mechanism, not a
  production quality gain. Measure a real system with the same eval set
  before and after, and report a holdout number separately from the
  tuning number.

## Files

- `main.py`: the corpus, TF-IDF retrieval, both reranker paths, and the
  before/after printout, commented in plain language
- `README.md`: this file
- `requirements.txt`: sentence-transformers (the real cross-encoder);
  without it the program falls back to the labelled stand-in scorer
- `output.txt`: a real run with the cross-encoder, captured so you can
  see the expected result

## Try this

1. In `main.py`, set `TOP_K` to 3 and run again. After retrieval the
   model would never even see the real answer: it sits at #3, and the
   top 3 are two stuffers and the answer. After reranking it is #1.
2. Add a 9th chunk that answers a different question well (for example,
   "How do I delete a workspace?") and run again. It should score low
   on both passes, because relevance is about this question, not about
   being a good chunk in general.
3. Run with `RERANKER=standin python main.py` to force the stand-in even
   when sentence-transformers is installed, and compare the two rankings.

## Newsletter

Explained in simple words in Layer by Layer, Edition 3:
"Reranking: the cheapest quality upgrade in RAG."
