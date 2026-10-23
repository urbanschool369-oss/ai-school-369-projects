# rag-freshness: your index is a photograph, not a window

**The idea in simple words.** When you build a RAG index, you take a
photograph of your documents at one moment. The document then changes. The
photograph does not. Your chatbot keeps answering from the old photograph,
confidently, until someone notices.

This project shows the three habits that keep the photograph current:

1. **Version stamps.** Every chunk carries the version of its document.
   When a document changes, only its chunks get re-embedded.
2. **Reindex on change.** A file hash decides what actually changed, so
   you never pay to reindex the whole corpus for one edited paragraph.
3. **Deletion propagation.** When a document is deleted, its chunks leave
   the index in the same run. Nothing answers from a document that no
   longer exists.

And the safety net: **20 canary questions** with known answers. If a
refresh breaks an answer, the canaries fail loudly before your users do.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

`main.py` builds a tiny index from `docs/`, answers the canary questions,
then edits one document, deletes another, refreshes the index, and answers
again. It works on a scratch copy of `docs/`, so the demo never touches
the originals and you can run it as often as you like. Watch the second
round of answers: the edited answer changes, the deleted document stops
answering, and the unchanged answers stay identical.

## What to notice in the output

- The refresh log lists only the chunks that changed (1 document edited =
  its chunks only, not the corpus).
- The deleted document's chunks are gone from the index count.
- Canary results print PASS/FAIL per question, before and after.

## Files

- `main.py`: the whole flow, commented in plain language
- `docs/`: three tiny sample documents (one gets edited, one deleted)
- `canaries.json`: the 20 guard questions and their expected answers
- `requirements.txt`: the one dependency (numpy, for the vectors)
- `output.txt`: a real run, captured so you can see the expected result

## Try this

1. Edit a document WITHOUT running the refresh. Ask its canary question.
   That stale answer is the bug this project exists to prevent.
2. Break a chunker setting and watch which canaries catch it.

## Newsletter

Explained in simple words in Layer by Layer, Edition 1:
"Your RAG did not notice the document changed."
