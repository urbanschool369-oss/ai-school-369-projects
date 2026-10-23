# Author: Shekar Kaki
"""rag-freshness: keep a tiny RAG index honest when documents change.

The idea in simple words: a search index is a photograph of your documents
at one moment. When a document changes, the photograph does not. This demo
shows the three habits that keep the photograph current:

  1. Version stamps: every chunk remembers which version of its document
     it came from.
  2. Reindex on change only: a file hash tells us what actually changed,
     so we re-embed just those chunks, not the whole corpus.
  3. Deletion propagation: when a document disappears, its chunks leave
     the index in the same run.

A set of 20 canary questions (questions with known answers) checks the
index before and after a change, so a stale answer fails loudly.

Embeddings here use a tiny bag-of-words embedder (one vector slot per
known word, shared by the whole index) so the demo runs offline with no
model download and no API key. In a real system you would swap in a
proper embedding model; the freshness logic around it stays exactly
the same.
"""

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
THRESHOLD = 0.30    # best match below this score counts as "no answer"

# Very common words carry no meaning for search, so we drop them.
STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "did", "i", "you", "my",
    "me", "it", "to", "of", "for", "in", "on", "and", "or", "if", "will",
    "can", "how", "what", "when", "there", "be", "am", "is", "was",
}


def stem(word):
    """Crude word stemmer: makes 'returns' and 'return' match.

    This is deliberately simple. It trims common endings so small word
    differences do not break matching in this tiny demo.
    """
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "i"
    for ending in ("ing", "ed", "es", "s"):
        if len(word) > len(ending) + 2 and word.endswith(ending):
            return word[: -len(ending)]
    return word


def tokens(text):
    """Split text into cleaned, stemmed content words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [stem(w) for w in words if w not in STOPWORDS]


def embed(text, vocab):
    """Turn text into a vector: one slot per known word, counting uses.

    The vocabulary is shared by every chunk and every question, so two
    texts that use the same words get vectors pointing in similar
    directions, and cosine similarity measures how similar. Words that
    appear in no document get no slot: they cannot match anything anyway.
    Simple, fast, and fully offline.
    """
    vec = np.zeros(len(vocab), dtype=float)
    for tok in tokens(text):
        slot = vocab.get(tok)
        if slot is not None:
            vec[slot] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def chunk_document(text):
    """Split a document into chunks. Our tiny docs keep one sentence per
    line, so each line is one chunk. Real systems chunk by size and
    overlap; the freshness logic does not care how you chunk."""
    return [line.strip() for line in text.splitlines() if line.strip()]


class Index:
    """A minimal in-memory index: a list of chunks plus a manifest.

    The manifest is the freshness brain. For each document it remembers
    the file hash (what the content is) and a version number (how many
    times the content has changed since we first saw it).
    """

    def __init__(self):
        self.chunks = []    # each: {"doc", "version", "text", "vector"}
        self.manifest = {}  # doc name: {"hash": str, "version": int}
        self.vocab = {}     # word: vector slot, shared by all embeddings

    def _grow_vocab(self, texts):
        """Add any new words to the shared vocabulary.

        Existing chunk vectors get zero padded to the new size. Padding
        is not re-embedding: their numbers do not change, the vector
        just gains empty slots for words those chunks do not use.
        """
        for text in texts:
            for tok in tokens(text):
                if tok not in self.vocab:
                    self.vocab[tok] = len(self.vocab)
        size = len(self.vocab)
        for chunk in self.chunks:
            missing = size - len(chunk["vector"])
            if missing > 0:
                chunk["vector"] = np.pad(chunk["vector"], (0, missing))

    def refresh(self, docs_dir):
        """Compare the folder with the manifest and apply only the diff.

        Returns a list of human readable log lines describing exactly
        what changed. Unchanged documents are skipped and cost nothing.
        """
        log = []
        on_disk = {p.name: p for p in sorted(docs_dir.glob("*.txt"))}

        # The vocabulary must know every word before anything is embedded.
        self._grow_vocab([p.read_text(encoding="utf-8")
                          for p in on_disk.values()])

        # Pass 1: deletions. A file that vanished takes its chunks with it.
        for name in list(self.manifest):
            if name not in on_disk:
                kept = [c for c in self.chunks if c["doc"] != name]
                dropped = len(self.chunks) - len(kept)
                self.chunks = kept
                del self.manifest[name]
                log.append(f"  DELETED {name}: removed {dropped} chunks")

        # Pass 2: new and changed files. The hash decides, not the clock.
        for name, path in on_disk.items():
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            known = self.manifest.get(name)
            if known and known["hash"] == digest:
                log.append(f"  SAME    {name}: unchanged, skipped")
                continue
            version = 1 if known is None else known["version"] + 1
            # Drop the old chunks for this document, then embed fresh ones.
            self.chunks = [c for c in self.chunks if c["doc"] != name]
            fresh = chunk_document(raw.decode("utf-8"))
            for text in fresh:
                self.chunks.append(
                    {"doc": name, "version": version,
                     "text": text, "vector": embed(text, self.vocab)}
                )
            self.manifest[name] = {"hash": digest, "version": version}
            action = "INDEXED" if known is None else "CHANGED"
            log.append(
                f"  {action} {name}: embedded {len(fresh)} chunks "
                f"(version {version})"
            )
        return log

    def answer(self, question):
        """Return the best matching chunk text, or None for 'no answer'.

        This is extractive on purpose: the demo returns the source
        sentence itself, so you can see exactly which chunk answered.
        """
        if not self.chunks:
            return None
        qvec = embed(question, self.vocab)
        scores = [float(qvec @ c["vector"]) for c in self.chunks]
        best = int(np.argmax(scores))
        if scores[best] < THRESHOLD:
            return None
        return self.chunks[best]["text"]


def run_canaries(index, canaries, phase):
    """Ask every canary question and check the answer against the truth
    for this phase ('before' or 'after' the document change).

    An expected value of None means the honest answer is 'no answer':
    the document that held the fact is gone, so the system must not
    invent one. Returns the number of failures.
    """
    failures = 0
    for canary in canaries:
        expected = canary[phase]
        got = index.answer(canary["question"])
        if expected is None:
            ok = got is None
            shown = got if got else "(no answer)"
        else:
            ok = got is not None and expected.lower() in got.lower()
            shown = got if got else "(no answer)"
        print(f"  [{'PASS' if ok else 'FAIL'}] Q{canary['id']:>2}: "
              f"{canary['question']}")
        print(f"        answer: {shown}")
        if not ok:
            failures += 1
    return failures


def main():
    canaries = json.loads((HERE / "canaries.json").read_text())

    # Work on a scratch copy of docs/ so the demo can edit and delete
    # freely and you can run it again and again with the same result.
    scratch = Path(tempfile.mkdtemp(prefix="rag-freshness-"))
    docs_dir = scratch / "docs"
    shutil.copytree(HERE / "docs", docs_dir)
    print(f"Working on a scratch copy of docs/ at {scratch}\n")

    index = Index()

    print("STEP 1: Build the index")
    for line in index.refresh(docs_dir):
        print(line)
    print(f"  Index now holds {len(index.chunks)} chunks.\n")

    print("STEP 2: Canary check on the fresh index (expect all PASS)")
    fails = run_canaries(index, canaries, "before")
    print(f"  Canary result: {20 - fails}/20 passed.\n")

    print("STEP 3: The world changes (but nobody refreshes yet)")
    returns = docs_dir / "returns-policy.txt"
    returns.write_text(
        returns.read_text().replace("within 30 days of delivery",
                                    "within 60 days of delivery")
    )
    (docs_dir / "shipping-guide.txt").unlink()
    print("  returns-policy.txt edited: 30 days became 60 days.")
    print("  shipping-guide.txt deleted.")
    stale = index.answer("How many days do I have to return an item?")
    print(f"  Asking the OLD index anyway: {stale}")
    print("  That answer is stale. The photograph does not know.\n")

    print("STEP 4: Refresh (only the diff is applied)")
    for line in index.refresh(docs_dir):
        print(line)
    print(f"  Index now holds {len(index.chunks)} chunks.")
    fresh = index.answer("How many days do I have to return an item?")
    print(f"  Same question, refreshed index: {fresh}")
    gone = index.answer("How long does standard shipping take?")
    print(f"  Shipping question now: {gone if gone else '(no answer)'}")
    print("  The deleted document stopped answering, as it should.\n")

    print("STEP 5: Canary check after the change (expect all PASS)")
    fails = run_canaries(index, canaries, "after")
    print(f"  Canary result: {20 - fails}/20 passed.")
    if fails:
        print("  Something is stale or missing. The canaries caught it.")
        raise SystemExit(1)
    print("\nDone. The index noticed the document changed. That is the point.")


if __name__ == "__main__":
    main()
