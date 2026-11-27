# Author: Shekar Kaki
"""reranking-lab: retrieval finds candidates. Reranking finds relevance.

The idea in simple words: a RAG pipeline has two jobs, and most systems
only do the first one well. Retrieval is the fast, blunt pass: it embeds
the question and each chunk separately and compares vectors, so the
question and the chunk never actually meet. That is what makes it fast
enough to search millions of chunks, and it is also what makes it
imprecise. Similarity is not relevance.

Reranking is the careful second pass over a small candidate set: it reads
the question and each candidate together, as a pair, and scores true
relevance. Same corpus, same question. The only change is the second pass.

This demo runs one question through both passes over the same 8 chunks,
so you can watch the answer move:

  Question: "How do I reset a user's API key?"
  Chunk 1 holds the real reset steps. The others are keyword-stuffed
  pages, near-misses (create a key, reset a password, rotate a service
  key), or unrelated. Retrieval scores them in a tight band and the
  keyword-stuffer wins. Reranking spreads them out: the real answer
  jumps to the top and the stuffing sinks.

There is no real vector database here. Retrieval is TF-IDF cosine
similarity, written out in plain Python so you can see exactly what it
does (and exactly what it cannot do: it never reads the question and a
chunk together). The reranker is one of two things, and the program tells
you which one it used at the top of its output:

  * A real cross-encoder (cross-encoder/ms-marco-MiniLM-L-6-v2) when
    sentence-transformers is installed. It reads each (question, chunk)
    pair together and scores relevance. See requirements.txt.
  * Otherwise, a lightweight stand-in pairwise scorer, labelled as such
    in the output. It reads the question and the chunk together with
    plain term rules (exact phrase, ordered how-to sequence, imperative
    steps, and a penalty for repeating one keyword). It is a stand-in
    for a hosted cross-encoder, not a cross-encoder. Swap in the real
    one with: pip install -r requirements.txt

That keeps the demo free and identical on every run, so you
can study the second pass itself, which is the part that matters. After
the first run downloads the cross-encoder model, it runs offline; the
stand-in path needs no download at all.
"""

import math
import os
import re
import sys

# ------------------------------------------------------------------ query
# The question from the channel's reranking Short concept.

QUERY = "How do I reset a user's API key?"

# ----------------------------------------------------------------- corpus
# 8 chunks. Chunk 1 is the only one that answers the question. The rest
# are the traps a retriever falls for: keyword stuffing (2 and 6),
# near-misses that share words with the question (3, 4, 5, 8), and one
# chunk with nothing to do with it (7).

CHUNKS = [
    ("1. Reset a user's API key (the real answer)",
     "To reset a user's API key, open the admin console, go to Users, "
     "select the user, click API Keys, then click Reset key. The old key "
     "stops working immediately and a new key is shown once, so copy it "
     "now: it will not be shown again."),
    ("2. API keys overview (keyword-stuffed)",
     "API key, API key, API key. This page is about API keys, users, "
     "reset, and passwords. API key reset user key. Read about API keys, "
     "user accounts, and key management here."),
    ("3. Create a new API key (near-miss)",
     "To create a new API key, open your profile, go to Developer "
     "settings, and click Generate new key. Give the key a name and set "
     "what it is allowed to do."),
    ("4. Reset your password (near-miss)",
     "To reset your password, click Forgot password on the sign-in page. "
     "We email you a link that expires in one hour, and the old password "
     "keeps working until you use it."),
    ("5. Rotate a service account key (near-miss)",
     "To rotate a service account key, run: cli service-keys rotate "
     "--service billing. The old key stays valid for 24 hours while the "
     "new one rolls out."),
    ("6. Resetting things (keyword-stuffed)",
     "Resetting things: reset your API key, reset your password, reset "
     "your settings. Users ask how to reset API keys all the time, so "
     "this guide covers reset, API, key, and user topics."),
    ("7. Billing and invoices (unrelated)",
     "Billing: upgrade your plan from Settings, then Billing. Invoices "
     "are emailed on the first of the month and receipts live under "
     "Billing history."),
    ("8. API authentication docs (near-miss)",
     "API authentication uses Bearer tokens. Pass your key in the "
     "Authorization header. Keys are scoped to one workspace and cannot "
     "be shared across workspaces."),
]

TOP_K = 5  # the model reads the top 5 after each pass, not the top 8

# ------------------------------------------------------- retrieval: TF-IDF
# Plain term matching, written out so the mechanism is visible. The query
# and each chunk are tokenized separately and compared as vectors. They
# never meet. That is the whole limitation this demo is about.

STOPWORDS = {
    "how", "do", "i", "a", "an", "the", "to", "of", "and", "or", "in",
    "on", "is", "it", "for", "with", "you", "your", "we", "this", "that",
    "then", "so", "by", "at", "from", "are", "be", "as", "if",
}


def tokenize(text):
    """Lowercase words, split on anything that is not a letter or digit.

    Possessives are folded away first ("user's" -> "user"), and
    single-letter leftovers are dropped, so the query "How do I reset a
    user's API key?" becomes the four content terms reset, user, api, key.
    """
    text = text.lower().replace("'s", "")
    words = re.findall(r"[a-z0-9]+", text)
    return [w for w in words if len(w) > 1 and w not in STOPWORDS]


def tfidf_vectors(docs):
    """Return (vectors, idf): one sparse TF-IDF vector per document."""
    tokenized = [tokenize(d) for d in docs]
    doc_count = len(docs)
    doc_freq = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    idf = {term: math.log(doc_count / df) + 1.0
           for term, df in doc_freq.items()}
    vectors = []
    for tokens in tokenized:
        counts = {}
        for term in tokens:
            counts[term] = counts.get(term, 0) + 1
        total = len(tokens) or 1
        vectors.append({term: (count / total) * idf[term]
                        for term, count in counts.items()})
    return vectors, idf


def cosine(vec_a, vec_b):
    """Cosine similarity between two sparse vectors (0 to 1)."""
    dot = sum(value * vec_b.get(term, 0.0)
              for term, value in vec_a.items())
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(query, chunks, top_k):
    """Rank chunks by TF-IDF cosine similarity to the query alone."""
    docs = [text for _title, text in chunks]
    vectors, idf = tfidf_vectors(docs + [query])
    query_vec = vectors[-1]
    scored = []
    for index, (title, _text) in enumerate(chunks):
        score = cosine(query_vec, vectors[index])
        scored.append((score, index, title))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k], scored  # top-k, and the full ranking


# ------------------------------------------- reranker: real or stand-in?
# A cross-encoder reads (question, chunk) as one pair. The stand-in below
# does the same job with plain rules, and says so loudly: its scores are
# NOT cross-encoder scores. To use the real one:
#   pip install -r requirements.txt
#   python main.py

IMPERATIVES = {"open", "go", "click", "select", "run", "copy", "give",
               "set", "use", "pass"}
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_cross_encoder():
    """Return a CrossEncoder, or None if sentence-transformers is absent."""
    if os.environ.get("RERANKER") == "standin":
        return None
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None
    try:
        return CrossEncoder(CROSS_ENCODER_MODEL)
    except Exception as exc:  # noqa: BLE001 - model download can fail
        print(f"Cross-encoder unavailable ({exc}); using the stand-in.",
              file=sys.stderr)
        return None


def standin_score(query, chunk_text):
    """Lightweight pairwise scorer: a STAND-IN, not a cross-encoder.

    Reads the question and the chunk together with plain term rules.
    Scores 0-100. Honest about what it is: term matching with a
    stuffing penalty, not a learned relevance model.
    """
    query_terms = set(tokenize(query))
    chunk_tokens = tokenize(chunk_text)
    chunk_lower = chunk_text.lower()
    present = query_terms & set(chunk_tokens)
    score = 20.0 * (len(present) / max(len(query_terms), 1))
    if "api key" in chunk_lower or "api keys" in chunk_lower:
        score += 25.0  # the exact phrase, read as a pair
    reset_pos = chunk_lower.find("reset")
    key_pos = chunk_lower.find("key")
    if reset_pos != -1 and key_pos != -1 and reset_pos < key_pos:
        score += 20.0  # reset comes before key: a how-to sequence
    imperatives = {t for t in chunk_tokens if t in IMPERATIVES}
    if len(imperatives) >= 2:
        score += 15.0  # step language: a how-to answer, not a mention
    if "user" in chunk_tokens and imperatives:
        score += 10.0  # the user is the object of an action
    for term in query_terms:
        count = chunk_tokens.count(term)
        if count >= 4 and count / max(len(chunk_tokens), 1) > 0.15:
            # One keyword repeated until it is a large share of the
            # chunk: stuffing, not relevance. A real answer repeats the
            # object of the question too, but only as a small share of
            # a longer text, so it is not penalized.
            score -= 30.0
            break
    return max(0.0, min(100.0, score))


def rerank(query, chunks, cross_encoder):
    """Score every chunk against the query, as a pair. Returns a ranking.

    The ranking is (score, index, title). The two modes use different
    scales, and the output says which one was used, so never compare a
    number from one mode with a number from the other:
    * cross-encoder: the model's raw logit (higher = more relevant).
    * stand-in: 0-100 from the term rules.
    """
    if cross_encoder is not None:
        pairs = [(query, text) for _title, text in chunks]
        raw = cross_encoder.predict(pairs)
        scored = [(float(raw[index]), index, title)
                  for index, (title, _text) in enumerate(chunks)]
    else:
        scored = [(standin_score(query, text), index, title)
                  for index, (title, text) in enumerate(chunks)]
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


# -------------------------------------------------------------------- main

def show(title, ranking, top_k, scale=1.0):
    print(f"\n{title}")
    for rank, (score, _index, chunk_title) in enumerate(ranking[:top_k], 1):
        print(f"  {rank}. {score * scale:6.2f}  {chunk_title}")


def main():
    print("reranking-lab: the same question, two passes over 8 chunks.")
    print(f"Question: {QUERY}")
    print(f"Corpus: {len(CHUNKS)} chunks. The model reads the top "
          f"{TOP_K} after each pass.")

    cross_encoder = load_cross_encoder()
    if cross_encoder is not None:
        print(f"Reranker: REAL cross-encoder "
              f"({CROSS_ENCODER_MODEL}).")
    else:
        print("Reranker: STAND-IN pairwise scorer (term rules, labelled "
              "as such). Install the real cross-encoder with:\n"
              "  pip install -r requirements.txt")

    retrieved_top, retrieved_full = retrieve(QUERY, CHUNKS, TOP_K)
    show("PASS 1 — RETRIEVAL ONLY (TF-IDF cosine, question and chunk "
         "scored separately; scores shown 0-100)", retrieved_full, TOP_K,
         scale=100.0)

    if cross_encoder is not None:
        pass2_title = ("PASS 2 — AFTER RERANKING (real cross-encoder; "
                       "raw logits, higher = more relevant)")
    else:
        pass2_title = ("PASS 2 — AFTER RERANKING (stand-in pairwise "
                       "scorer; scores 0-100)")
    reranked = rerank(QUERY, CHUNKS, cross_encoder)
    show(pass2_title, reranked, TOP_K)

    # The lesson, computed from the actual rankings, not asserted.
    answer_title = CHUNKS[0][0]
    before = next(r for r, (_s, i, _t) in
                  enumerate(retrieved_full, 1) if i == 0)
    after = next(r for r, (_s, i, _t) in
                 enumerate(reranked, 1) if i == 0)
    print(f"\nThe chunk with the reset steps ({answer_title}) moved from "
          f"#{before} after retrieval to #{after} after reranking.")
    if cross_encoder is not None:
        print("The stuffed chunks stay near the top: they mention the "
              "right words, so the cross-encoder scores them too. "
              "Reranking decides which chunk the model reads first; it "
              "does not punish repetition by itself. Better chunking is "
              "the fix for stuffing, not a stronger reranker.")
    else:
        print("Retrieval scores sit in a tight band; the keyword-stuffed "
              "chunks win on repetition. Reranking spreads the scores "
              "out: the real answer rises, the stuffing sinks.")
    print("\nDone. Same corpus, same question. The second pass is the "
          "difference.")


if __name__ == "__main__":
    main()
