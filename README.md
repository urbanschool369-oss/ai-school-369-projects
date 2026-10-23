# AI School 369 Projects

Small, runnable projects that teach production AI engineering, one idea at
a time. Every project pairs with an edition of **Layer by Layer**, the AI
School 369 email newsletter: the newsletter explains the concept in simple
words, this repo lets you run it.

Each project is deliberately small (100–300 lines), heavily commented in
plain language, and built to be read top to bottom in one sitting.

## Projects

Projects 1 and 2 are built and runnable. Projects 3 to 6 are on the way,
one per newsletter edition.

| # | Project | The one idea | Newsletter edition |
|---|---|---|---|
| 1 | [rag-freshness](projects/rag-freshness/) | A RAG index is a photograph, not a window. Version stamps, reindex-on-change, and deletion propagation keep it honest. | E1: Your RAG did not notice the document changed |
| 2 | [agent-tool-loop](projects/agent-tool-loop/) | An agent loop needs a budget: max steps, retries, and a circuit breaker, or it spins forever. | E2: Anatomy of an agent tool loop |
| 3 | [reranking-lab](projects/reranking-lab/) | Reranking is the cheapest quality upgrade in RAG. Same query, with and without, side by side. | E3: Reranking, the cheap upgrade |
| 4 | [agent-memory-sql](projects/agent-memory-sql/) | Agent memory you can inspect: plain SQLite, no black box. | E4: Agent memory you can actually inspect |
| 5 | [canary-evals](projects/canary-evals/) | 20 canary questions that guard a RAG pipeline against silent regressions. | E5: Canary evals |
| 6 | [confirmation-gates](projects/confirmation-gates/) | Autonomy is earned per action, not granted per agent. Confirmation gates sized by blast radius. | E6: Autonomy is earned per action |

## How to run any project

```bash
cd projects/<project-name>
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Each project folder has its own README with the concept in simple words,
what to notice in the output, and exercises to try.

## The newsletter

New project + a simple explanation, every Tuesday and Friday.
Subscribe: (link added at launch)

## License

MIT

---

Created by Shekar Kaki — AI School 369
