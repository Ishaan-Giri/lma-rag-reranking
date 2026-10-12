# RAG Assignment: Baseline vs. Cross-Encoder Re-Ranking

Two retrieval pipelines over the same chunked rulebook:

- **Pipeline 1 (baseline):** a bi-encoder (`sentence-transformers/all-MiniLM-L6-v2`) indexes chunks in ChromaDB. The top 10 are retrieved by cosine similarity, and the top 3 go to the LLM.
- **Pipeline 2 (re-ranking):** the same top 10 are re-scored by a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and the re-ranked top 3 go to the LLM with the identical prompt.

The LLM is Google Gemini (`gemini-3.5-flash`, temperature 0, thinking disabled).

**Document:** *Twilight Imperium: Fourth Edition, Living Rules Reference v1.3* (Fantasy Flight Games, 38 pages).

Everything is in a single notebook: [`rag_rerank.ipynb`](rag_rerank.ipynb). It is committed with its outputs, so the results can be read without running it.

## Repository layout

| Path | Contents |
|---|---|
| `rag_rerank.ipynb` | Full pipeline: chunking, indexing, both pipelines, stress tests, LLM answers, latency, scaling |
| `results/chunks.json` | The 225 chunks shared by both pipelines |
| `results/rank_shifts.csv` | Bi-encoder vs. cross-encoder rank of the gold chunk for all 77 stress-test queries |
| `results/answers.json` | Answer 1 vs. Answer 2 for the failure cases |
| `results/latency.csv`, `results/latency_totals.csv` | Per-stage and per-pipeline latency |
| `results/scaling.csv` | Cross-encoder on top 10 vs. the whole corpus, with extrapolation |
| `results/llm_cache.json` | Cached Gemini responses, so a re-run reproduces the same answers without API calls |

## Running it

1. Download the rulebook PDF (the official free file from Fantasy Flight Games) and save it as `data/ti4_living_rules_reference_v1_3.pdf`:
   <https://images-cdn.fantasyflightgames.com/filer_public/da/df/dadf9f07-78f3-43ac-9cce-dd6b55b24ec2/ti4_living_rules_reference_v1_3_web.pdf>
2. Set up the environment:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA build
   pip install -r requirements.txt
   echo "GEMINI_API_KEY=your-key" > .env                                # key from https://aistudio.google.com
   ```
3. Open `rag_rerank.ipynb` and choose the `.venv` kernel. In VS Code, use **Select Kernel → Python Environments → .venv**. Then run all cells.

The two models download from Hugging Face on first run, about 90 MB each.
