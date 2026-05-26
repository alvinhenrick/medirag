---
title: Medirag
emoji: 🐨
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
short_description: Ask plain-language questions about your medication — grounded in FDA DailyMed labels.
---

<table>
    <tr>
        <td>
            <img src="doc/images/MediRag.png" alt="MediRAG" width="150"/>
        </td>
        <td>
            <h1>MediRAG</h1>
        </td>
    </tr>
</table>

**MediRAG** replaces the tiny-print patient information leaflet that comes with your medication.
Ask anything about a drug — side effects, ingredients, dosing, contraindications, drug
interactions, what the pill looks like — and get plain-language answers grounded in the FDA's
DailyMed structured product labels.

## Features

- **Patient-friendly Q&A**: Conversational answers, not regulatory wall-of-text.
- **Full SPL coverage**: Every section (narrative + structured product data) is indexed —
  ingredients with strengths, NDC codes, pill color/shape/imprint, manufacturer, route.
- **Hybrid retrieval**: Vector search (PubMedBERT) + BM25 full-text in one query, so brand
  names and ingredient names match exactly while paraphrased questions still hit.
- **Cross-drug queries**: Metadata-filtered search ("which other drugs contain
  atorvastatin?", "is there an IV form?") via ingredient UNII codes and SQL filters.
- **Input and output guardrails**: DSPy-based safety checks on both the user's question and
  the model's answer.
- **Streaming answers**: Token-level streaming via `dspy.streamify`.
- **Semantic cache**: Reuses answers for semantically similar queries.
- **Model picker**: GPT-4o-mini, GPT-4o.

## Architecture

```
DailyMed SPL zips
    │
    ▼  medirag.index.runner  (streaming per-part)
parse_spl  →  LanceDB  →  (optional) upload to HF Hub
    │           │
    │           ▼
    │      PubMedBERT (NeuML/pubmedbert-base-embeddings, 768d)
    │      Hybrid index: vector + BM25
    │
    ▼  app.py
DSPy module (input guard → retrieve → ChainOfThought → output guard)
    │
    ▼  dspy.streamify
Streamed answer in Gradio
```

**Key tech**:

- [DSPy](https://dspy.ai) — module composition, prompting, streaming, guardrails
- [LanceDB](https://lancedb.com) — embedded vector store with native embeddings + hybrid search
- [PubMedBERT](https://huggingface.co/NeuML/pubmedbert-base-embeddings) — biomedical embeddings
- [Gradio](https://gradio.app) — UI
- [DailyMed](https://dailymed.nlm.nih.gov) — FDA drug labels (free, public)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/alvinhenrick/medirag.git
   cd medirag
   ```

2. Install dependencies (uses [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv sync
   ```

3. Create a `.env` file with your model API key:

   ```bash
   OPENAI_API_KEY=...
   HF_TOKEN=...             # optional, only for publishing the index
   ```

4. Get the index. Either pull the prebuilt one from Hugging Face:

   ```bash
   HF_DATASET=alvinhenrick/medirag-dailymed uv run app.py
   # → app downloads lance_db/ from HF on first start
   ```

   …or build it yourself from DailyMed
   ([official release page](https://dailymed.nlm.nih.gov/dailymed/spl-resources-all-drug-labels.cfm)):

   ```bash
   # Quick smoke build: 100 SPLs from one part (~1-2 min)
   uv run python -m medirag.index.runner \
       --source path/to/dm_spl_release_human_rx_part1.zip \
       --db ./lance_db \
       --limit 100

   # Full single-part build (~30-60 min on a Mac)
   uv run python -m medirag.index.runner \
       --source path/to/dm_spl_release_human_rx_part1.zip \
       --db ./lance_db

   # All 6 parts from official URLs (streams downloads, peak disk ~5 GB)
   uv run python -m medirag.index.runner --all --db ./lance_db
   ```

5. Run the app:

   ```bash
   LANCE_DB_PATH=./lance_db uv run app.py
   ```

   Open the URL printed by Gradio, pick a model, ask a question.

## Publishing the index to Hugging Face

The built index is a self-contained directory (`lance_db/`). To publish so other instances
of the app can `snapshot_download` it:

```bash
uv run python -m medirag.index.runner --all --db ./lance_db \
    --publish-repo alvinhenrick/medirag-dailymed
```

Set `HF_TOKEN` in your env or pass `--publish-token`. Consumers point at it via
`HF_DATASET=alvinhenrick/medirag-dailymed`.

## Testing

```bash
uv run pytest tests/
```

22 tests covering:

- SPL XML extraction (`tests/core/test_xml_reader.py`)
- LanceDB indexer + retrieval scenarios (`tests/index/test_lance.py`)
- End-to-end runner against a synthetic zip (`tests/index/test_runner.py`)
- Semantic cache (`tests/cache/test_semantic_cache.py`)

Tests run on the sample SPL XML in `tests/data/` — no DailyMed download needed.

## Project Layout

```
medirag/
├── core/
│   └── reader.py        # SPL XML → ProductCard + SectionRecord dataclasses
├── index/
│   ├── lance.py         # LanceIndexer (PubMedBERT + LanceDB + hybrid search)
│   └── runner.py        # CLI to build/publish the index from DailyMed zips
├── rag/
│   ├── dspy.py          # DspyRAG module + DailyMedRetrieve + stream_answer
│   └── qa_rag.py        # Cache + streaming pipeline
├── cache/
│   └── local.py         # Semantic cache (numpy-backed)
└── guardrail/
    ├── input.py         # Block harmful/off-topic inputs
    └── output.py        # Block unsafe model outputs

app.py                   # Gradio app
tests/                   # Unit + integration tests
```

## Roadmap

- [ ] Index all 6 DailyMed parts and publish to HF Hub
- [ ] Daily/weekly automatic rebuild from DailyMed update files
- [ ] LLM evaluation harness on a curated patient-question benchmark
- [ ] Optional reranker for top-k results
- [ ] OpenTelemetry traces for retrieval + LM calls

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
