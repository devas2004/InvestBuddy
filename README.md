# InvestBuddy — SEC Filings RAG

A Retrieval-Augmented Generation (RAG) system that indexes SEC 10-K and 10-Q
filings for **Alphabet (GOOGL)**, **Microsoft (MSFT)**, and **Amazon (AMZN)**,
then answers financial questions in plain language with citations back to the
specific filing and section each claim came from.

**Stack:** Voyage AI `voyage-finance-2` embeddings · Google Gemini 2.5 Flash ·
Chroma vector store · SEC EDGAR data API

---

## Setup

### 1. Prerequisites

- Python 3.11+
- API keys for [Voyage AI](https://www.voyageai.com/) and
  [Google AI Studio](https://aistudio.google.com/)

### 2. Install dependencies

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in VOYAGE_API_KEY and GEMINI_API_KEY
```

---

## Usage

### Ingest filings

Fetches the 4 most recent 10-K and 10-Q filings for each company from SEC
EDGAR, parses them into section-aware chunks (~600 tokens each), embeds the
chunks with Voyage AI, and persists them in a local Chroma vector store.

Raw HTML files are cached in `data/filings/`. Already-indexed chunk IDs are
skipped on subsequent runs, so re-running ingest after adding a company is
cheap.

```bash
python -m investbuddy.cli ingest
```

Watch the log output — it prints how many named sections were detected per
filing so you can see how well section detection worked.

### Ask a question

```bash
python -m investbuddy.cli ask "What are Alphabet's main sources of revenue?"
python -m investbuddy.cli ask "What risks does Amazon cite related to competition?" --k 8
python -m investbuddy.cli ask "How did Microsoft's cloud revenue trend over the past year?"
```

The answer is grounded exclusively in the indexed filings.  Every factual
claim is cited inline with a `[chunk_id]` tag.  The cited source metadata
(company, form type, filing date, section, EDGAR URL) is printed below the
answer so you can verify each claim.

---

## Project structure

```
investbuddy/
├── ingest.py    # EDGAR fetch → parse → chunk → embed → Chroma
├── llm.py       # Swappable LLM wrapper (default: Gemini 2.5 Flash)
├── answer.py    # Embed question → retrieve → generate grounded answer
└── cli.py       # CLI entry point (click)
data/
├── filings/     # Cached raw HTML downloads (gitignored)
└── chroma/      # Chroma vector store (gitignored)
requirements.txt
.env.example
```

---

## Swapping the LLM provider

`investbuddy/llm.py` is the **only** file that imports a provider SDK.
Re-implement `generate(system, user, max_tokens) -> str` there to use a
different provider (OpenAI, Anthropic, Ollama, etc.) without touching any
other module.

---

## Notes on section detection

SEC filing HTML is inconsistent.  InvestBuddy uses regex heuristics to detect
standard `Item X` section headings.  If a section cannot be confidently
identified, the content falls under `section="unknown"` rather than being
discarded — you'll see the count in the ingest log.  The table of contents
near the top of each filing may cause a few early false-positive section
labels; this is a known limitation of text-extraction-based parsing.
