"""
RAG query path: embed a question → retrieve top-k chunks from Chroma
→ generate a grounded, citation-backed answer via the LLM.

Usage:
    python -m investbuddy.cli ask "What are Microsoft's main revenue segments?"
"""

import os
import re
import logging
from typing import Optional

from dotenv import load_dotenv
import voyageai
import chromadb

from investbuddy.llm import generate

load_dotenv()

log = logging.getLogger(__name__)

_CHROMA_DIR_DEFAULT = "data/chroma"
_CHROMA_COLLECTION  = "sec_filings"
_VOYAGE_MODEL       = "voyage-finance-2"

_SYSTEM_PROMPT = """\
You are a financial analyst assistant working exclusively with SEC filings.

Rules you must follow without exception:
1. Answer ONLY using information from the CONTEXT CHUNKS provided below.
   Do not rely on any prior knowledge about these companies.
2. After every factual claim, immediately cite the source chunk by appending
   its identifier in brackets, e.g. "Revenue grew 12% [GOOGL_10K_2024-01-30_item_7_mda_0]."
3. If a chunk supports a claim, cite it — do not omit citations.
4. If the provided context does not contain enough information to answer the
   question, say exactly: "The provided filings do not contain sufficient
   information to answer this question."
5. Be concise and precise. Use plain language.
"""

_voyage: Optional[voyageai.Client] = None
_collection: Optional[chromadb.Collection] = None


def _get_voyage() -> voyageai.Client:
    global _voyage
    if _voyage is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set — check your .env file")
        _voyage = voyageai.Client(api_key=api_key)
    return _voyage


def _get_collection(chroma_dir: str) -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=chroma_dir)
        _collection = client.get_or_create_collection(_CHROMA_COLLECTION)
    return _collection


def _build_context(docs: list[str], metas: list[dict]) -> str:
    """
    Format retrieved chunks as a labelled block for the LLM.
    Each chunk is prefixed with its [chunk_id] and key metadata so the model
    can cite it precisely.
    """
    parts = ["CONTEXT CHUNKS:"]
    for doc, meta in zip(docs, metas):
        header = (
            f"[{meta['chunk_id']}] "
            f"{meta['company']} | {meta['form']} | "
            f"Filed: {meta['filing_date']} | Period: {meta['fiscal_period']} | "
            f"Section: {meta['section']}"
        )
        parts.append(f"\n{header}\n{doc}")
    return "\n".join(parts)


def answer_question(
    question: str,
    k: int = 6,
    chroma_dir: str = _CHROMA_DIR_DEFAULT,
) -> dict:
    """
    Retrieve the top-k most relevant chunks and generate a grounded answer.

    Returns:
        {
            "answer":    str         — LLM response with inline [chunk_id] citations,
            "sources":   list[dict]  — metadata for each chunk cited in the answer,
            "retrieved": list[dict]  — metadata for all k retrieved chunks,
        }
    """
    # 1. Embed the question with the query input type.
    voyage = _get_voyage()
    embed_result = voyage.embed([question], model=_VOYAGE_MODEL, input_type="query")
    query_embedding = embed_result.embeddings[0]

    # 2. Retrieve from Chroma.
    collection = _get_collection(chroma_dir)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas"],
    )

    docs: list[str]  = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]

    if not docs:
        return {
            "answer": (
                "No filings have been indexed yet. "
                "Run `python -m investbuddy.cli ingest` first."
            ),
            "sources":   [],
            "retrieved": [],
        }

    log.info("Retrieved %d chunks for question: %r", len(docs), question)

    # 3. Build the context block and call the LLM.
    context = _build_context(docs, metas)
    user_message = f"Question: {question}\n\n{context}"
    answer = generate(_SYSTEM_PROMPT, user_message, max_tokens=1500)

    # 4. Parse [chunk_id] citations from the answer to identify cited sources.
    # Chunk IDs follow the pattern: TICKER_FORM_DATE_SECTION_INDEX
    cited_ids = set(re.findall(r"\[([A-Za-z0-9_\-]+)\]", answer))
    sources = [meta for meta in metas if meta.get("chunk_id") in cited_ids]

    return {
        "answer":    answer,
        "sources":   sources,
        "retrieved": metas,
    }
