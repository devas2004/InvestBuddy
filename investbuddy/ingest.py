"""
Fetch SEC filings from EDGAR, parse into section-aware chunks,
embed with Voyage AI, and persist in Chroma.

Usage:
    python -m investbuddy.cli ingest
"""

import io
import os
import re
import time
import uuid
import logging
import warnings
from itertools import groupby
from pathlib import Path
from typing import Optional

import requests
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
from dotenv import load_dotenv
import voyageai
import chromadb

load_dotenv()

log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
FILINGS_DIR = DATA_DIR / "filings"
CHROMA_DIR = DATA_DIR / "chroma"

# ── EDGAR endpoints and fair-access config ─────────────────────────────────────
EDGAR_SUBMISSIONS = "https://data.sec.gov/submissions"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

# EDGAR requires a descriptive User-Agent identifying the application and contact.
_HEADERS = {
    "User-Agent": "InvestBuddy-Portfolio-RAG research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# ── Target companies ───────────────────────────────────────────────────────────
COMPANIES: dict[str, dict] = {
    "GOOGL": {"name": "Alphabet Inc.",          "cik": "0001652044"},
    "MSFT":  {"name": "Microsoft Corporation",   "cik": "0000789019"},
    "AMZN":  {"name": "Amazon.com Inc.",         "cik": "0001018724"},
}

FORM_TYPES = ["10-K", "10-Q"]
MAX_FILINGS = 4  # most-recent filings per company per form type

# ── Chunking parameters ────────────────────────────────────────────────────────
# ~600 tokens at ≈4 chars/token; 10% overlap
CHUNK_CHARS = 2400
OVERLAP_CHARS = 240

# ── Voyage AI ─────────────────────────────────────────────────────────────────
VOYAGE_MODEL = "voyage-finance-2"
VOYAGE_BATCH = 64  # well under the 128-doc per-request limit

# ── Chroma ────────────────────────────────────────────────────────────────────
CHROMA_COLLECTION = "sec_filings"

# ── Section heading patterns ───────────────────────────────────────────────────
# More-specific sub-items (1A, 7A, 9A, 9B) are listed before their parent
# items (1, 7, 9) so the regex alternation finds the right label first.
_SECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bitem\s+1a\b", re.I), "Item 1A: Risk Factors"),
    (re.compile(r"\bitem\s+1b\b", re.I), "Item 1B: Unresolved Staff Comments"),
    (re.compile(r"\bitem\s+7a\b", re.I), "Item 7A: Market Risk"),
    (re.compile(r"\bitem\s+9a\b", re.I), "Item 9A: Controls and Procedures"),
    (re.compile(r"\bitem\s+9b\b", re.I), "Item 9B: Other Information"),
    (re.compile(r"\bitem\s+1\b",  re.I), "Item 1: Business"),
    (re.compile(r"\bitem\s+2\b",  re.I), "Item 2: Properties"),
    (re.compile(r"\bitem\s+3\b",  re.I), "Item 3: Legal Proceedings"),
    (re.compile(r"\bitem\s+4\b",  re.I), "Item 4: Mine Safety"),
    (re.compile(r"\bitem\s+5\b",  re.I), "Item 5: Market for Equity"),
    (re.compile(r"\bitem\s+6\b",  re.I), "Item 6: Selected Financial Data"),
    (re.compile(r"\bitem\s+7\b",  re.I), "Item 7: MD&A"),
    (re.compile(r"\bitem\s+8\b",  re.I), "Item 8: Financial Statements"),
    (re.compile(r"\bitem\s+9\b",  re.I), "Item 9: Changes in Disagreements"),
]

# Regex that matches the unique table-replacement markers we inject during parsing.
_TABLE_MARKER_RE = re.compile(r"__TBL[0-9a-f]{16}__")


# ── Lazy singletons ────────────────────────────────────────────────────────────

_voyage_client: Optional[voyageai.Client] = None
_chroma_collection: Optional[chromadb.Collection] = None


def _get_voyage() -> voyageai.Client:
    global _voyage_client
    if _voyage_client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set — check your .env file")
        _voyage_client = voyageai.Client(api_key=api_key)
    return _voyage_client


def _get_collection() -> chromadb.Collection:
    global _chroma_collection
    if _chroma_collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _chroma_collection = client.get_or_create_collection(CHROMA_COLLECTION)
    return _chroma_collection


# ── EDGAR fetch helpers ────────────────────────────────────────────────────────

def _get(url: str) -> requests.Response:
    """Rate-limited GET that stays within EDGAR's ≤10 req/s fair-access limit."""
    time.sleep(0.12)  # ≈8 req/s
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    return resp


def _collect_filings(block: dict, form_type: str, results: list[dict], cap: int) -> None:
    """Append matching filings from one EDGAR submissions JSON block into results."""
    forms      = block.get("form", [])
    accessions = block.get("accessionNumber", [])
    dates      = block.get("filingDate", [])
    reports    = block.get("reportDate", [])
    docs       = block.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if len(results) >= cap:
            return
        if form == form_type:
            results.append({
                "accession":        accessions[i],
                "filing_date":      dates[i],
                "report_date":      reports[i] if i < len(reports) else "",
                "primary_document": docs[i]    if i < len(docs)    else "",
            })


def fetch_filing_list(cik: str, form_type: str) -> list[dict]:
    """
    Return up to MAX_FILINGS recent filings of form_type for the given CIK.
    Paginates through EDGAR's submissions files only if the recent block is short.
    """
    url = f"{EDGAR_SUBMISSIONS}/CIK{cik}.json"
    log.info("Fetching submission index: CIK %s", cik)
    data = _get(url).json()

    results: list[dict] = []
    _collect_filings(data["filings"]["recent"], form_type, results, MAX_FILINGS)

    for extra in data["filings"].get("files", []):
        if len(results) >= MAX_FILINGS:
            break
        log.info("Fetching additional submissions page: %s", extra["name"])
        extra_data = _get(f"{EDGAR_SUBMISSIONS}/{extra['name']}").json()
        _collect_filings(extra_data, form_type, results, MAX_FILINGS)

    log.info("Found %d %s filings for CIK %s", len(results), form_type, cik)
    return results


def download_filing(cik: str, ticker: str, form_type: str, filing: dict) -> Optional[Path]:
    """
    Download the filing's primary document as HTML.
    Returns the local cache path, or None if the download fails.
    Re-uses cached files on subsequent runs.
    """
    doc = filing.get("primary_document", "")
    if not doc:
        log.warning("No primary document listed for %s %s %s", ticker, form_type, filing["filing_date"])
        return None

    accession_nodash = filing["accession"].replace("-", "")
    cik_int = str(int(cik))  # EDGAR archive URLs use un-padded CIK
    url = f"{EDGAR_ARCHIVE}/{cik_int}/{accession_nodash}/{doc}"

    safe_form = form_type.replace("-", "")
    cache_name = f"{ticker}_{safe_form}_{filing['filing_date']}_{accession_nodash}.html"
    cache_path = FILINGS_DIR / cache_name

    if cache_path.exists():
        log.info("Cache hit: %s", cache_name)
        return cache_path

    log.info("Downloading %s %s filed %s", ticker, form_type, filing["filing_date"])
    try:
        resp = _get(url)
    except requests.HTTPError as exc:
        log.warning("HTTP %s for %s — skipping", exc.response.status_code, url)
        return None

    cache_path.write_bytes(resp.content)
    log.info("Saved %s (%.0f KB)", cache_name, len(resp.content) / 1024)
    return cache_path


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _table_to_text(table_tag) -> str:
    """Convert a <table> element to a readable string. Falls back to raw text."""
    try:
        dfs = pd.read_html(io.StringIO(str(table_tag)))
        if dfs:
            return "\n".join(df.to_string(index=False) for df in dfs)
    except Exception:
        pass
    return table_tag.get_text(" | ", strip=True)


def _detect_section(line: str) -> Optional[str]:
    """Return a section label if line looks like a section heading, else None."""
    for pattern, label in _SECTION_PATTERNS:
        if pattern.search(line):
            return label
    return None


def parse_filing(html_path: Path) -> list[dict]:
    """
    Parse a filing HTML into a list of content blocks:
        [{"section": str, "text": str, "is_table": bool}, ...]

    Section labels are detected with best-effort regex on short text lines.
    All content that cannot be labelled falls under section="unknown" rather
    than being discarded. Tables are extracted atomically via pandas.read_html.

    The detection log line shows how many named sections were found per filing.
    """
    raw = html_path.read_bytes()
    soup = BeautifulSoup(raw, "lxml")
    for tag in soup(["script", "style", "head"]):
        tag.decompose()

    # Replace every <table> with a UUID-based marker before text extraction.
    # This lets us keep table content atomic (never split mid-table) while
    # still using BeautifulSoup's get_text() for everything else.
    table_store: dict[str, str] = {}
    for table in soup.find_all("table"):
        marker = f"__TBL{uuid.uuid4().hex[:16]}__"
        table_store[marker] = _table_to_text(table)
        table.replace_with(marker)

    body = soup.find("body") or soup
    lines = body.get_text("\n").split("\n")

    current_section = "unknown"
    blocks: list[dict] = []
    sections_detected = 0

    for line in lines:
        # Strip out any table markers to get the "clean" text of this line,
        # then check whether it is a section heading (headings are short).
        clean = _TABLE_MARKER_RE.sub("", line).strip()

        if 0 < len(clean) <= 150:
            detected = _detect_section(clean)
            if detected and detected != current_section:
                current_section = detected
                sections_detected += 1
                # Emit any tables embedded on the same line as the heading.
                for marker in _TABLE_MARKER_RE.findall(line):
                    t = table_store.get(marker, "").strip()
                    if t:
                        blocks.append({"section": current_section, "text": t, "is_table": True})
                continue  # don't emit the heading text itself as a block

        # Emit text parts and table markers in document order.
        parts   = _TABLE_MARKER_RE.split(line)
        markers = _TABLE_MARKER_RE.findall(line)

        for i, part in enumerate(parts):
            if part.strip():
                blocks.append({"section": current_section, "text": part.strip(), "is_table": False})
            if i < len(markers):
                t = table_store.get(markers[i], "").strip()
                if t:
                    blocks.append({"section": current_section, "text": t, "is_table": True})

    known = {b["section"] for b in blocks if b["section"] != "unknown"}
    unknown_count = sum(1 for b in blocks if b["section"] == "unknown")
    log.info(
        "%s → %d named sections detected (%s) | %d blocks labelled 'unknown'",
        html_path.name,
        len(known),
        ", ".join(sorted(known)) or "none",
        unknown_count,
    )
    return blocks


# ── Chunking ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def chunk_blocks(blocks: list[dict]) -> list[dict]:
    """
    Merge small text blocks and split large ones into CHUNK_CHARS-sized pieces
    with OVERLAP_CHARS overlap. Tables are always kept as atomic single chunks.

    Returns: [{"section": str, "text": str, "chunk_index": int}, ...]
    """
    result: list[dict] = []
    # Track the running chunk count per section name across ALL groupby groups.
    # A section like "Item 8" can appear multiple times (TOC + body), and each
    # appearance creates a new groupby group — without this counter, chunk_index
    # would reset to 0 each time, producing duplicate IDs.
    section_counters: dict[str, int] = {}

    for section, grp in groupby(blocks, key=lambda b: b["section"]):
        texts = _chunk_section(list(grp))
        start = section_counters.get(section, 0)
        for i, text in enumerate(texts):
            result.append({"section": section, "text": text, "chunk_index": start + i})
        section_counters[section] = start + len(texts)

    return result


def _chunk_section(blocks: list[dict]) -> list[str]:
    """Produce target-size text strings from one section's worth of blocks."""
    results: list[str] = []
    buffer = ""

    for block in blocks:
        text = block["text"]

        if block["is_table"]:
            # Flush accumulated text before emitting the table atomically.
            if buffer.strip():
                results.append(buffer.strip())
                buffer = ""
            results.append(text)  # table is its own chunk regardless of size
        else:
            if buffer and len(buffer) + 1 + len(text) > CHUNK_CHARS:
                results.append(buffer.strip())
                # Seed the next chunk with a trailing overlap window.
                overlap = buffer[-OVERLAP_CHARS:].lstrip()
                buffer = (overlap + "\n" + text).lstrip()
            else:
                buffer = (buffer + "\n" + text).lstrip()

    if buffer.strip():
        results.append(buffer.strip())

    return results


# ── Embed & store ──────────────────────────────────────────────────────────────

def _make_chunk_id(ticker: str, form: str, filing_date: str, section: str, idx: int) -> str:
    safe_form    = form.replace("-", "")
    section_slug = _slugify(section)[:40]
    return f"{ticker}_{safe_form}_{filing_date}_{section_slug}_{idx}"


def embed_and_store(
    chunks: list[dict],
    ticker: str,
    company_name: str,
    form: str,
    filing_date: str,
    fiscal_period: str,
    source_url: str,
) -> int:
    """
    Embed new chunks with Voyage AI and upsert them into Chroma.
    Chunks whose IDs are already in Chroma are skipped to avoid redundant
    API calls on re-runs.

    Returns the count of newly embedded and stored chunks.
    """
    if not chunks:
        return 0

    collection = _get_collection()
    voyage = _get_voyage()

    chunk_ids = [
        _make_chunk_id(ticker, form, filing_date, c["section"], c["chunk_index"])
        for c in chunks
    ]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "chunk_id":      cid,
            "company":       company_name,
            "ticker":        ticker,
            "form":          form,
            "filing_date":   filing_date,
            "fiscal_period": fiscal_period,
            "section":       c["section"],
            "source_url":    source_url,
        }
        for c, cid in zip(chunks, chunk_ids)
    ]

    # Identify which IDs are already stored to avoid re-embedding.
    existing = collection.get(ids=chunk_ids, include=[])
    existing_ids = set(existing["ids"])
    new_idx = [i for i, cid in enumerate(chunk_ids) if cid not in existing_ids]

    if not new_idx:
        log.info("All %d chunks already indexed — skipping", len(chunks))
        return 0

    log.info(
        "Embedding %d new chunks (%d already exist)",
        len(new_idx), len(chunks) - len(new_idx),
    )

    new_texts  = [texts[i]     for i in new_idx]
    new_ids    = [chunk_ids[i] for i in new_idx]
    new_metas  = [metadatas[i] for i in new_idx]

    # Embed in batches, with a brief pause between batches to respect rate limits.
    all_embeddings: list[list[float]] = []
    for start in range(0, len(new_texts), VOYAGE_BATCH):
        batch = new_texts[start : start + VOYAGE_BATCH]
        result = voyage.embed(batch, model=VOYAGE_MODEL, input_type="document")
        all_embeddings.extend(result.embeddings)
        if start + VOYAGE_BATCH < len(new_texts):
            time.sleep(0.5)

    collection.add(
        ids=new_ids,
        documents=new_texts,
        embeddings=all_embeddings,
        metadatas=new_metas,
    )
    log.info("Stored %d chunks in Chroma", len(new_ids))
    return len(new_ids)


# ── Orchestration ──────────────────────────────────────────────────────────────

def run_ingest() -> None:
    """Fetch, parse, chunk, embed, and index all configured filings."""
    FILINGS_DIR.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    total_new = 0

    for ticker, company in COMPANIES.items():
        cik  = company["cik"]
        name = company["name"]

        for form_type in FORM_TYPES:
            filings = fetch_filing_list(cik, form_type)

            for filing in filings:
                html_path = download_filing(cik, ticker, form_type, filing)
                if html_path is None:
                    continue

                blocks = parse_filing(html_path)
                if not blocks:
                    log.warning("No content parsed from %s", html_path.name)
                    continue

                chunks = chunk_blocks(blocks)
                log.info(
                    "%s %s %s → %d chunks from %d blocks",
                    ticker, form_type, filing["filing_date"], len(chunks), len(blocks),
                )

                accession_nodash = filing["accession"].replace("-", "")
                cik_int = str(int(cik))
                source_url = (
                    f"{EDGAR_ARCHIVE}/{cik_int}/{accession_nodash}"
                    f"/{filing['primary_document']}"
                )

                new_count = embed_and_store(
                    chunks=chunks,
                    ticker=ticker,
                    company_name=name,
                    form=form_type,
                    filing_date=filing["filing_date"],
                    fiscal_period=filing["report_date"],
                    source_url=source_url,
                )
                total_chunks += len(chunks)
                total_new += new_count

    log.info(
        "Ingest complete — %d total chunks across all filings, %d newly stored",
        total_chunks, total_new,
    )
