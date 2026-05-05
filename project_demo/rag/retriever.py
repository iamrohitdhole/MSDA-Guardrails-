"""
Evidence Retriever for RAG Guardrail System

Handles:
- Chunking drug records into 200-token snippets
- Building BM25 index over chunks
- Retrieving top-k evidence snippets for queries
"""

import pickle
import re
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel

from project_demo.rag import (
    EVIDENCE_INDEX_PATH,
    CHUNK_SIZE_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_FIELDS,
    BM25_K1,
    BM25_B,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Simple tokenizer (matches eval_llm_metrics.py style)
TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)


def clean_markup(text: str) -> str:
    """Strip markdown and HTML markup so tokens like _Aspirin_ become aspirin."""
    import re as _re
    text = _re.sub(r'<[^>]+>', ' ', text)          # HTML tags
    text = _re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # [text](url)
    text = _re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)  # **bold** / *italic*
    text = _re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)    # _italic_ / __bold__
    text = _re.sub(r'`[^`]+`', ' ', text)           # `code`
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words."""
    return TOKEN_RE.findall(clean_markup(text).lower())


def count_tokens(text: str) -> int:
    """Count tokens in text."""
    return len(tokenize(text))


def chunk_text(text: str, max_tokens: int = CHUNK_SIZE_TOKENS, overlap: int = CHUNK_OVERLAP_TOKENS) -> List[str]:
    """
    Split text into chunks of approximately max_tokens.
    Uses sentence boundaries where possible, falls back to word boundaries.

    Returns list of chunk strings.
    """
    if not text or len(text.strip()) < 50:
        return []

    # Split into sentences (simple heuristic: .!? followed by space or end)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = count_tokens(sentence)

        if current_tokens + sent_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sent_tokens
        else:
            # Save current chunk if non-empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                # Take last few sentences for overlap
                overlap_sents = []
                overlap_count = 0
                for s in reversed(current_chunk):
                    s_tokens = count_tokens(s)
                    if overlap_count + s_tokens <= overlap:
                        overlap_sents.insert(0, s)
                        overlap_count += s_tokens
                    else:
                        break
                current_chunk = overlap_sents
                current_tokens = overlap_count
            else:
                current_chunk = []
                current_tokens = 0

            current_chunk.append(sentence)
            current_tokens += sent_tokens

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def normalize_drug_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Align DQ/silver column names with CHUNK_FIELDS expectations.
    Maps alternate column names onto canonical keys when the canonical field is empty.
    """
    out = dict(record)

    def _fill(canonical: str, *alts: str) -> None:
        cur = out.get(canonical)
        if cur is not None and str(cur).strip():
            return
        for a in alts:
            v = out.get(a)
            if v is not None and str(v).strip():
                out[canonical] = v
                return

    _fill("mechanism_of_action", "mechanism", "mechanism_of_action_text")
    _fill("drug_interactions", "interactions", "drug_interaction", "interactions_text")
    _fill("food_interactions", "food_interaction", "food_interactions_text")
    _fill("pharmacodynamics", "pharmacodynamic", "pharmacodynamics_text")
    _fill("toxicity", "toxicities", "toxicity_text")
    _fill("indication", "indications", "indication_text")
    _fill("description", "general_description", "description_text")
    _fill("absorption", "absorption_text")
    _fill("half_life", "halflife", "half_life_text")
    _fill("metabolism", "metabolism_text")
    _fill("protein_binding", "protein_binding_text")
    _fill("route_of_elimination", "elimination", "route_of_elimination_text")

    return out


def chunk_record(record: Dict[str, Any], drug_id: str, drug_name: str) -> List[Dict[str, Any]]:
    """
    Chunk a single drug record into snippets with metadata.

    Returns list of chunk dicts with:
    - drug_id, drug_name, field_name, chunk_idx, chunk_text
    """
    chunks = []

    for field_name in CHUNK_FIELDS:
        text = record.get(field_name)
        if not text:
            continue

        # Strip markup before chunking so stored text is clean
        text_str = clean_markup(str(text))
        field_chunks = chunk_text(text_str)

        for idx, text_chunk in enumerate(field_chunks):
            chunks.append({
                "drug_id": drug_id,
                "drug_name": drug_name,
                "field_name": field_name,
                "chunk_idx": idx,
                "chunk_text": text_chunk,
                "token_count": count_tokens(text_chunk),
            })

    return chunks


class BM25Index:
    """
    Vectorized BM25 over a sparse term-frequency matrix.

    Scoring:
        bm25(d, q) = sum_{t in q} IDF(t) * (tf(t, d) * (k1 + 1))
                                          / (tf(t, d) + k1 * (1 - b + b * dl(d) / avgdl))

    The implementation:
      - CountVectorizer builds a CSR (n_docs × n_terms) matrix of raw term
        frequencies (the previous version used TF-IDF values as `tf`, which
        mixes formulas — we now use raw TF as BM25 specifies).
      - IDF uses the BM25 form: log((N - df + 0.5) / (df + 0.5) + 1).
      - At query time we slice only the columns for query terms, convert
        that small block (n_docs × |Q|) to dense, broadcast the per-doc length
        norm, and reduce to a length-N_docs score vector via numpy.
      - Top-k via np.argpartition (O(N), no full sort).

    The score() method also takes optional `boost_indices` so the
    EvidenceRetriever can lift drug-name-matched chunks before top-k cuts in.
    """

    FORMAT_VERSION = 2

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.vectorizer: Optional[CountVectorizer] = None
        self.tf_matrix: Optional[sparse.csr_matrix] = None
        self.idf: Optional[np.ndarray] = None
        self.doc_lengths: Optional[np.ndarray] = None
        self.avg_doc_length: float = 0.0
        self._dl_norm: Optional[np.ndarray] = None  # cached: 1 - b + b * dl/avgdl
        self.corpus: List[str] = []

    # ---- Build ---------------------------------------------------------------

    def fit(self, corpus: List[str]) -> "BM25Index":
        self.corpus = corpus
        self.doc_lengths = np.array(
            [count_tokens(doc) for doc in corpus], dtype=np.float32
        )
        self.avg_doc_length = (
            float(self.doc_lengths.mean()) if self.doc_lengths.size else 1.0
        )

        # Raw term frequencies. min_df=2 trims hapaxes; max_df=0.85 trims very
        # common terms (was 0.95) to reduce common-word noise at this scale.
        self.vectorizer = CountVectorizer(
            tokenizer=tokenize,
            token_pattern=None,
            lowercase=False,
            min_df=2,
            max_df=0.85,
            dtype=np.float32,
        )
        self.tf_matrix = self.vectorizer.fit_transform(corpus).tocsr()

        # BM25 IDF
        df = np.asarray((self.tf_matrix > 0).sum(axis=0)).ravel().astype(np.float32)
        N = float(self.tf_matrix.shape[0])
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

        # Per-doc length-norm factor (precomputed once)
        avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
        self._dl_norm = (
            (1.0 - self.b) + self.b * (self.doc_lengths / avgdl)
        ).astype(np.float32)

        logger.info(
            "Built BM25 index: %d docs, %d vocab terms, avg_doc_len=%.1f",
            self.tf_matrix.shape[0], self.tf_matrix.shape[1], self.avg_doc_length,
        )
        return self

    # ---- Score ---------------------------------------------------------------

    def _ensure_dl_norm(self) -> None:
        if self._dl_norm is None and self.doc_lengths is not None:
            avgdl = self.avg_doc_length if self.avg_doc_length > 0 else 1.0
            self._dl_norm = (
                (1.0 - self.b) + self.b * (self.doc_lengths / avgdl)
            ).astype(np.float32)

    def score(
        self,
        query: str,
        top_k: int = 5,
        boost_indices: Optional[np.ndarray] = None,
        boost_factor: float = 1.0,
    ) -> List[Tuple[int, float]]:
        if self.tf_matrix is None or self.vectorizer is None or self.idf is None:
            return []
        self._ensure_dl_norm()

        q_terms = set(tokenize(query))
        vocab = self.vectorizer.vocabulary_
        term_cols = [vocab[t] for t in q_terms if t in vocab]
        if not term_cols:
            return []

        # Slice only the columns we need: (n_docs × |Q|) sparse block.
        tf_q = self.tf_matrix[:, term_cols]
        # Densify the small block — typically (317k × ~5) ≈ 6 MB.
        tf_q_dense = tf_q.toarray()  # (n_docs, Q)

        idf_q = self.idf[term_cols]                    # (Q,)
        denom = tf_q_dense + self.k1 * self._dl_norm[:, None]   # (n_docs, Q)
        # tf=0 → numerator is 0, so the per-cell score is 0 regardless of denom.
        numer = tf_q_dense * (self.k1 + 1.0)
        # Safe divide: denom is always > 0 because dl_norm > 0 when avgdl > 0.
        per_term = (numer / denom) * idf_q[None, :]
        scores = per_term.sum(axis=1)                  # (n_docs,)

        if boost_indices is not None and boost_factor != 1.0 and len(boost_indices):
            # Boost ONLY chunks that already had a non-zero BM25 score, so that
            # an irrelevant chunk of a name-matched drug doesn't outrank a
            # relevant chunk of the right drug.
            mask = scores[boost_indices] > 0
            if mask.any():
                hot = boost_indices[mask]
                scores[hot] = scores[hot] * boost_factor

        # Top-k via argpartition (O(N))
        n = scores.shape[0]
        k = min(top_k, n)
        if k == 0:
            return []
        top_idx = np.argpartition(-scores, k - 1)[:k]
        # Order the survivors
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(int(i), float(scores[i])) for i in top_idx if scores[i] > 0]


class EvidenceRetriever:
    """
    Main retriever class for RAG guardrail system.

    Usage:
        retriever = EvidenceRetriever()
        retriever.build_index(drug_records)
        retriever.save_index(path)

        # Later...
        retriever.load_index(path)
        snippets = retriever.retrieve_evidence(query, top_k=5)
    """

    # Drug-name boost: how much to multiply BM25 score for chunks whose drug
    # is named (or synonym-matched) in the query. Empirically a 2.5–3× lift
    # is enough to surface the right drug without drowning out genuinely
    # relevant cross-drug chunks (e.g. drug-interaction text).
    DRUG_NAME_BOOST = 3.0

    # Minimum length for a name to be considered for boost matching. Avoids
    # spurious matches on short names that are also common English words
    # (e.g. "ace", "ice", "ash").
    MIN_NAME_LEN_FOR_BOOST = 4

    def __init__(self):
        self.bm25_index: Optional[BM25Index] = None
        self.chunk_metadata: List[Dict[str, Any]] = []
        self.corpus: List[str] = []
        # name_lower → np.array of chunk indices that belong to that drug.
        # Populated on every build/load so retrieval works without a parquet.
        self._name_to_chunks: Dict[str, np.ndarray] = {}
        # Reverse map drug_id → drug_name (for synonym lookups when we add
        # synonyms loaded from a parquet).
        self._drug_id_to_name: Dict[str, str] = {}

    # ── Index assembly ────────────────────────────────────────────────────────

    def build_index(self, drug_records: List[Dict[str, Any]]) -> None:
        """Build BM25 index + drug-name index from drug records."""
        all_chunks = []
        self.chunk_metadata = []

        logger.info("Building index from %d drug records...", len(drug_records))

        # Capture synonyms while we have the records in hand — we won't have
        # them after the index is loaded from disk later.
        synonyms_by_id: Dict[str, List[str]] = {}

        for record in drug_records:
            record = normalize_drug_record(record)
            drug_id = (
                record.get("drugbank_id")
                or record.get("id")
                or record.get("drug_id", "unknown")
            )
            drug_name = record.get("name") or record.get("drug_name", "Unknown")
            syns = record.get("synonyms")
            if syns is None:
                syns_iter: List[Any] = []
            elif isinstance(syns, np.ndarray):
                syns_iter = syns.tolist()
            elif isinstance(syns, (list, tuple)):
                syns_iter = list(syns)
            else:
                syns_iter = []
            if syns_iter:
                synonyms_by_id[str(drug_id)] = [str(s) for s in syns_iter if s]

            record_chunks = chunk_record(record, str(drug_id), drug_name)
            all_chunks.extend(record_chunks)
            self.chunk_metadata.extend(record_chunks)

        self.corpus = [c["chunk_text"] for c in self.chunk_metadata]
        if not self.corpus:
            raise ValueError("No chunks generated - check input data")

        self.bm25_index = BM25Index()
        self.bm25_index.fit(self.corpus)

        # Stash synonyms on the metadata for the first chunk of each drug, so
        # they survive the pickle round-trip.
        seen_drug: set = set()
        for ch in self.chunk_metadata:
            d_id = ch.get("drug_id")
            if d_id and d_id not in seen_drug:
                ch["_synonyms"] = synonyms_by_id.get(str(d_id), [])
                seen_drug.add(d_id)

        self._build_name_index()
        logger.info(
            "Index built: %d chunks, %d distinct drug names indexed for boost",
            len(self.corpus), len(self._name_to_chunks),
        )

    def _build_name_index(self) -> None:
        """
        Build a {lowercase_name → chunk_indices} map covering each drug's
        canonical name and any synonyms stashed on its first chunk.
        """
        name_to_idxs: Dict[str, List[int]] = defaultdict(list)
        id_to_name: Dict[str, str] = {}
        # Collect synonyms keyed by drug_id (only first occurrence carries them)
        synonyms_by_id: Dict[str, List[str]] = {}
        for ch in self.chunk_metadata:
            d_id = ch.get("drug_id")
            if d_id and "_synonyms" in ch and d_id not in synonyms_by_id:
                synonyms_by_id[d_id] = list(ch.get("_synonyms") or [])

        for idx, ch in enumerate(self.chunk_metadata):
            name = (ch.get("drug_name") or "").strip().lower()
            d_id = ch.get("drug_id")
            if d_id and d_id not in id_to_name and name:
                id_to_name[d_id] = name
            if name and len(name) >= self.MIN_NAME_LEN_FOR_BOOST:
                name_to_idxs[name].append(idx)

            # Index synonyms against this same drug's chunks
            for syn in synonyms_by_id.get(d_id, []):
                s = (syn or "").strip().lower()
                if s and len(s) >= self.MIN_NAME_LEN_FOR_BOOST:
                    name_to_idxs[s].append(idx)

        self._name_to_chunks = {
            n: np.array(sorted(set(v)), dtype=np.int64) for n, v in name_to_idxs.items()
        }
        self._drug_id_to_name = id_to_name

    # ── Query path ────────────────────────────────────────────────────────────

    def _detect_drug_chunks(self, query: str) -> Optional[np.ndarray]:
        """
        Find chunk indices for any drug whose name or synonym appears in the
        query. Substring match is enough for medical queries — drug names are
        rare strings, false positives are unlikely above MIN_NAME_LEN_FOR_BOOST.
        """
        if not self._name_to_chunks:
            return None
        q_lower = query.lower()
        hits: List[np.ndarray] = []
        for name, idxs in self._name_to_chunks.items():
            if name in q_lower:
                hits.append(idxs)
        if not hits:
            return None
        if len(hits) == 1:
            return hits[0]
        return np.unique(np.concatenate(hits))

    def retrieve_evidence(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k evidence snippets for a query (with drug-name boost)."""
        if not self.bm25_index or not self.corpus:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        boost_idx = self._detect_drug_chunks(query)
        scored = self.bm25_index.score(
            query,
            top_k=top_k,
            boost_indices=boost_idx,
            boost_factor=self.DRUG_NAME_BOOST,
        )

        results = []
        for doc_idx, score in scored:
            if doc_idx < len(self.chunk_metadata):
                chunk_info = {
                    k: v for k, v in self.chunk_metadata[doc_idx].items()
                    if not k.startswith("_")
                }
                chunk_info["bm25_score"] = float(score)
                results.append(chunk_info)
        return results

    def save_index(self, path: Path) -> None:
        """Serialize index to pickle file (format v2: vectorized BM25)."""
        idx = self.bm25_index
        data = {
            "format_version": BM25Index.FORMAT_VERSION,
            "corpus": self.corpus,
            "chunk_metadata": self.chunk_metadata,
            "bm25_params": {
                "k1": idx.k1 if idx else BM25_K1,
                "b": idx.b if idx else BM25_B,
            },
            "doc_lengths": idx.doc_lengths if idx is not None else None,
            "avg_doc_length": idx.avg_doc_length if idx is not None else 0,
            "vectorizer": idx.vectorizer if idx is not None else None,
            "tf_matrix": idx.tf_matrix if idx is not None else None,
            "idf": idx.idf if idx is not None else None,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Index saved to %s (format v%d)", path, BM25Index.FORMAT_VERSION)

    def load_index(self, path: Path) -> None:
        """Load index from pickle file. Supports format v1 (legacy) and v2."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.corpus = data["corpus"]
        self.chunk_metadata = data["chunk_metadata"]

        fmt = int(data.get("format_version", 1))
        self.bm25_index = BM25Index(
            k1=data["bm25_params"]["k1"],
            b=data["bm25_params"]["b"],
        )

        if fmt >= 2 and "tf_matrix" in data and data.get("tf_matrix") is not None:
            # New format
            self.bm25_index.doc_lengths = np.asarray(data["doc_lengths"], dtype=np.float32)
            self.bm25_index.avg_doc_length = float(data["avg_doc_length"])
            self.bm25_index.vectorizer = data["vectorizer"]
            self.bm25_index.tf_matrix = data["tf_matrix"]
            self.bm25_index.idf = np.asarray(data["idf"], dtype=np.float32)
        else:
            # Legacy v1: had a TfidfVectorizer + tfidf_matrix and Python-loop scoring.
            # Rather than carry two scoring paths, refit a CountVectorizer from
            # the corpus on load. One-time cost; future saves are v2.
            logger.warning(
                "Index pickle is legacy format v1; refitting BM25 in v2 format on load."
            )
            self.bm25_index.fit(self.corpus)

        self._build_name_index()
        logger.info(
            "Index loaded from %s: %d chunks, %d distinct names indexed for boost",
            path, len(self.corpus), len(self._name_to_chunks),
        )


def inspect_delta_path(spark, delta_path: str) -> None:
    """
    Print row count, schema, sample rows, and recent Delta history (debugging MinIO vs Spark).
    """
    logger.info("Inspecting Delta path: %s", delta_path)
    try:
        df = spark.read.format("delta").load(delta_path)
    except Exception as e:
        logger.error("Failed to load Delta: %s", e)
        raise

    df.printSchema()
    n = df.count()
    logger.info("Row count: %d", n)
    df.show(5, truncate=100)

    try:
        from delta.tables import DeltaTable

        dt = DeltaTable.forPath(spark, delta_path)
        logger.info("Last Delta operations (check for empty WRITE / overwrite):")
        dt.history(5).show(20, truncate=False)
    except Exception as e:
        logger.warning("Could not read Delta history (delta-spark / path issue): %s", e)


def load_drug_records_from_delta(spark, delta_path: str) -> List[Dict[str, Any]]:
    """
    Load drug records from Delta table.

    Args:
        spark: SparkSession
        delta_path: Path to Delta table (e.g., "s3a://bronze/silver/drugs_clean/")

    Returns:
        List of drug record dicts
    """
    try:
        df = spark.read.format("delta").load(delta_path)
    except Exception as e:
        raise ValueError(
            f"Cannot read Delta at {delta_path!r}: {e}\n"
            "Check MinIO is running, bucket/path exists, and the table was written with "
            "`promote_and_report.py --publish-s3`."
        ) from e

    n = df.count()
    logger.info("Delta columns (%d): %s", len(df.columns), df.columns)
    logger.info("Delta row count: %d", n)
    if n == 0:
        raise ValueError(
            f"Delta at {delta_path!r} has 0 rows. The path may exist but is empty.\n"
            "Fix: re-run promotion with good data:\n"
            "  spark-submit ... project_demo/dq/promote_and_report.py "
            "--good-path artifacts/dq/good_silver.parquet ... --publish-s3\n"
            "Or build the RAG index from local promoted silver:\n"
            "  python -m project_demo.rag.build_index --local-path silver/silver_drugs.parquet"
        )

    # Convert to list of dicts
    records = []
    for row in df.collect():
        record = {}
        for field in df.columns:
            record[field] = row[field]
        records.append(record)

    return records


def build_index_from_delta(
    spark,
    delta_path: str,
    output_path: Path,
    sample_limit: Optional[int] = None
) -> EvidenceRetriever:
    """
    Build evidence index from Delta table.

    Args:
        spark: SparkSession
        delta_path: Delta table path
        output_path: Where to save index
        sample_limit: Optional limit on records

    Returns:
        EvidenceRetriever with built index
    """
    logger.info(f"Loading drug records from Delta: {delta_path}")
    records = load_drug_records_from_delta(spark, delta_path)

    if sample_limit:
        records = records[:sample_limit]

    retriever = EvidenceRetriever()
    retriever.build_index(records)
    retriever.save_index(output_path)

    return retriever


# Convenience function for simple usage
def retrieve_evidence(query: str, top_k: int = 5, index_path: Path = EVIDENCE_INDEX_PATH) -> List[Dict[str, Any]]:
    """
    Load index and retrieve evidence in one call.

    Args:
        query: Search query
        top_k: Number of results
        index_path: Path to saved index

    Returns:
        List of evidence snippets
    """
    retriever = EvidenceRetriever()
    retriever.load_index(index_path)
    return retriever.retrieve_evidence(query, top_k=top_k)
