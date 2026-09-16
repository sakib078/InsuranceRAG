"""Central configuration.

The encoder is an eval variable, not a deployment profile: both models under test run the
identical pipeline over byte-identical chunks. See `docs/plan.md`, Deviation 8.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
PDF_DIR = DATA_DIR / "pdfs"
EVALS_DIR = REPO_ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results"


class Chunking(StrEnum):
    """How the corpus was cut. An eval variable: the uniform arm exists to be compared."""

    CLAUSE = "clause"
    UNIFORM = "uniform"


class Encoder(StrEnum):
    QWEN3 = "qwen3"
    BGE_M3 = "bge-m3"


@dataclass(frozen=True)
class EncoderSpec:
    """A bi-encoder and its own family's reranker - never a mixed pipeline."""

    bi_encoder: str
    cross_encoder: str
    dim: int


ENCODERS: dict[Encoder, EncoderSpec] = {
    Encoder.QWEN3: EncoderSpec("Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Reranker-0.6B", 1024),
    Encoder.BGE_M3: EncoderSpec("BAAI/bge-m3", "BAAI/bge-reranker-v2-m3", 1024),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="IRAG_", extra="ignore")

    encoder: Encoder = Encoder.QWEN3
    #: clause | uniform. Switches the chunk directory and the pgvector collection together,
    #: so the two corpora can never be read through each other.
    chunking: Chunking = Chunking.CLAUSE

    # --- storage ---
    # No default: credentials live in .env only, so none can be committed by accident.
    postgres_dsn: str = Field(..., description="psycopg3 DSN, e.g. postgresql+psycopg://...")

    # --- retrieval knobs (every one of these is an eval variable) ---
    dense_top_k: int = 50
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # reciprocal rank fusion smoothing constant
    #: Per-channel RRF weights. Equal weighting let sparse *inject* candidates - a junk hit at
    #: sparse rank 1 ties dense rank 1 at any `rrf_k`, and with 5 slots that is round-robin.
    #: Below ~0.5 sparse can only *promote* what dense already found, which is all an unreliable
    #: channel should be trusted to do. Measured: equal weights cost 0.737 -> 0.632 single and
    #: 0.450 -> 0.200 exclusion.
    dense_weight: float = 1.0
    sparse_weight: float = 0.7

    # --- chunking, measured against the reference tokenizer ---
    max_chunk_tokens: int = 800
    uniform_chunk_tokens: int = 512  # baseline row only
    uniform_overlap_tokens: int = 64

    # --- generation (open-weight model on an OpenAI-compatible endpoint) ---
    # groq | gemini - see insurance_rag/providers.py.
    generation_provider: str = "groq"
    # Optional so ingestion and retrieval run without it; `chain.py` fails loudly when it is needed.
    groq_api_key: str | None = Field(default=None, description="IRAG_GROQ_API_KEY, from .env")
    generation_model: str = "openai/gpt-oss-120b"

    # --- evaluation: the judge runs off a different provider, for a separate rate limit
    # and to keep a model family from grading its own output. These keys are unprefixed in .env.
    gemini_eval_key: str | None = Field(default=None, validation_alias="GEMINI_API_EVAL_KEY")
    openrouter_key: str | None = Field(default=None, validation_alias="OPENROUTER_KEY")
    # pydantic-settings reads .env into this object, never into os.environ, so the LangSmith
    # client has to be handed the key rather than left to find it.
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_dataset: str = "insurance-rag-golden"
    # See insurance_rag/providers.py for the options. A different provider from the generator
    # means a separate budget; a different family means it cannot favour its own phrasing.
    # Ordered "provider/model" candidates, split on the FIRST slash so Groq's
    # "qwen/qwen3.8-27b" survives. Failover moves down the list as each budget runs out.
    #   3.5-flash-lite  15 RPM / 500 RPD  - best judge with capacity for a whole run
    #   3.1-flash-lite  15 RPM / 500 RPD  - same limits, separate budget
    #   groq qwen       200k TPD of its own: Groq quotas are per-model, so judging here never
    #                   touches the budget generation spends on gpt-oss-120b
    #   openrouter      last, because its free pool returns 429 "Provider returned error"
    #                   under load - fine as a final fallback, wrong as a dependency
    # Gemini needs the "models/" prefix on its OpenAI-compatible endpoint. gemma-4-31b-it is
    # NOT in the chain on Gemini - it returns 500 above max_tokens=16 there, and a 5xx is not
    # exhaustion, so it would raise and end the run rather than fall through.
    judge_chain: str = (
        "gemini/models/gemini-3.5-flash-lite,"
        "gemini/models/gemini-3.1-flash-lite,"
        "groq/qwen/qwen3.8-27b,"
        "openrouter/google/gemma-4-31b-it:free"
    )

    # --- agent ---
    max_agent_steps: int = 6
    enable_web_fallback: bool = False  # must stay false for every eval run

    # --- tracing ---
    trace_log_path: Path = DATA_DIR / "traces.jsonl"

    @property
    def chunks_dir(self) -> Path:
        return DATA_DIR / ("chunks" if self.chunking is Chunking.CLAUSE else "chunks_uniform")

    @property
    def collection_name(self) -> str:
        """The clause-aware name is unchanged, so the frozen index is never touched."""
        suffix = "" if self.chunking is Chunking.CLAUSE else "_uniform"
        return f"chunks_{self.encoder}{suffix}"

    @property
    def spec(self) -> EncoderSpec:
        return ENCODERS[self.encoder]

    @property
    def bi_encoder_model(self) -> str:
        return self.spec.bi_encoder


settings = Settings()
