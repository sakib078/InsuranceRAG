"""Central configuration; the encoder is an eval variable (docs/plan.md, Deviation 8)."""

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
    #: clause | uniform; switches the chunk directory and pgvector collection together.
    chunking: Chunking = Chunking.CLAUSE

    # --- storage (no DSN default; credentials live in .env only) ---
    postgres_dsn: str = Field(..., description="psycopg3 DSN, e.g. postgresql+psycopg://...")

    # --- retrieval knobs (every one of these is an eval variable) ---
    dense_top_k: int = 50
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    #: The ladder's wide rung, separate from `fusion_top_k`.
    ladder_top_k: int = 25
    rrf_k: int = 60  # reciprocal rank fusion smoothing constant
    #: Per-channel RRF weights; below ~0.5 sparse can only promote what dense found.
    dense_weight: float = 1.0
    sparse_weight: float = 0.7

    # --- chunking, measured against the reference tokenizer ---
    max_chunk_tokens: int = 800
    uniform_chunk_tokens: int = 512  # baseline row only
    uniform_overlap_tokens: int = 64

    # --- generation (groq | gemini, see insurance_rag/providers.py) ---
    generation_provider: str = "groq"
    # Optional so ingestion and retrieval run without it; `chain.py` fails loudly when it is needed.
    groq_api_key: str | None = Field(default=None, description="IRAG_GROQ_API_KEY, from .env")
    generation_model: str = "openai/gpt-oss-120b"

    # --- evaluation: judge runs off a different provider; keys are unprefixed in .env ---
    gemini_eval_key: str | None = Field(default=None, validation_alias="GEMINI_API_EVAL_KEY")
    openrouter_key: str | None = Field(default=None, validation_alias="OPENROUTER_KEY")
    mistral_api_key: str | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    nvidia_api_key: str | None = Field(default=None, validation_alias="NVIDIA_API_KEY")
    # pydantic-settings never fills os.environ, so the LangSmith client is handed this key.
    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_dataset: str = "insurance-rag-golden"
    # Ordered "provider/model" judge candidates; see insurance_rag/providers.py.
    judge_chain: str = (
        "gemini/models/gemini-3.5-flash-lite,"
        "gemini/models/gemini-3.1-flash-lite,"
        "groq/qwen/qwen3.8-27b,"
        "nvidia/z-ai/glm-5.3-flash,"
        "openrouter/google/gemma-4-31b-it:free,"
        "nvidia/deepseek-ai/deepseek-v4-flash-0731"
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
