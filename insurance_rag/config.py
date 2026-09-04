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

    # --- storage ---
    postgres_dsn: str = "postgresql+psycopg://insurance_01:project-pass@localhost:6024/langchain"

    # --- retrieval knobs (every one of these is an eval variable) ---
    dense_top_k: int = 50
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # reciprocal rank fusion smoothing constant

    # --- chunking, measured against the reference tokenizer ---
    min_chunk_tokens: int = 400
    max_chunk_tokens: int = 800
    uniform_chunk_tokens: int = 512  # baseline row only
    uniform_overlap_tokens: int = 64

    # --- generation ---
    anthropic_api_key: str | None = Field(default=None)
    generation_model: str = "claude-opus-5"

    # --- agent ---
    max_agent_steps: int = 6
    enable_web_fallback: bool = False  # must stay false for every eval run

    # --- tracing ---
    trace_log_path: Path = DATA_DIR / "traces.jsonl"

    @property
    def spec(self) -> EncoderSpec:
        return ENCODERS[self.encoder]

    @property
    def bi_encoder_model(self) -> str:
        return self.spec.bi_encoder

    @property
    def cross_encoder_model(self) -> str:
        return self.spec.cross_encoder

    @property
    def embedding_dim(self) -> int:
        return self.spec.dim


settings = Settings()
