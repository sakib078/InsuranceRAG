"""Central configuration.

Two model profiles exist by design:

  EVAL  - larger models, used to produce the numbers in the README results table.
  SERVE - smaller models, used in the deployed container so the image fits the
          Azure Container Apps free grant (scale-to-zero, ~1GB resident).

Both profiles run the identical retrieval pipeline. The README reports numbers
for both, so the deployed demo is never misrepresented as the evaluated system.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INDEX_DIR = DATA_DIR / "index"
EVALS_DIR = REPO_ROOT / "evals"
RESULTS_DIR = EVALS_DIR / "results"


class Profile(StrEnum):
    EVAL = "eval"
    SERVE = "serve"


MODEL_PROFILES: dict[Profile, dict[str, str]] = {
    Profile.EVAL: {
        "embedding_model": "BAAI/bge-base-en-v1.5",
        "reranker_model": "BAAI/bge-reranker-base",
    },
    Profile.SERVE: {
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="IRAG_", extra="ignore"
    )

    profile: Profile = Profile.EVAL

    # --- retrieval knobs (every one of these is an eval variable) ---
    dense_top_k: int = 20
    sparse_top_k: int = 20
    fusion_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60  # reciprocal rank fusion smoothing constant

    # --- chunking ---
    max_chunk_tokens: int = 512
    chunk_overlap_tokens: int = 64

    # --- generation ---
    anthropic_api_key: str | None = Field(default=None)
    generation_model: str = "claude-opus-5"

    # --- tracing ---
    trace_log_path: Path = REPO_ROOT / "data" / "traces.jsonl"

    @property
    def embedding_model(self) -> str:
        return MODEL_PROFILES[self.profile]["embedding_model"]

    @property
    def reranker_model(self) -> str:
        return MODEL_PROFILES[self.profile]["reranker_model"]


settings = Settings()
