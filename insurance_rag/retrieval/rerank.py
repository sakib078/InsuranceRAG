"""Cross-encoder rerank: query and clause read together, which the bi-encoder never does.

The bi-encoder scored every provision offline, before any question existed. This reads the pair
in one forward pass with full attention between them, so it can tell a clause that answers the
question from one that merely shares its vocabulary.

Qwen3-Reranker is a causal LM, not a classification head: it is asked a yes/no question and
scored on the logits of those two tokens. `sentence_transformers.CrossEncoder` cannot load it.
"""

from __future__ import annotations

from functools import lru_cache

from insurance_rag.config import settings
from insurance_rag.schema import Chunk

__all__ = ["rerank", "score_pairs", "INSTRUCT"]

_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

#: Naming exclusions is deliberate - a limiting provision is a relevant answer, not a near miss.
INSTRUCT = (
    "Given a question about Ontario auto insurance, decide whether this provision answers it, "
    "or states a condition or exclusion that limits the answer."
)

#: An 800-token provision plus its context header and the prompt scaffolding, with headroom.
MAX_LENGTH = 1024

#: Peak memory is batch x MAX_LENGTH of activations; 4 keeps a 0.6B model inside laptop RAM.
BATCH_SIZE = 4


@lru_cache(maxsize=1)
def _model():
    """Tokenizer, model, and the two token ids the score is read from. Loaded once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = settings.cross_encoder_model
    # Left padding: the score is the logit at the final position, which must be a real token.
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side="left")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(device).eval()
    return tokenizer, model, tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes")


def _encode(tokenizer, query: str, texts: list[str]):
    """Truncate the document, never the scaffolding - the yes/no cue has to survive."""
    prefix = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    batch = tokenizer(
        [f"<Instruct>: {INSTRUCT}\n<Query>: {query}\n<Document>: {text}" for text in texts],
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_LENGTH - len(prefix) - len(suffix),
    )
    batch["input_ids"] = [prefix + ids + suffix for ids in batch["input_ids"]]
    batch["attention_mask"] = [[1] * len(ids) for ids in batch["input_ids"]]
    return tokenizer.pad(batch, padding=True, return_tensors="pt")


def score_pairs(query: str, texts: list[str]) -> list[float]:
    """P(yes) per pair, in order. A calibrated probability - which cosine distance never was."""
    import torch

    tokenizer, model, no_id, yes_id = _model()
    scores: list[float] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = _encode(tokenizer, query, texts[start : start + BATCH_SIZE])
        batch = {key: value.to(model.device) for key, value in batch.items()}
        with torch.no_grad():
            logits = model(**batch).logits[:, -1, :]
        pair = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
        scores.extend(pair.float().log_softmax(dim=1)[:, 1].exp().tolist())
    return scores


def rerank(query: str, chunks: list[Chunk], *, k: int) -> list[tuple[Chunk, float]]:
    """Re-score the fused pool and keep the best `k`. No threshold - see docs/plan.md, Phase 5."""
    if not chunks:
        return []
    scored = list(zip(chunks, score_pairs(query, [c.text for c in chunks]), strict=True))
    scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
    return scored[:k]
