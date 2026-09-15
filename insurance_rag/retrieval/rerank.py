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


def _device_dtype():
    """fp16 is a GPU format; asking a CPU for it is slower than fp32, not faster."""
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, (torch.float16 if device == "cuda" else torch.float32)


@lru_cache(maxsize=1)
def _model():
    """Tokenizer and model for whichever arm is configured. Loaded once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

    spec = settings.reranker_spec
    device, dtype = _device_dtype()

    if spec.backend == "causal":
        # Left padding: the score is the logit at the final position, which must be a real token.
        tokenizer = AutoTokenizer.from_pretrained(spec.model, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(spec.model, dtype=dtype)
        ids = (tokenizer.convert_tokens_to_ids("no"), tokenizer.convert_tokens_to_ids("yes"))
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec.model)
        model = AutoModelForSequenceClassification.from_pretrained(spec.model, dtype=dtype)
        ids = None

    return tokenizer, model.to(device).eval(), ids


def _encode_causal(tokenizer, query: str, texts: list[str]):
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
    """P(relevant) per pair, in order. A calibrated probability - which cosine distance was not."""
    import torch

    tokenizer, model, ids = _model()
    causal = ids is not None
    scores: list[float] = []

    for start in range(0, len(texts), BATCH_SIZE):
        window = texts[start : start + BATCH_SIZE]
        if causal:
            batch = _encode_causal(tokenizer, query, window)
        else:
            batch = tokenizer(
                [query] * len(window), window, padding=True, truncation="only_second",
                max_length=MAX_LENGTH, return_tensors="pt",
            )
        batch = {key: value.to(model.device) for key, value in batch.items()}

        with torch.no_grad():
            logits = model(**batch).logits

        if causal:
            # The yes/no logits at the final position, read as a two-way choice.
            no_id, yes_id = ids
            pair = torch.stack([logits[:, -1, no_id], logits[:, -1, yes_id]], dim=1)
            scores.extend(pair.float().log_softmax(dim=1)[:, 1].exp().tolist())
        else:
            # One regression head, so the probability is a plain sigmoid over it.
            scores.extend(logits[:, 0].float().sigmoid().tolist())
    return scores


def rerank(query: str, chunks: list[Chunk], *, k: int) -> list[tuple[Chunk, float]]:
    """Re-score the fused pool and keep the best `k`. No threshold - see docs/plan.md, Phase 5."""
    if not chunks:
        return []
    scored = list(zip(chunks, score_pairs(query, [c.text for c in chunks]), strict=True))
    scored.sort(key=lambda pair: (-pair[1], pair[0].chunk_id))
    return scored[:k]
