"""Deterministic guardrails: clean and screen the question, screen the answer. No model calls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["check_input", "check_output", "redact", "Intake", "Release"]

RULES_PATH = Path(__file__).with_name("policy_rules.json")

#: Personal-data formats, redacted before any model call, web query or log line.
REDACT = {
    "EMAIL": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    "LICENCE": re.compile(r"\b[A-Z]\d{4}[- ]?\d{5}[- ]?\d{5}\b"),
    "SIN": re.compile(r"\b\d{3}[- ]\d{3}[- ]\d{3}\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)"),
    "POSTAL": re.compile(r"\b[ABCEGHJ-NPRSTVXY]\d[A-Z][ -]?\d[A-Z]\d\b", re.I),
    "VIN": re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)[A-HJ-NPR-Z0-9]{17}\b", re.I),
    "ADDRESS": re.compile(
        r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|"
        r"Boulevard|Blvd|Lane|Ln|Court|Ct|Crescent|Cres|Way)\b\.?"
    ),
    "ID": re.compile(
        r"\b(?:policy|claim|licen[cs]e|vin)\s*(?:number|no\.?|#|id)?\s*[:#]?\s*(?:is\s+)?"
        r"(?=[A-Z0-9-]*\d)[A-Z0-9-]{5,}\b",
        re.I,
    ),
    "NAME": re.compile(r"(?i:\bmy name is\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?"),
}

#: Contact details that must come from an excerpt, never from the model.
CONTACT = (REDACT["PHONE"], REDACT["EMAIL"], re.compile(r"https?://\S+|www\.\S+", re.I))


@lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def redact(text: str) -> tuple[str, list[str]]:
    """The text with personal data replaced by [LABEL], and the labels that fired."""
    labels = []
    for label, pattern in REDACT.items():
        text, count = pattern.subn(f"[{label}]", text)
        if count:
            labels.append(label)
    return text, labels


@dataclass(frozen=True)
class Intake:
    """The cleaned question, what fired, and the fixed answer when the question is stopped."""

    question: str
    notes: list[str]
    blocked: str = ""
    answer: str = ""


def _matches(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.I) for pattern in patterns)


def check_input(question: str) -> Intake:
    """Redact, then stop injection, legal-strategy, recommendation and out-of-scope questions."""
    clean, labels = redact(question)
    notes = [f"redacted:{label}" for label in labels]
    for rule in rules()["input"]:
        if not _matches(rule["patterns"], clean):
            continue
        if rule.get("unless") and re.search(rule["unless"], clean, re.I):
            continue
        notes.append(rule["id"])
        if rule["action"] == "block":
            return Intake(clean, notes, rule["id"], rules()["templates"][rule["id"]])
    return Intake(clean, notes)


@dataclass(frozen=True)
class Release:
    """What may be shown: the answer and summary after screening, the footer, what fired."""

    answer: str
    summary: str
    blocked: str
    notes: list[str]
    disclaimer: str


def _flagged(text: str, context: str) -> str:
    """The first rule whose wording appears in `text` but not word for word in the excerpts."""
    checks = [(rule["id"], [re.compile(p, re.I) for p in rule["patterns"]])
              for rule in rules()["output"]]
    checks.append(("contact", list(CONTACT)))
    for rule_id, patterns in checks:
        for pattern in patterns:
            if any(m.group(0).lower() not in context for m in pattern.finditer(text)):
                return rule_id
    return ""


def _trim(text: str) -> str:
    """Cut an answer that stops mid-sentence back to its last full sentence."""
    if not text.strip() or text.rstrip()[-1] in '.!?]")*':
        return text
    masked = re.sub(r"\[[^\]]*\]?", lambda m: "_" * len(m.group()), text)  # keep citations whole
    ends = [m.end() for m in re.finditer(r"[.!?](?=\s|$)", masked)]
    return text[: ends[-1]] if ends else text


def check_output(
    answer: str, summary: str, excerpts: str, blocked: str, notes: list[str]
) -> Release:
    """Screen the answer and summary; a flagged answer is replaced, never edited."""
    notes, context = list(notes), excerpts.lower()
    if not blocked:
        rule_id = _flagged(answer, context)
        if rule_id:
            answer, blocked = rules()["templates"]["unsafe"], rule_id
            notes.append(f"output:{rule_id}")
        elif (trimmed := _trim(answer)) != answer:
            answer = trimmed
            notes.append("trimmed")
    if summary and _flagged(summary, ""):
        summary = ""
        notes.append("web_summary_dropped")
    extra = [text for key, text in rules()["notes"].items() if key in notes]
    return Release(answer, summary, blocked, notes, " ".join([rules()["disclaimer"], *extra]))
