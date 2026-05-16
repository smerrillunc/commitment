from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


SENTENCE_BOUNDARY_CHARS = ".!?"
SENTENCE_TRAILING_CLOSERS = "\"')}]"
COMMON_ABBREVIATIONS = {
    "approx.",
    "cf.",
    "dr.",
    "e.g.",
    "etc.",
    "fig.",
    "i.e.",
    "jr.",
    "mr.",
    "mrs.",
    "ms.",
    "no.",
    "p.m.",
    "a.m.",
    "prof.",
    "sr.",
    "st.",
    "u.k.",
    "u.s.",
    "vs.",
}


def _is_decimal_point(text: str, idx: int) -> bool:
    return (
        text[idx] == "."
        and idx > 0
        and idx + 1 < len(text)
        and text[idx - 1].isdigit()
        and text[idx + 1].isdigit()
    )


def _next_nonspace_index(text: str, start: int) -> int | None:
    idx = start
    while idx < len(text):
        if not text[idx].isspace():
            return idx
        idx += 1
    return None


def _token_before_period(text: str, idx: int) -> str:
    start = idx
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.'")
    while start > 0 and text[start - 1] in allowed:
        start -= 1
    token = text[start : idx + 1].lstrip("\"'([")
    return token.lower()


def _looks_like_initialism(token: str) -> bool:
    return bool(re.fullmatch(r"(?:[a-z]\.){2,}", token))


def _is_abbreviation(text: str, idx: int) -> bool:
    if text[idx] != ".":
        return False

    token = _token_before_period(text, idx)
    if token in COMMON_ABBREVIATIONS or _looks_like_initialism(token):
        return True

    next_idx = _next_nonspace_index(text, idx + 1)
    if next_idx is None:
        return False

    bare = token[:-1]
    if bare.isalpha() and len(bare) <= 3 and text[next_idx].islower():
        return True

    return False


def _is_sentence_boundary(text: str, idx: int) -> bool:
    char = text[idx]
    if char not in SENTENCE_BOUNDARY_CHARS:
        return False
    if char == "." and (_is_decimal_point(text, idx) or _is_abbreviation(text, idx)):
        return False
    return True


def _iter_sentence_bounds(text: str) -> Iterator[tuple[int, int]]:
    n_chars = len(text)
    start = 0
    while start < n_chars and text[start].isspace():
        start += 1
    if start >= n_chars:
        return

    idx = start
    while idx < n_chars:
        if _is_sentence_boundary(text, idx):
            end = idx + 1
            while end < n_chars and text[end] in SENTENCE_TRAILING_CLOSERS:
                end += 1
            yield start, end
            start = end
            while start < n_chars and text[start].isspace():
                start += 1
            idx = start
            continue
        idx += 1

    if start < n_chars:
        end = n_chars
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            yield start, end


def split_sentences(text: Any) -> List[str]:
    return [span["text"] for span in split_sentence_spans(text)]


def split_sentence_spans(text: Any) -> List[Dict[str, Any]]:
    if not isinstance(text, str) or not text.strip():
        return []

    spans: list[dict[str, Any]] = []
    for start, end in _iter_sentence_bounds(text):
        trimmed_start = start
        trimmed_end = end
        while trimmed_start < trimmed_end and text[trimmed_start].isspace():
            trimmed_start += 1
        while trimmed_end > trimmed_start and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        if trimmed_end <= trimmed_start:
            continue
        spans.append(
            {
                "start": trimmed_start,
                "end": trimmed_end,
                "text": text[trimmed_start:trimmed_end],
            }
        )
    return spans


def read_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, Any]], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_sentence_id(example_id: str, sentence_idx: int) -> str:
    return f"{example_id}/sent_{sentence_idx:04d}"


def build_sentence_records(
    examples: Iterable[Dict[str, Any]],
    *,
    text_field: str = "action_reasoning",
    example_id_field: str = "example_id",
    include_example_fields: Optional[List[str]] = None,
) -> Iterator[Dict[str, Any]]:
    include_example_fields = include_example_fields or []

    for example in examples:
        example_id = example.get(example_id_field) or example.get("example_id")
        if not example_id:
            continue

        text = example.get(text_field)
        for idx, sent in enumerate(split_sentence_spans(text)):
            record = {
                "sentence_id": make_sentence_id(example_id, idx),
                "example_id": example_id,
                "source_field": text_field,
                "sentence_idx": idx,
                "sentence_text": sent["text"],
                "start": sent["start"],
                "end": sent["end"],
            }
            for field in include_example_fields:
                if field in example:
                    record[field] = example[field]
            yield record

