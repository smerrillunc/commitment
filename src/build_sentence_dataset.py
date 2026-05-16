#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

THIS_FILE = Path(__file__).resolve()
COMMITMENT_ROOT = THIS_FILE.parents[1]
UTILS_DIR = COMMITMENT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from sentence_pipeline import build_sentence_records, write_jsonl


LABEL_FILTER_ALL = "all"
LABEL_FILTER_CORRECT_ONLY = "correct_only"
LABEL_FILTER_INCORRECT_ONLY = "incorrect_only"
LABEL_FILTER_CHOICES = [
    LABEL_FILTER_ALL,
    LABEL_FILTER_CORRECT_ONLY,
    LABEL_FILTER_INCORRECT_ONLY,
]


def normalize_label_filter(
    label_filter: str,
    *,
    only_correct: bool = False,
    only_incorrect: bool = False,
) -> str:
    if only_correct and only_incorrect:
        raise ValueError("Cannot set both --only_correct and --only_incorrect.")
    if only_correct:
        return LABEL_FILTER_CORRECT_ONLY
    if only_incorrect:
        return LABEL_FILTER_INCORRECT_ONLY
    return label_filter


def keep_record_for_label_filter(record: Dict[str, Any], label_filter: str) -> bool:
    is_correct = record.get("is_correct")
    if label_filter == LABEL_FILTER_ALL:
        return True
    if label_filter == LABEL_FILTER_CORRECT_ONLY:
        return is_correct is True
    if label_filter == LABEL_FILTER_INCORRECT_ONLY:
        return is_correct is False
    raise ValueError(f"Unknown label_filter: {label_filter}")


def iter_commitment_records(input_root: str | Path) -> Iterator[Dict[str, Any]]:
    root = Path(input_root)
    if root.is_file():
        candidate_files = [root]
    else:
        primary = root / "commitment_samples.jsonl"
        if primary.exists():
            candidate_files = [primary]
        else:
            candidate_files = sorted(root.glob("*.jsonl"))
    for path in candidate_files:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def _pick_text(rec: Dict[str, Any], primary_field: str, fallback_field: Optional[str]) -> Optional[str]:
    text = rec.get(primary_field)
    if isinstance(text, str) and text.strip():
        return text
    if fallback_field:
        alt = rec.get(fallback_field)
        if isinstance(alt, str) and alt.strip():
            return alt
    return None


def _write_examples(
    records: Iterable[Dict[str, Any]],
    out_path: Path,
    *,
    text_field: str,
    fallback_text_field: Optional[str],
    label_filter: str,
    limit: int,
    target_correct: int,
    target_incorrect: int,
    stats: Dict[str, int],
) -> list[Dict[str, Any]]:
    by_example_id: dict[str, Dict[str, Any]] = {}
    duplicate_counts = defaultdict(int)

    for rec in records:
        stats["seen_records"] += 1
        example_id = rec.get("example_id")
        if not example_id:
            stats["skipped_missing_example_id"] += 1
            continue
        if not keep_record_for_label_filter(rec, label_filter):
            continue
        text = _pick_text(rec, text_field, fallback_text_field)
        if not text:
            stats["skipped_missing_text"] += 1
            continue

        prepared = dict(rec)
        prepared[text_field] = text
        duplicate_counts[example_id] += 1
        by_example_id.setdefault(example_id, prepared)

    stats["duplicate_example_id_groups"] = sum(1 for count in duplicate_counts.values() if count > 1)
    stats["duplicate_example_ids_dropped"] = sum(max(0, count - 1) for count in duplicate_counts.values())

    selected: list[Dict[str, Any]] = []
    correct_written = 0
    incorrect_written = 0

    for example_id in sorted(by_example_id):
        prepared = by_example_id[example_id]
        if prepared.get("is_correct") is True and target_correct > 0 and correct_written >= target_correct:
            continue
        if prepared.get("is_correct") is False and target_incorrect > 0 and incorrect_written >= target_incorrect:
            continue

        selected.append(prepared)
        if prepared.get("is_correct") is True:
            correct_written += 1
            stats["written_correct"] += 1
        elif prepared.get("is_correct") is False:
            incorrect_written += 1
            stats["written_incorrect"] += 1
        stats["written_examples"] += 1

        if limit and len(selected) >= limit:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for prepared in selected:
            handle.write(json.dumps(prepared, ensure_ascii=False) + "\n")

    return selected


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a sentence-level accuracy-localization dataset from commitment miner JSONL."
    )
    parser.add_argument("--input_root", type=str, required=True, help="Path to commitment_samples.jsonl or its directory.")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory.")
    parser.add_argument("--text_field", type=str, default="action_reasoning")
    parser.add_argument("--fallback_text_field", type=str, default="reasoning")
    parser.add_argument("--label_filter", type=str, choices=LABEL_FILTER_CHOICES, default=LABEL_FILTER_ALL)
    parser.add_argument("--only_correct", action="store_true", default=False)
    parser.add_argument("--only_incorrect", action="store_true", default=False)
    parser.add_argument("--target_correct", type=int, default=0)
    parser.add_argument("--target_incorrect", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args(argv)

    label_filter = normalize_label_filter(
        args.label_filter,
        only_correct=args.only_correct,
        only_incorrect=args.only_incorrect,
    )

    out_dir = Path(args.out_dir)
    examples_path = out_dir / "examples.jsonl"
    sentences_path = out_dir / "sentences.jsonl"

    stats = {
        "seen_records": 0,
        "skipped_missing_example_id": 0,
        "skipped_missing_text": 0,
        "duplicate_example_id_groups": 0,
        "duplicate_example_ids_dropped": 0,
        "written_examples": 0,
        "written_correct": 0,
        "written_incorrect": 0,
    }

    examples = _write_examples(
        iter_commitment_records(args.input_root),
        examples_path,
        text_field=args.text_field,
        fallback_text_field=args.fallback_text_field,
        label_filter=label_filter,
        limit=args.limit,
        target_correct=args.target_correct,
        target_incorrect=args.target_incorrect,
        stats=stats,
    )

    sentences = build_sentence_records(
        examples,
        text_field=args.text_field,
        example_id_field="example_id",
        include_example_fields=[
            "base_example_id",
            "answer_status",
            "is_correct",
            "dataset_name",
            "split",
            "dataset_index",
            "model_name",
            "problem",
            "options",
            "question",
            "prompt",
            "messages",
            "correct_option",
            "correct_option_text",
            "predicted_option",
            "problem_category",
        ],
    )
    write_jsonl(sentences, sentences_path)

    print(f"Wrote examples: {examples_path}")
    print(f"Wrote sentences: {sentences_path}")
    print(f"Label filter: {label_filter}")
    print(f"Examples written: {stats['written_examples']}")
    print(f"Correct examples written: {stats['written_correct']}")
    print(f"Incorrect examples written: {stats['written_incorrect']}")
    print(f"Skipped missing example_id: {stats['skipped_missing_example_id']}")
    print(f"Skipped missing text: {stats['skipped_missing_text']}")
    print(f"Duplicate example_id groups collapsed: {stats['duplicate_example_id_groups']}")
    print(f"Duplicate rows dropped by example_id: {stats['duplicate_example_ids_dropped']}")


if __name__ == "__main__":
    main()
