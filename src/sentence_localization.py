#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import torch

THIS_FILE = Path(__file__).resolve()
COMMITMENT_ROOT = THIS_FILE.parents[1]
UTILS_DIR = COMMITMENT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from sentence_pipeline import read_jsonl, split_sentence_spans

from mathqa_utils import normalize_final_answer, parse_mathqa_options, parse_model_output


LABEL_FILTER_ALL = "all"
LABEL_FILTER_CORRECT_ONLY = "correct_only"
LABEL_FILTER_INCORRECT_ONLY = "incorrect_only"
LABEL_FILTER_CHOICES = [
    LABEL_FILTER_ALL,
    LABEL_FILTER_CORRECT_ONLY,
    LABEL_FILTER_INCORRECT_ONLY,
]


def _import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("This script requires vLLM in the active environment.") from exc
    return LLM, SamplingParams


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


def wilson_interval(num_success: int, num_total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if num_total <= 0:
        return None, None
    phat = num_success / num_total
    denom = 1.0 + (z * z / num_total)
    center = (phat + (z * z) / (2 * num_total)) / denom
    margin = (
        z
        * math.sqrt((phat * (1.0 - phat) / num_total) + ((z * z) / (4 * num_total * num_total)))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _record_id(example: Dict[str, Any]) -> str:
    return str(example.get("example_id") or example.get("base_example_id") or "unknown")


def _example_output_path(out_dir: Path, example: Dict[str, Any]) -> Path:
    safe_id = _record_id(example).replace("/", "__")
    return out_dir / f"sentence_localization_{safe_id}.json"


def _filter_pending_examples(
    examples: List[Dict[str, Any]],
    *,
    out_dir: Path,
    overwrite: bool,
) -> tuple[List[Dict[str, Any]], int]:
    if overwrite:
        return examples, 0
    pending: list[Dict[str, Any]] = []
    existing_outputs = 0
    for example in examples:
        if _example_output_path(out_dir, example).exists():
            existing_outputs += 1
            continue
        pending.append(example)
    return pending, existing_outputs


def _shard_examples(example_list: List[Dict[str, Any]], *, shard_id: int, num_shards: int) -> List[Dict[str, Any]]:
    if num_shards <= 1:
        return example_list
    return [example for i, example in enumerate(example_list) if (i % num_shards) == shard_id]


def _extract_raw_text(example: Dict[str, Any], text_field: str) -> Optional[str]:
    candidates = [text_field, "action_reasoning", "reasoning", "action_raw_text"]
    for key in candidates:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _load_sentences(sentences_path: Optional[str]) -> Dict[str, List[Dict[str, Any]]]:
    if not sentences_path:
        return {}
    path = Path(sentences_path)
    if not path.exists():
        return {}

    by_example: Dict[str, List[Dict[str, Any]]] = {}
    for sentence in read_jsonl(path):
        example_id = sentence.get("example_id")
        if not example_id:
            continue
        by_example.setdefault(example_id, []).append(sentence)

    for items in by_example.values():
        items.sort(key=lambda item: item.get("sentence_idx", 0))

    return by_example


def _build_prefix_messages(
    prompt: str,
    prompt_messages: Optional[List[Dict[str, Any]]],
    prefix_text: str,
) -> List[Dict[str, Any]]:
    base_messages = list(prompt_messages) if isinstance(prompt_messages, list) and prompt_messages else []
    if not base_messages and prompt:
        base_messages = [{"role": "system", "content": prompt}]
    return base_messages + [{"role": "assistant", "content": prefix_text}]


def _render_prefix_prompt(
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[List[Dict[str, Any]]],
    prefix_text: str,
) -> str:
    if not isinstance(prompt_messages, list) or not prompt_messages:
        return prompt + prefix_text

    if not prefix_text:
        try:
            return tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            return prompt

    messages = _build_prefix_messages(prompt, prompt_messages, prefix_text)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
    except (TypeError, ValueError):
        try:
            base_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return base_prompt + prefix_text
        except Exception:
            return prompt + prefix_text


def sample_correctness_for_prefix(
    llm: LLM,
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[List[Dict[str, Any]]],
    prefix_text: str,
    *,
    correct_option: str,
    option_map: dict[str, str],
    n_samples: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    base_seed: int,
) -> tuple[float, int, int, list[dict[str, Any]]]:
    _, SamplingParams = _import_vllm()
    rendered_prompt = _render_prefix_prompt(
        tokenizer,
        prompt,
        prompt_messages,
        prefix_text,
    )
    sampling_params = SamplingParams(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_tokens=max_new_tokens,
        seed=base_seed,
    )
    outputs = llm.generate(prompts=[rendered_prompt], sampling_params=sampling_params)

    num_correct = 0
    num_valid = 0
    generations: list[dict[str, Any]] = []

    for output in outputs:
        for sample_output in output.outputs:
            gen_text = sample_output.text
            full_generation_text = prefix_text + gen_text
            parsed = parse_model_output(full_generation_text)
            predicted_option = normalize_final_answer(
                parsed["Final Answer"],
                option_map=option_map,
            )
            is_correct = (
                predicted_option is not None and predicted_option == correct_option
            )
            if predicted_option is not None:
                num_valid += 1
                if is_correct:
                    num_correct += 1
            generations.append(
                {
                    "prompt": prompt,
                    "prefix_text": prefix_text,
                    "gen_text": gen_text,
                    "full_generation_text": full_generation_text,
                    "final_answer": parsed["Final Answer"],
                    "predicted_option": predicted_option,
                    "parse_success": bool(parsed["parse_success"]),
                    "parse_error": parsed["Parse Error"],
                    "matched_final_answer_block": parsed["matched_final_answer_block"],
                    "is_correct": bool(is_correct),
                }
            )

    correct_rate = 0.5 if num_valid == 0 else (num_correct / num_valid)
    return correct_rate, num_correct, num_valid, generations


def localize_correctness_by_sentence(
    llm: LLM,
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[List[Dict[str, Any]]],
    raw_text: str,
    sentences: List[Dict[str, Any]],
    *,
    correct_option: str,
    option_map: dict[str, str],
    n_samples: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    base_seed: int,
    mode: str,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []

    for idx, sent in enumerate(sentences):
        if mode == "prefix":
            prefix_text = raw_text[: sent["end"]]
        elif mode == "sentence_only":
            prefix_text = sent["text"]
        else:
            raise ValueError(f"Unknown mode: {mode}")

        correct_rate, num_correct, num_valid, generations = sample_correctness_for_prefix(
            llm,
            tokenizer,
            prompt,
            prompt_messages,
            prefix_text,
            correct_option=correct_option,
            option_map=option_map,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_seed=base_seed + idx + 1,
        )
        ci_low, ci_high = wilson_interval(num_correct, num_valid)
        history.append(
            {
                "sentence_idx": idx,
                "char_span": (sent["start"], sent["end"]),
                "sentence_text": sent["text"],
                "target_sentence_text": sent["text"],
                "prompt": prompt,
                "prefix_text": prefix_text,
                "correct_rate": correct_rate,
                "num_correct": num_correct,
                "num_valid": num_valid,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "generations": generations,
            }
        )
    return history


def _pick_midpoint(left_idx: int, right_idx: int, sent_idxs: List[int], *, min_spacing: int, n_sent: int) -> Optional[int]:
    if right_idx - left_idx <= 1:
        return None
    candidate = (left_idx + right_idx) // 2
    if candidate < 1 or candidate > n_sent:
        return None
    if any(abs(candidate - idx) < min_spacing for idx in sent_idxs):
        return None
    return candidate


def next_high_gradient_sentence(history: List[Dict[str, Any]], *, min_spacing: int, n_sent: int) -> Optional[int]:
    history_by_idx = {
        int(item["sentence_end_idx"]): item
        for item in history
        if item.get("sentence_end_idx") is not None
    }
    history_sorted = sorted(history_by_idx.values(), key=lambda item: item["sentence_end_idx"])
    sent_idxs = [int(item["sentence_end_idx"]) for item in history_sorted]
    correct_rates = [float(item["correct_rate"]) for item in history_sorted]
    if len(sent_idxs) < 2:
        return None

    intervals: list[tuple[float, int, int]] = []
    for i in range(len(sent_idxs) - 1):
        left_idx = sent_idxs[i]
        right_idx = sent_idxs[i + 1]
        gap = right_idx - left_idx
        if gap <= 1:
            continue
        diff = abs(correct_rates[i + 1] - correct_rates[i])
        slope = diff / gap if gap else 0.0
        intervals.append((slope, left_idx, right_idx))

    intervals.sort(key=lambda item: item[0], reverse=True)
    for _, left_idx, right_idx in intervals:
        candidate = _pick_midpoint(left_idx, right_idx, sent_idxs, min_spacing=min_spacing, n_sent=n_sent)
        if candidate is not None:
            return candidate
    return None


def next_largest_gap_sentence(history: List[Dict[str, Any]], *, min_spacing: int, n_sent: int) -> Optional[int]:
    history_by_idx = {
        int(item["sentence_end_idx"]): item
        for item in history
        if item.get("sentence_end_idx") is not None
    }
    history_sorted = sorted(history_by_idx.values(), key=lambda item: item["sentence_end_idx"])
    sent_idxs = [int(item["sentence_end_idx"]) for item in history_sorted]
    if len(sent_idxs) < 2:
        return None

    gaps: list[tuple[int, int, int]] = []
    for i in range(len(sent_idxs) - 1):
        left_idx = sent_idxs[i]
        right_idx = sent_idxs[i + 1]
        gaps.append((right_idx - left_idx, left_idx, right_idx))

    gaps.sort(key=lambda item: item[0], reverse=True)
    for _, left_idx, right_idx in gaps:
        candidate = _pick_midpoint(left_idx, right_idx, sent_idxs, min_spacing=min_spacing, n_sent=n_sent)
        if candidate is not None:
            return candidate
    return None


def localize_correctness_adaptive_sentences(
    llm: LLM,
    tokenizer: Any,
    prompt: str,
    prompt_messages: Optional[List[Dict[str, Any]]],
    raw_text: str,
    sentences: List[Dict[str, Any]],
    *,
    correct_option: str,
    option_map: dict[str, str],
    n_samples: int,
    coarse_iters: int,
    refinement_iters: int,
    min_step_size: int,
    min_spacing: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    base_seed: int,
    compute_full_score: bool,
) -> dict[str, Any]:
    n_sent = len(sentences)
    if n_sent == 0:
        return {
            "raw_text": raw_text,
            "prompt": prompt,
            "history": [],
            "full_score": None,
        }

    def _prefix_text(sent_end_idx: int) -> str:
        if sent_end_idx <= 0:
            return ""
        end_char = sentences[sent_end_idx - 1]["end"]
        return raw_text[:end_char]

    history: list[dict[str, Any]] = []
    checked: dict[int, dict[str, Any]] = {}
    seed_counter = 0

    def _next_seed() -> int:
        nonlocal seed_counter
        seed_counter += 1
        return base_seed + seed_counter

    def _probe_sentence(sent_end_idx: int, *, seed: Optional[int] = None) -> dict[str, Any]:
        if sent_end_idx in checked:
            return checked[sent_end_idx]

        prefix_text = _prefix_text(sent_end_idx)
        seed_value = seed if seed is not None else _next_seed()
        correct_rate, num_correct, num_valid, generations = sample_correctness_for_prefix(
            llm,
            tokenizer,
            prompt,
            prompt_messages,
            prefix_text,
            correct_option=correct_option,
            option_map=option_map,
            n_samples=n_samples,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            base_seed=seed_value,
        )
        ci_low, ci_high = wilson_interval(num_correct, num_valid)

        if sent_end_idx > 0:
            sent = sentences[sent_end_idx - 1]
            char_span = (sent["start"], sent["end"])
            sent_text = sent["text"]
            sent_idx_inclusive = sent_end_idx - 1
        else:
            char_span = (0, 0)
            sent_text = ""
            sent_idx_inclusive = None

        probe = {
            "sentence_end_idx": sent_end_idx,
            "sentence_idx_inclusive": sent_idx_inclusive,
            "char_span": char_span,
            "sentence_text": sent_text,
            "target_sentence_text": sent_text,
            "prompt": prompt,
            "prefix_text": prefix_text,
            "correct_rate": correct_rate,
            "num_correct": num_correct,
            "num_valid": num_valid,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "seed": seed_value,
            "generations": generations,
        }
        history.append(probe)
        checked[sent_end_idx] = probe
        return probe

    full_score = None
    full_probe = _probe_sentence(n_sent, seed=base_seed)
    if compute_full_score:
        full_score = full_probe

    _probe_sentence(1)

    left = 0
    right = n_sent
    earliest_idx = None
    earliest_stats = None
    steps = 0

    while left < right and steps < coarse_iters and (right - left) > min_step_size:
        steps += 1
        mid = (left + right) // 2
        probe = _probe_sentence(mid)
        if probe["num_valid"] <= 0:
            break
        if probe["correct_rate"] >= 0.5:
            left = mid
        else:
            earliest_idx = mid
            earliest_stats = probe
            right = mid

    for _ in range(refinement_iters):
        next_idx = next_high_gradient_sentence(history, min_spacing=min_spacing, n_sent=n_sent)
        if next_idx is None:
            next_idx = next_largest_gap_sentence(history, min_spacing=min_spacing, n_sent=n_sent)
        if next_idx is None:
            break
        _probe_sentence(next_idx)

    history = sorted(history, key=lambda item: item["sentence_end_idx"])
    candidate_prefix_end_idxs = sorted(
        {int(item["sentence_end_idx"]) for item in history if item.get("sentence_end_idx") is not None}
    )
    candidate_sentence_idxs = sorted({idx - 1 for idx in candidate_prefix_end_idxs if idx > 0})

    return {
        "raw_text": raw_text,
        "prompt": prompt,
        "history": history,
        "full_score": full_score,
        "left_sentence_end_idx": left,
        "right_sentence_end_idx": earliest_idx,
        "earliest_low_correct_prefix": earliest_stats,
        "candidate_sentence_idxs": candidate_sentence_idxs,
        "candidate_prefix_end_idxs": candidate_prefix_end_idxs,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Batch sentence-level commitment accuracy localization.")
    parser.add_argument("--examples_path", type=str, required=True)
    parser.add_argument("--sentences_path", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--jsonl_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.2)
    parser.add_argument("--max_new_tokens", type=int, default=3000)
    parser.add_argument("--base_seed", type=int, default=1234)
    parser.add_argument("--mode", type=str, default="prefix", choices=["prefix", "sentence_only"])
    parser.add_argument("--method", type=str, default="adaptive", choices=["adaptive", "full"])
    parser.add_argument("--coarse_iters", type=int, default=8)
    parser.add_argument("--refinement_iters", type=int, default=8)
    parser.add_argument("--min_step_size", type=int, default=1)
    parser.add_argument("--min_spacing", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--label_filter", type=str, choices=LABEL_FILTER_CHOICES, default=LABEL_FILTER_ALL)
    parser.add_argument("--only_correct", action="store_true", default=False)
    parser.add_argument("--only_incorrect", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--rebalance_pending_shards", action="store_true", default=False)
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--text_field", type=str, default="action_reasoning")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    args = parser.parse_args(argv)

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard_id must be in [0, num_shards)")

    label_filter = normalize_label_filter(
        args.label_filter,
        only_correct=args.only_correct,
        only_incorrect=args.only_incorrect,
    )

    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    example_list = [example for example in read_jsonl(args.examples_path) if keep_record_for_label_filter(example, label_filter)]
    if args.limit:
        example_list = example_list[: args.limit]
    total_candidate_examples = len(example_list)

    if args.rebalance_pending_shards:
        if out_dir is None:
            print("Ignoring --rebalance_pending_shards because --out_dir is not set.")
        elif args.overwrite:
            print("Ignoring --rebalance_pending_shards because --overwrite was requested.")
        else:
            example_list, existing_outputs = _filter_pending_examples(
                example_list,
                out_dir=out_dir,
                overwrite=args.overwrite,
            )
            print(
                "Pending-aware sharding: "
                f"{len(example_list)} pending / {total_candidate_examples} total examples "
                f"({existing_outputs} already localized)."
            )

    example_list = _shard_examples(example_list, shard_id=args.shard_id, num_shards=args.num_shards)
    total_examples = len(example_list)
    print(f"Shard {args.shard_id}/{args.num_shards}: {total_examples} examples (label_filter={label_filter})")
    if total_examples == 0:
        print("No examples to process for this shard.")
        return

    sentences_by_example = _load_sentences(args.sentences_path)

    visible_gpu_count = max(1, torch.cuda.device_count())
    tensor_parallel_size = max(1, int(args.tensor_parallel_size))
    if tensor_parallel_size > visible_gpu_count:
        raise ValueError(
            f"tensor_parallel_size={tensor_parallel_size} exceeds visible GPU count={visible_gpu_count}."
        )

    LLM, _ = _import_vllm()
    llm = LLM(
        model=args.model_name,
        max_model_len=args.max_new_tokens,
        seed=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
    )
    tokenizer = llm.get_tokenizer()

    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else None
    if jsonl_path and args.num_shards > 1:
        jsonl_path = jsonl_path.with_suffix(f".shard{args.shard_id}.jsonl")

    jsonl_fh = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_fh = jsonl_path.open("w", encoding="utf-8")

    processed = 0
    skipped = 0

    for idx, example in enumerate(example_list):
        example_id = _record_id(example)

        out_path = None
        if out_dir:
            out_path = _example_output_path(out_dir, example)
            if out_path.exists() and not args.overwrite:
                continue

        raw_text = _extract_raw_text(example, args.text_field)
        prompt = example.get("prompt")
        prompt_messages = example.get("messages")
        correct_option = example.get("correct_option")
        option_map = parse_mathqa_options(example.get("options", ""))

        if not raw_text or not isinstance(prompt, str) or not prompt.strip():
            skipped += 1
            continue
        if not isinstance(correct_option, str) or correct_option not in option_map:
            skipped += 1
            continue

        sentence_records = sentences_by_example.get(example_id)
        if sentence_records:
            sentences = [
                {
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "text": item.get("sentence_text"),
                }
                for item in sentence_records
                if item.get("start") is not None
                and item.get("end") is not None
                and isinstance(item.get("sentence_text"), str)
            ]
        else:
            sentences = split_sentence_spans(raw_text)

        if not sentences:
            skipped += 1
            continue

        if args.method == "full":
            history = localize_correctness_by_sentence(
                llm,
                tokenizer,
                prompt,
                prompt_messages,
                raw_text,
                sentences,
                correct_option=correct_option,
                option_map=option_map,
                n_samples=args.n_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                mode=args.mode,
            )
            record = {
                "example_id": example_id,
                "base_example_id": example.get("base_example_id"),
                "answer_status": example.get("answer_status"),
                "is_correct": example.get("is_correct"),
                "correct_option": correct_option,
                "raw_text": raw_text,
                "prompt": prompt,
                "history": history,
            }
        else:
            record = localize_correctness_adaptive_sentences(
                llm,
                tokenizer,
                prompt,
                prompt_messages,
                raw_text,
                sentences,
                correct_option=correct_option,
                option_map=option_map,
                n_samples=args.n_samples,
                coarse_iters=args.coarse_iters,
                refinement_iters=args.refinement_iters,
                min_step_size=args.min_step_size,
                min_spacing=args.min_spacing,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                max_new_tokens=args.max_new_tokens,
                base_seed=args.base_seed,
                compute_full_score=True,
            )
            record["example_id"] = example_id
            record["base_example_id"] = example.get("base_example_id")
            record["answer_status"] = example.get("answer_status")
            record["is_correct"] = example.get("is_correct")
            record["correct_option"] = correct_option

        if out_path:
            out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if jsonl_fh:
            jsonl_fh.write(json.dumps(record) + "\n")
            if args.flush_every and ((processed + 1) % args.flush_every == 0):
                jsonl_fh.flush()

        processed += 1
        if args.log_every and (idx + 1) % args.log_every == 0:
            print(
                f"Processed {idx + 1}/{total_examples} examples "
                f"(shard {args.shard_id}, kept={processed}, skipped={skipped})"
            )

    if jsonl_fh:
        jsonl_fh.close()

    if out_dir and jsonl_path:
        print(f"Batch localization complete. Outputs in {out_dir} and {jsonl_path}")
    elif out_dir:
        print(f"Batch localization complete. Outputs in {out_dir}")
    elif jsonl_path:
        print(f"Batch localization complete. Output in {jsonl_path}")
    else:
        print("Batch localization complete.")


if __name__ == "__main__":
    main()
