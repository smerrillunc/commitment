#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

THIS_FILE = Path(__file__).resolve()
COMMITMENT_ROOT = THIS_FILE.parents[1]
UTILS_DIR = COMMITMENT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from mathqa_utils import (
    append_jsonl,
    build_combined_question,
    build_prompt_messages,
    extract_gold_choice,
    format_options_block,
    gold_option_text,
    normalize_final_answer,
    parse_mathqa_options,
    parse_model_output,
    read_jsonl,
    render_prompt,
    slugify,
)


def _import_vllm():
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("This script requires vLLM in the active environment.") from exc
    return LLM, SamplingParams


@dataclass
class QuestionState:
    dataset_index: int
    split: str
    problem: str
    options: str
    category: str | None
    correct_option: str
    option_map: dict[str, str]
    prompt: str
    messages: list[dict[str, str]]
    found_correct: bool = False
    found_incorrect: bool = False
    attempted_samples: int = 0
    next_sample_id: int = 0

    @property
    def base_example_id(self) -> str:
        return f"mathqa/{self.split}/{self.dataset_index:06d}"

    @property
    def complete(self) -> bool:
        return self.found_correct and self.found_incorrect


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Commitment miner for MathQA. For each selected question, it samples until it finds "
            "one correct and one incorrect reasoning trace so we can localize answer accuracy."
        )
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="math_qa")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_questions", type=int, default=100)
    parser.add_argument("--samples_per_round", type=int, default=4)
    parser.add_argument("--prompt_batch_size", type=int, default=16)
    parser.add_argument("--max_samples_per_question", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=3000)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--use_chat_template", action="store_true", default=True)
    parser.add_argument("--no_use_chat_template", dest="use_chat_template", action="store_false")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no_resume", dest="resume", action="store_false")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    parser.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    return parser


def _load_existing_status(output_path: Path) -> dict[str, dict[str, Any]]:
    status_by_base: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(output_path):
        base_id = record.get("base_example_id")
        if not base_id:
            continue
        info = status_by_base.setdefault(
            base_id,
            {
                "found_correct": False,
                "found_incorrect": False,
                "next_sample_id": 0,
            },
        )
        if record.get("is_correct") is True:
            info["found_correct"] = True
        if record.get("is_correct") is False:
            info["found_incorrect"] = True
        sample_id = record.get("sample_id")
        if isinstance(sample_id, int):
            info["next_sample_id"] = max(info["next_sample_id"], sample_id + 1)
    return status_by_base


def _make_record(
    *,
    args: argparse.Namespace,
    question_state: QuestionState,
    parsed: dict[str, Any],
    raw_completion: str,
    sample_id: int,
) -> dict[str, Any]:
    predicted_option = normalize_final_answer(
        parsed["Final Answer"],
        option_map=question_state.option_map,
    )
    is_correct = bool(
        predicted_option is not None and predicted_option == question_state.correct_option
    )
    answer_status = "correct" if is_correct else "incorrect"
    example_id = f"{question_state.base_example_id}/{answer_status}"

    return {
        "example_id": example_id,
        "base_example_id": question_state.base_example_id,
        "dataset_name": args.dataset_name,
        "split": question_state.split,
        "dataset_index": int(question_state.dataset_index),
        "model_name": args.model_name,
        "sample_id": int(sample_id),
        "answer_status": answer_status,
        "is_correct": is_correct,
        "problem": question_state.problem,
        "options": question_state.options,
        "question": build_combined_question(question_state.problem, question_state.options),
        "prompt": question_state.prompt,
        "messages": question_state.messages,
        "problem_category": question_state.category,
        "correct_option": question_state.correct_option,
        "correct_option_text": gold_option_text(question_state.correct_option, question_state.option_map),
        "predicted_option": predicted_option,
        "raw_completion": raw_completion,
        "action_reasoning": parsed["Reasoning"],
        "action_raw_text": raw_completion,
        "reasoning": parsed["Reasoning"],
        "final_answer": parsed["Final Answer"],
        "final_answer_normalized": predicted_option,
        "parse_success": bool(parsed["parse_success"]),
        "parse_error": parsed["Parse Error"],
        "matched_final_answer_block": parsed["matched_final_answer_block"],
        "completion_chars": int(len(raw_completion)),
        "options_block": format_options_block(question_state.options),
    }


def _iter_candidate_indices(dataset_size: int, seed: int) -> list[int]:
    indices = list(range(dataset_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    random.seed(args.seed)
    LLM, SamplingParams = _import_vllm()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "commitment_samples.jsonl"
    summary_path = out_dir / "run_summary.json"

    run_name = f"mathqa_commitment_{slugify(args.model_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_config = vars(args).copy()
    run_config["run_name"] = run_name
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        trust_remote_code=args.trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
    )
    llm = LLM(
        model=args.model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )

    existing_status = _load_existing_status(samples_path) if args.resume else {}

    candidate_indices = _iter_candidate_indices(len(dataset), args.seed)
    selected_states: list[QuestionState] = []
    for dataset_index in candidate_indices:
        row = dataset[int(dataset_index)]
        correct_option = extract_gold_choice(row.get("correct"))
        option_map = parse_mathqa_options(row.get("options", ""))
        if correct_option is None or correct_option not in option_map:
            continue

        messages = build_prompt_messages(row["Problem"], row["options"])
        prompt = render_prompt(
            row["Problem"],
            row["options"],
            tokenizer,
            use_chat_template=args.use_chat_template,
        )

        state = QuestionState(
            dataset_index=int(dataset_index),
            split=args.split,
            problem=row["Problem"],
            options=row["options"],
            category=row.get("category"),
            correct_option=correct_option,
            option_map=option_map,
            prompt=prompt,
            messages=messages,
        )
        existing = existing_status.get(state.base_example_id)
        if existing:
            state.found_correct = bool(existing.get("found_correct"))
            state.found_incorrect = bool(existing.get("found_incorrect"))
            state.next_sample_id = int(existing.get("next_sample_id", 0))
        selected_states.append(state)
        if len(selected_states) >= args.num_questions:
            break

    if len(selected_states) < args.num_questions:
        print(
            f"Warning: only found {len(selected_states)} usable questions "
            f"for requested num_questions={args.num_questions}."
        )

    sampling_params = SamplingParams(
        n=args.samples_per_round,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    completed_before = sum(1 for state in selected_states if state.complete)
    appended_records = 0
    processed_rounds = 0

    while True:
        pending = [
            state
            for state in selected_states
            if not state.complete and state.attempted_samples < args.max_samples_per_question
        ]
        if not pending:
            break

        batch = pending[: args.prompt_batch_size]
        prompts = [state.prompt for state in batch]
        outputs = llm.generate(prompts, sampling_params)
        processed_rounds += 1

        batch_records: list[dict[str, Any]] = []
        for state, output in zip(batch, outputs):
            for sample_output in output.outputs:
                sample_id = state.next_sample_id
                state.next_sample_id += 1
                state.attempted_samples += 1

                raw_completion = sample_output.text
                parsed = parse_model_output(raw_completion)
                record = _make_record(
                    args=args,
                    question_state=state,
                    parsed=parsed,
                    raw_completion=raw_completion,
                    sample_id=sample_id,
                )

                if record["is_correct"] and not state.found_correct:
                    state.found_correct = True
                    batch_records.append(record)
                elif (record["is_correct"] is False) and not state.found_incorrect:
                    state.found_incorrect = True
                    batch_records.append(record)

                if state.complete or state.attempted_samples >= args.max_samples_per_question:
                    break

        append_jsonl(samples_path, batch_records)
        appended_records += len(batch_records)

        completed_now = sum(1 for state in selected_states if state.complete)
        if args.log_every and (processed_rounds % args.log_every == 0):
            print(
                f"Rounds={processed_rounds} complete_questions={completed_now}/{len(selected_states)} "
                f"new_records={appended_records}"
            )

    complete_questions = sum(1 for state in selected_states if state.complete)
    summary = {
        "run_name": run_name,
        "dataset_name": args.dataset_name,
        "split": args.split,
        "model_name": args.model_name,
        "requested_questions": args.num_questions,
        "selected_questions": len(selected_states),
        "completed_before_resume": completed_before,
        "completed_after_run": complete_questions,
        "new_records_written": appended_records,
        "output_path": str(samples_path),
        "max_samples_per_question": args.max_samples_per_question,
        "samples_per_round": args.samples_per_round,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote samples: {samples_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
