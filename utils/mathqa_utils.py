from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SYSTEM_PROMPT = """
You are solving a multiple-choice math problem.

Think through the problem carefully. Your response must end with a JSON object on the final line
that uses exactly this schema:
{"Final Answer": "a"}

Rules:
- Put all reasoning before the JSON object.
- Do not use markdown code fences.
- Do not add any text after the JSON object.
- The final answer should be the single best option letter from a, b, c, d, or e.
"""

USER_PROMPT_TEMPLATE = """
Solve this MathQA multiple-choice problem.

Problem:
{problem}

Options:
{options}
"""

FINAL_ANSWER_JSON_RE = re.compile(
    r'\{\s*"Final Answer"\s*:\s*(?P<value>"(?:\\.|[^"\\])*"|[^{}]+?)\s*\}',
    flags=re.DOTALL,
)
OPTION_RE = re.compile(
    r"([A-Ea-e])\s*[\)\].:]\s*(.*?)(?=(?:\s*,\s*[A-Ea-e]\s*[\)\].:])|$)",
    flags=re.DOTALL,
)


def slugify(value: str) -> str:
    value = value.replace("/", "__")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_")


def append_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    in_path = Path(path)
    if not in_path.exists():
        return records
    with in_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_whitespace(text: Any) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_mathqa_options(options_text: str) -> dict[str, str]:
    cleaned = normalize_whitespace(options_text)
    if not cleaned:
        return {}

    matches = list(OPTION_RE.finditer(cleaned))
    if not matches:
        return {}

    option_map: dict[str, str] = {}
    for match in matches:
        label = match.group(1).lower()
        text = match.group(2).strip().strip(",").strip()
        if text:
            option_map[label] = text
    return option_map


def format_options_block(options_text: str) -> str:
    option_map = parse_mathqa_options(options_text)
    if not option_map:
        return normalize_whitespace(options_text)
    return "\n".join(f"{label}) {text}" for label, text in option_map.items())


def build_combined_question(problem: str, options_text: str) -> str:
    problem = normalize_whitespace(problem)
    options_block = format_options_block(options_text)
    return f"{problem}\n\nOptions:\n{options_block}".strip()


def build_prompt_messages(problem: str, options_text: str) -> list[dict[str, str]]:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        problem=normalize_whitespace(problem),
        options=format_options_block(options_text),
    ).strip()
    return [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt},
    ]


def render_prompt(
    problem: str,
    options_text: str,
    tokenizer: Any,
    *,
    use_chat_template: bool = True,
) -> str:
    messages = build_prompt_messages(problem, options_text)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return (
        f"System:\n{SYSTEM_PROMPT.strip()}\n\n"
        f"User:\n{messages[-1]['content']}\n\n"
        "Assistant:\n"
    )


def clean_reasoning(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"^<think>\s*", "", text)
    text = re.sub(r"\s*</think>\s*$", "", text)
    return text.strip()


def coerce_final_answer_value(raw_value: str) -> str | None:
    raw_value = (raw_value or "").strip().rstrip(",").strip()
    if not raw_value:
        return None

    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        if raw_value.startswith('"') and raw_value.endswith('"') and len(raw_value) >= 2:
            value = raw_value[1:-1]
        else:
            value = raw_value

        value = value.replace(r"\\n", " ")
        value = value.replace(r"\\t", " ")
        value = value.replace(r"\\r", " ")
        value = value.replace(r'\\"', '"')
        value = value.replace(r"\\\\", r"\\")
        value = re.sub(r"\s+", " ", value)
        return value.strip() or None

    return str(parsed_value).strip() or None


def parse_model_output(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text or ""
    matches = list(FINAL_ANSWER_JSON_RE.finditer(raw_text))
    if matches:
        match = matches[-1]
        candidate = match.group(0)
        reasoning = clean_reasoning(raw_text[: match.start()])

        try:
            parsed_json = json.loads(candidate)
        except json.JSONDecodeError as exc:
            final_answer = coerce_final_answer_value(match.group("value"))
            return {
                "Reasoning": reasoning,
                "Final Answer": final_answer,
                "Parse Error": f"{type(exc).__name__}: {exc}",
                "parse_success": False,
                "raw_completion": raw_text,
                "matched_final_answer_block": candidate,
            }

        final_answer = str(parsed_json.get("Final Answer", "")).strip() or None
        return {
            "Reasoning": reasoning,
            "Final Answer": final_answer,
            "Parse Error": None,
            "parse_success": True,
            "raw_completion": raw_text,
            "matched_final_answer_block": candidate,
        }

    fallback = re.search(r"Final Answer\s*[:=]\s*(?P<answer>.+)$", raw_text, flags=re.DOTALL)
    if fallback:
        reasoning = clean_reasoning(raw_text[: fallback.start()])
        final_answer = fallback.group("answer").strip().strip("`").strip().strip('"').strip()
        return {
            "Reasoning": reasoning,
            "Final Answer": final_answer or None,
            "Parse Error": "No valid final JSON object found; used fallback Final Answer parsing.",
            "parse_success": False,
            "raw_completion": raw_text,
            "matched_final_answer_block": None,
        }

    return {
        "Reasoning": clean_reasoning(raw_text),
        "Final Answer": None,
        "Parse Error": "No final answer JSON object or fallback pattern found.",
        "parse_success": False,
        "raw_completion": raw_text,
        "matched_final_answer_block": None,
    }


def extract_gold_choice(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"\b([A-Ea-e])\b", str(value))
    if not match:
        return None
    return match.group(1).lower()


def _normalize_option_text(text: Any) -> str:
    value = normalize_whitespace(text)
    value = value.strip("`").strip().strip('"').strip("'").strip()
    value = value.lower()
    value = value.replace("$", "")
    value = value.replace(",", "")
    value = value.rstrip(".")
    value = re.sub(r"\s+", " ", value)
    return value


def extract_choice_label(text: Any, option_map: dict[str, str] | None = None) -> str | None:
    if text is None:
        return None

    raw_value = normalize_whitespace(text)
    if not raw_value:
        return None

    stripped = raw_value.strip("`").strip().strip('"').strip("'").strip()

    direct_patterns = [
        r"(?i)\boption\s*([a-e])\b",
        r"(?i)\banswer(?:\s+is)?\s*[:=-]?\s*([a-e])\b",
        r"(?i)^\(?\s*([a-e])\s*\)?$",
        r"(?i)^\(?\s*([a-e])\s*[\)\].:-].*$",
    ]
    for pattern in direct_patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1).lower()

    compact = re.sub(r"[^A-Za-z]", "", stripped).lower()
    if compact in {"a", "b", "c", "d", "e"}:
        return compact

    if option_map:
        normalized_candidate = _normalize_option_text(stripped)
        for label, option_text in option_map.items():
            option_norm = _normalize_option_text(option_text)
            if normalized_candidate == option_norm:
                return label

    return None


def normalize_final_answer(text: Any, option_map: dict[str, str] | None = None) -> str | None:
    label = extract_choice_label(text, option_map=option_map)
    if label is not None:
        return label

    value = _normalize_option_text(text)
    return value or None


def gold_option_text(correct_label: str | None, option_map: dict[str, str]) -> str | None:
    if not correct_label:
        return None
    return option_map.get(correct_label.lower())

