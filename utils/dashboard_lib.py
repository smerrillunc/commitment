from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None

from sentence_pipeline import split_sentence_spans


DEFAULT_LOCALIZATION_ROOT = "/playpen-ssd/smerrill/commitment/results/localization"
DEFAULT_SCAN_LIMIT = 500


def _safe_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def list_models(localization_root: str | Path) -> List[str]:
    root = Path(localization_root)
    if not root.exists():
        return []
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def list_run_tags(localization_root: str | Path, model: str) -> List[str]:
    model_root = Path(localization_root) / model
    if not model_root.exists():
        return []
    return sorted(path.name for path in model_root.iterdir() if path.is_dir())


def list_localization_paths(
    localization_root: str | Path,
    model: str,
    run_tag: str,
    *,
    limit: Optional[int] = DEFAULT_SCAN_LIMIT,
    filename_filter: str = "",
    status_filter: str = "all",
) -> List[Path]:
    run_root = Path(localization_root) / model / run_tag / "localization"
    if not run_root.exists():
        return []

    needle = filename_filter.strip().lower()
    out: List[Path] = []
    for path in sorted(run_root.glob("*.json")):
        name_lower = path.name.lower()
        if needle and needle not in name_lower:
            continue
        if status_filter == "correct" and "__correct" not in name_lower:
            continue
        if status_filter == "incorrect" and "__incorrect" not in name_lower:
            continue
        out.append(path)
        if limit is not None and len(out) >= int(limit):
            break
    return out


def load_local_record(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def preview_value(value: object, limit: int = 120) -> object:
    if isinstance(value, str):
        compact = " ".join(value.split())
        if len(compact) > limit:
            return compact[: limit - 3] + "..."
        return compact
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    return type(value).__name__


def summarize_record_schema(record: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    top_level_df = pd.DataFrame(
        [
            {
                "key": key,
                "python_type": type(value).__name__,
                "preview": preview_value(value),
            }
            for key, value in record.items()
        ]
    ).sort_values("key").reset_index(drop=True)

    history = record.get("history") or []
    history_keys = sorted({key for row in history for key in row.keys()})
    history_df = pd.DataFrame({"history_key": history_keys})

    generations = []
    for row in history:
        generations.extend(row.get("generations") or [])
    generation_keys = sorted({key for row in generations for key in row.keys()})
    generation_df = pd.DataFrame({"generation_key": generation_keys})
    return top_level_df, history_df, generation_df


def normalize_history(history: Iterable[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for probe in history:
        out = dict(probe)
        sent_end = out.get("sentence_end_idx")
        sent_idx = out.get("sentence_idx_inclusive")
        if sent_idx is None:
            sent_idx = out.get("sentence_idx")
        if sent_end is None and sent_idx is not None:
            sent_end = int(sent_idx) + 1
        if sent_end is not None:
            sent_idx = int(sent_end) - 1 if int(sent_end) > 0 else None
        out["sentence_end_idx"] = int(sent_end) if sent_end is not None else None
        out["sentence_idx"] = int(sent_idx) if sent_idx is not None else None

        char_span = out.get("char_span")
        if isinstance(char_span, list):
            char_span = tuple(char_span)
        if isinstance(char_span, tuple) and len(char_span) == 2:
            out["char_span"] = (int(char_span[0]), int(char_span[1]))
        else:
            out["char_span"] = None

        out["correct_rate"] = _safe_float(out.get("correct_rate"))
        out["num_correct"] = _safe_int(out.get("num_correct"))
        out["num_valid"] = _safe_int(out.get("num_valid"))
        out["ci_low"] = _safe_float(out.get("ci_low"))
        out["ci_high"] = _safe_float(out.get("ci_high"))
        normalized.append(out)
    return normalized


def flatten_history(history: List[Dict], raw_text: str) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        generations = probe.get("generations") or []
        for sample_id, generation in enumerate(generations):
            rows.append(
                {
                    "step_id": step_id,
                    "sentence_end_idx": probe.get("sentence_end_idx"),
                    "sentence_idx": probe.get("sentence_idx"),
                    "sample_id": sample_id,
                    "correct_rate_step": probe.get("correct_rate"),
                    "num_correct_step": probe.get("num_correct"),
                    "num_valid_step": probe.get("num_valid"),
                    "gen_text": generation.get("gen_text"),
                    "full_generation_text": generation.get("full_generation_text"),
                    "is_correct": generation.get("is_correct"),
                    "parse_error": generation.get("parse_error"),
                    "parse_success": generation.get("parse_success"),
                    "final_answer": generation.get("final_answer"),
                    "predicted_option": generation.get("predicted_option"),
                    "matched_final_answer_block": generation.get("matched_final_answer_block"),
                    "sentence_text": probe.get("sentence_text"),
                    "char_span": probe.get("char_span"),
                    "raw_text": raw_text,
                }
            )
    return pd.DataFrame(rows)


def build_stats(history: List[Dict]) -> pd.DataFrame:
    rows = []
    for step_id, probe in enumerate(history):
        sent_end = probe.get("sentence_end_idx")
        if sent_end is None:
            continue
        num_correct = int(probe.get("num_correct") or 0)
        num_valid = int(probe.get("num_valid") or 0)
        correct_rate = probe.get("correct_rate")
        if correct_rate is None and num_valid > 0:
            correct_rate = num_correct / num_valid
        p_value = binomtest(num_correct, num_valid, p=0.5).pvalue if (binomtest and num_valid > 0) else np.nan
        rows.append(
            {
                "step_id": step_id,
                "sentence_end_idx": sent_end,
                "sentence_idx": probe.get("sentence_idx"),
                "correct_rate": correct_rate,
                "num_correct": num_correct,
                "num_valid": num_valid,
                "ci_low": probe.get("ci_low"),
                "ci_high": probe.get("ci_high"),
                "p_value": p_value,
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("sentence_end_idx").reset_index(drop=True)
    return df


def build_sentence_spans(raw_text: str) -> Tuple[List[Dict], Dict[int, Dict]]:
    spans: List[Dict] = []
    for idx, span in enumerate(split_sentence_spans(raw_text or "")):
        start = _safe_int(span.get("start"))
        end = _safe_int(span.get("end"))
        if start is None or end is None or end <= start:
            continue
        spans.append(
            {
                "sentence_idx": idx,
                "start": start,
                "end": end,
                "text": span.get("text"),
            }
        )
    spans.sort(key=lambda item: (item.get("start", 0), item.get("sentence_idx", 0)))
    span_map = {int(item["sentence_idx"]): item for item in spans}
    return spans, span_map


def resolve_sentence_span(
    sentence_idx: Optional[int],
    sentence_span_map: Dict[int, Dict],
    probe: Optional[Dict] = None,
) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if sentence_idx is None or sentence_idx < 0:
        return None, None, None

    idx = int(sentence_idx)
    span = sentence_span_map.get(idx)
    if span:
        return span.get("text"), span.get("start"), span.get("end")

    if probe is None:
        return None, None, None

    char_span = probe.get("char_span")
    if isinstance(char_span, (list, tuple)) and len(char_span) == 2:
        return probe.get("sentence_text"), int(char_span[0]), int(char_span[1])
    return probe.get("sentence_text"), None, None


def compute_low_accuracy_sentence_idx(
    right_sentence_end_idx: Optional[int],
    df_stats: pd.DataFrame,
) -> Optional[int]:
    if right_sentence_end_idx is not None:
        idx = int(right_sentence_end_idx) - 1
        if idx >= 0:
            return idx

    if len(df_stats) == 0:
        return None

    candidates = df_stats[
        (df_stats["correct_rate"].notna())
        & (df_stats["num_valid"] > 0)
        & (df_stats["correct_rate"] <= 0.5)
    ]
    if len(candidates) == 0:
        return None
    return int(candidates.sort_values("sentence_idx").iloc[0]["sentence_idx"])


def plot_sentence_localization(
    df_stats: pd.DataFrame,
    low_accuracy_sentence_idx: Optional[int] = None,
    right_sentence_end_idx: Optional[int] = None,
) -> Tuple[plt.Figure, pd.DataFrame]:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    if len(df_stats) == 0:
        ax.set_title("No stats available")
        return fig, df_stats

    x = df_stats["sentence_idx"]
    y = df_stats["correct_rate"]
    ax.plot(x, y, color="#666666", linewidth=2, label="Correct rate")

    if low_accuracy_sentence_idx is not None:
        after_mask = x >= low_accuracy_sentence_idx
        before_mask = ~after_mask
    else:
        after_mask = pd.Series([False] * len(df_stats))
        before_mask = ~after_mask

    ax.scatter(x[before_mask], y[before_mask], color="#1f77b4", s=40, label="Before low-accuracy")
    if after_mask.any():
        ax.scatter(x[after_mask], y[after_mask], color="#d62728", s=40, label="Low-accuracy or after")

    if df_stats["ci_low"].notna().any():
        ax.fill_between(x, df_stats["ci_low"], df_stats["ci_high"], alpha=0.2, label="95% CI")

    ax.axhline(0.5, linestyle="--", linewidth=2, label="50% threshold")
    if right_sentence_end_idx is not None:
        ax.axvline(
            int(right_sentence_end_idx) - 1,
            linestyle="-.",
            linewidth=2,
            label=f"Stored low-accuracy boundary @ {int(right_sentence_end_idx) - 1}",
        )
    elif low_accuracy_sentence_idx is not None:
        ax.axvline(
            int(low_accuracy_sentence_idx),
            linestyle="-.",
            linewidth=2,
            label=f"First <= 0.5 @ {int(low_accuracy_sentence_idx)}",
        )

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Correct rate")
    if len(df_stats) and df_stats["p_value"].notna().any():
        ax.set_title(f"Sentence-level answer-accuracy localization\nmin p = {df_stats['p_value'].min():.1e}")
    else:
        ax.set_title("Sentence-level answer-accuracy localization")
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, df_stats


def _escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_html_attr(text: str) -> str:
    return _escape_html(text).replace('"', "&quot;").replace("'", "&#39;")


def strip_code_fences(text: str) -> str:
    if not text:
        return ""
    out = str(text).strip()
    out = re.sub(r"^```[A-Za-z0-9_+-]*\n?", "", out)
    out = re.sub(r"\n?```$", "", out)
    return out.strip()


def generation_bucket(generation: Dict) -> str:
    is_correct = generation.get("is_correct")
    if is_correct is True:
        return "correct"
    if is_correct is False:
        return "incorrect"
    return "invalid"


def _append_colored_segment(
    html_parts: List[str],
    full_text: str,
    start: int,
    end: int,
    color: str,
) -> None:
    if start >= end:
        return
    html_parts.append(f"<span style='color:{color}'>{_escape_html(full_text[start:end])}</span>")


def render_prefix_generation_with_sentence_indices_html(
    prefix_text: str,
    gen_text: str,
    selected_sentence_idx: Optional[int],
    sentence_labels: Optional[Dict[int, str]] = None,
) -> str:
    full_text = (prefix_text or "") + (gen_text or "")
    if not full_text:
        return "<i>No text available for rendering.</i>"

    spans = [
        span for span in split_sentence_spans(full_text)
        if span.get("start") is not None and span.get("end") is not None
    ]
    spans.sort(key=lambda span: span.get("start", 0))
    if not spans:
        return (
            "<div style='background-color:white; padding:5px; line-height:1.5; "
            "white-space:pre-wrap; font-family:monospace'>"
            f"{_escape_html(full_text)}"
            "</div>"
        )

    if selected_sentence_idx is not None:
        selected_sentence_idx = max(0, min(int(selected_sentence_idx), len(spans) - 1))

    html_parts: List[str] = []
    last = 0

    def color_for_idx(idx: int) -> str:
        if selected_sentence_idx is None:
            return "black"
        if idx < selected_sentence_idx:
            return "blue"
        if idx > selected_sentence_idx:
            return "green"
        return "black"

    for idx, span in enumerate(spans):
        start = int(span["start"])
        end = int(span["end"])
        color = color_for_idx(idx)
        if start > last:
            _append_colored_segment(html_parts, full_text, last, start, color)
        _append_colored_segment(html_parts, full_text, start, end, color)
        label = sentence_labels.get(idx) if sentence_labels else None
        marker = f"{idx + 1}"
        if label:
            marker = f"{marker} [{label}]"
        html_parts.append(
            f"<sup style='font-size:0.7em;color:{color}'>{_escape_html(marker)}</sup> "
        )
        last = end

    if last < len(full_text):
        tail_color = color_for_idx(len(spans) - 1) if spans else "black"
        _append_colored_segment(html_parts, full_text, last, len(full_text), tail_color)

    return (
        "<div style='background-color:white; padding:5px; line-height:1.5; "
        "white-space:pre-wrap; font-family:monospace'>"
        + "".join(html_parts)
        + "</div>"
    )


def build_generation_sentence_labels(
    prefix_text: str,
    prefix_mode: str,
    resolved_sentence_idx: Optional[int],
) -> Tuple[Optional[int], Dict[int, str]]:
    prefix_spans = [
        span for span in split_sentence_spans(prefix_text or "")
        if span.get("start") is not None and span.get("end") is not None
    ]
    prefix_spans.sort(key=lambda span: span.get("start", 0))
    display_selected_idx = (len(prefix_spans) - 1) if prefix_spans else None
    sentence_labels: Dict[int, str] = {}

    if not prefix_spans:
        return display_selected_idx, sentence_labels

    if prefix_mode == "sentence" and resolved_sentence_idx is not None:
        global_offset = int(resolved_sentence_idx) - (len(prefix_spans) - 1)
    else:
        global_offset = 0

    for local_idx in range(len(prefix_spans)):
        global_idx = global_offset + local_idx
        sentence_labels[local_idx] = f"S_{global_idx + 1}"

    return display_selected_idx, sentence_labels


def sentence_selector_label(sentence_idx: int, sentence_span_map: Dict[int, Dict]) -> str:
    idx = _safe_int(sentence_idx)
    if idx is None:
        return str(sentence_idx)

    span = sentence_span_map.get(idx, {})
    sentence_text = span.get("text") if isinstance(span, dict) else ""
    if not isinstance(sentence_text, str):
        sentence_text = ""
    sentence_text = " ".join(sentence_text.split())
    if len(sentence_text) > 120:
        sentence_text = sentence_text[:117] + "..."

    if sentence_text:
        return f"{idx}: {sentence_text}"
    return str(idx)
