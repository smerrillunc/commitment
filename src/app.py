from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import streamlit as st

THIS_FILE = Path(__file__).resolve()
COMMITMENT_ROOT = THIS_FILE.parents[1]
UTILS_DIR = COMMITMENT_ROOT / "utils"
if str(UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(UTILS_DIR))

from dashboard_lib import (
    DEFAULT_LOCALIZATION_ROOT,
    DEFAULT_SCAN_LIMIT,
    build_generation_sentence_labels,
    build_sentence_spans,
    build_stats,
    compute_low_accuracy_sentence_idx,
    flatten_history,
    generation_bucket,
    list_localization_paths,
    list_models,
    list_run_tags,
    load_local_record,
    normalize_history,
    plot_sentence_localization,
    render_prefix_generation_with_sentence_indices_html,
    resolve_sentence_span,
    sentence_selector_label,
    strip_code_fences,
    summarize_record_schema,
)


st.set_page_config(page_title="Commitment Localization Explorer", layout="wide")
st.title("Sentence-level Commitment Localization Dashboard")
st.caption("Local Streamlit explorer for commitment answer-accuracy localization runs.")


if "commitment_result_context" not in st.session_state:
    st.session_state.commitment_result_context = None


st.sidebar.header("Local Results")
localization_root = st.sidebar.text_input("Localization root", value=DEFAULT_LOCALIZATION_ROOT)
scan_limit = int(
    st.sidebar.number_input(
        "Max examples to list",
        min_value=20,
        max_value=10000,
        value=DEFAULT_SCAN_LIMIT,
        step=20,
        help="Only list the first N example files under the selected model/run.",
    )
)
filename_filter = st.sidebar.text_input(
    "Filename filter",
    value="",
    help="Optional substring filter applied to localization filenames.",
)
status_filter = st.sidebar.selectbox(
    "Answer status filter",
    ["all", "correct", "incorrect"],
    index=0,
)

models = list_models(localization_root)
if not models:
    st.sidebar.error("No model directories found under the selected localization root.")
    st.stop()

selected_model = st.sidebar.selectbox("Model", models, index=0)
runs = list_run_tags(localization_root, selected_model)
if not runs:
    st.sidebar.error("No run tags found for the selected model.")
    st.stop()

selected_run = st.sidebar.selectbox("Run tag", runs, index=len(runs) - 1)

repo_paths = list_localization_paths(
    localization_root,
    selected_model,
    selected_run,
    limit=scan_limit,
    filename_filter=filename_filter,
    status_filter=status_filter,
)

st.sidebar.caption(
    f"Showing {len(repo_paths):,} example files from "
    f"`{selected_model}/{selected_run}/localization`."
)

if not repo_paths:
    st.warning("No localization files matched the current model/run/filter.")
    st.stop()

example_options = {path.name: path for path in repo_paths}
selected_example_name = st.sidebar.selectbox("Example", list(example_options.keys()), index=0)
selected_path = example_options[selected_example_name]

try:
    result = load_local_record(selected_path)
except Exception as exc:
    st.error("Could not load the selected localization file.")
    st.code(type(exc).__name__ + ": " + str(exc))
    st.stop()

raw_text = result.get("raw_text") or ""
history = result.get("history") or []
if not history:
    st.error("No history found in this example.")
    st.stop()

history_norm = normalize_history(history)
df_plot = flatten_history(history_norm, raw_text)
df_stats = build_stats(history_norm)
example_id = str(result.get("example_id") or "")
base_example_id = str(result.get("base_example_id") or "")
right_sentence_end_idx = result.get("right_sentence_end_idx")
low_accuracy_sentence_idx = compute_low_accuracy_sentence_idx(right_sentence_end_idx, df_stats)
sentence_spans, sentence_span_map = build_sentence_spans(raw_text)

current_result_context: Tuple[str, str] = (str(selected_path), example_id)
if st.session_state.commitment_result_context != current_result_context:
    st.session_state.commitment_result_context = current_result_context
    for key in (
        "selected_sentence_idx",
        "selected_probe_step",
        "correct_sel",
        "incorrect_sel",
        "sample_selection_context",
    ):
        if key in st.session_state:
            del st.session_state[key]


st.subheader("Result Summary")
summary_lines = [
    f"Model: {selected_model}",
    f"Run tag: {selected_run}",
    f"Example ID: {example_id}",
    f"Base example ID: {base_example_id}",
    f"Answer status file: {result.get('answer_status')}",
    f"Original trace was correct: {result.get('is_correct')}",
    f"Correct option: {result.get('correct_option')}",
    f"File: {selected_path.name}",
    f"Probes: {len(history_norm)}",
]
if low_accuracy_sentence_idx is not None:
    summary_lines.append(f"First low-accuracy sentence idx (<= 0.5): {low_accuracy_sentence_idx}")
if right_sentence_end_idx is not None:
    summary_lines.append(f"Stored right sentence end idx: {right_sentence_end_idx}")
st.markdown("\n".join(f"- {line}" for line in summary_lines))

with st.expander("Prompt", expanded=False):
    prompt = result.get("prompt") or ""
    if prompt:
        st.text(prompt)
    else:
        st.info("No prompt stored in this example.")

with st.expander("Raw reasoning text", expanded=False):
    if raw_text:
        st.text(raw_text)
    else:
        st.info("No raw reasoning text stored in this example.")


st.subheader("Correct Rate vs Sentence Index")
if len(df_stats) > 0:
    fig1, df_stats = plot_sentence_localization(
        df_stats,
        low_accuracy_sentence_idx=low_accuracy_sentence_idx,
        right_sentence_end_idx=right_sentence_end_idx,
    )
    st.pyplot(fig1)
    with st.expander("Show probe statistics"):
        st.dataframe(df_stats, use_container_width=True)
else:
    st.info("No stats available to plot.")


st.subheader("Prefix Selector")
st.caption(
    "Choose where the fixed prefix should end. The dashboard holds the text fixed up to "
    "that sentence, then shows continuations sampled from that point."
)

available_idxs = sorted(int(value) for value in df_stats["sentence_idx"].dropna().unique()) if len(df_stats) else []
if not available_idxs:
    st.info("No sentence indices available for selection.")
    st.stop()

if st.session_state.get("selected_sentence_idx") not in available_idxs and "selected_sentence_idx" in st.session_state:
    del st.session_state["selected_sentence_idx"]

selected_sentence_idx = st.selectbox(
    "Sentence index",
    available_idxs,
    key="selected_sentence_idx",
    format_func=lambda idx: sentence_selector_label(idx, sentence_span_map),
)

probe_rows = [
    (idx, probe)
    for idx, probe in enumerate(history_norm)
    if probe.get("sentence_idx") == selected_sentence_idx
    or probe.get("sentence_end_idx") == selected_sentence_idx + 1
]
if not probe_rows:
    st.info("No probe found for this sentence.")
    st.stop()

probe = probe_rows[0][1]
if len(probe_rows) > 1:
    step_options = [idx for idx, _ in probe_rows]
    if st.session_state.get("selected_probe_step") not in step_options and "selected_probe_step" in st.session_state:
        del st.session_state["selected_probe_step"]
    selected_step = st.selectbox(
        "Probe step",
        step_options,
        key="selected_probe_step",
        format_func=lambda idx: f"step {idx}",
    )
    probe = dict(history_norm[selected_step])

resolved_sentence_text, resolved_start, resolved_end = resolve_sentence_span(
    selected_sentence_idx,
    sentence_span_map,
    probe=probe,
)
resolved_sentence_idx = int(selected_sentence_idx)
sentence_text = (resolved_sentence_text or probe.get("sentence_text") or "").strip()

st.markdown(
    f"Sentence idx: {resolved_sentence_idx} | Correct rate: {probe.get('correct_rate')} | "
    f"Valid samples: {probe.get('num_valid')}"
)
if sentence_text:
    st.markdown(f"Sentence: `{sentence_text}`")

bucket_counts = (
    pd.Series(
        [generation_bucket(generation) for generation in (probe.get("generations") or [])],
        name="count",
    )
    .value_counts()
    .reindex(["correct", "incorrect", "invalid"], fill_value=0)
)
st.dataframe(bucket_counts.to_frame(), use_container_width=False)


st.subheader("Sample Selector")
st.caption(
    "Select one continuation sampled from the chosen prefix. Correct and incorrect "
    "continuations are listed separately so you can compare how the same fixed prefix "
    "can lead to different outcomes."
)

subset = df_plot[
    (df_plot["sentence_idx"] == selected_sentence_idx)
    | (df_plot["sentence_end_idx"] == selected_sentence_idx + 1)
]
if subset.empty:
    st.info("No generations available for this probe.")
    st.stop()

sample_selection_context = (str(selected_path), int(selected_sentence_idx))
if st.session_state.get("sample_selection_context") != sample_selection_context:
    st.session_state.sample_selection_context = sample_selection_context
    st.session_state.correct_sel = "None"
    st.session_state.incorrect_sel = "None"


def on_correct_change():
    st.session_state.incorrect_sel = "None"


def on_incorrect_change():
    st.session_state.correct_sel = "None"


correct = subset[subset["is_correct"] == True]
incorrect = subset[subset["is_correct"] == False]
invalid_count = int((subset["is_correct"].isna()).sum())

correct_opts: Dict[str, int] = {
    f"Correct generation {row.sample_id}": int(row.sample_id)
    for _, row in correct.iterrows()
}
incorrect_opts: Dict[str, int] = {
    f"Incorrect generation {row.sample_id}": int(row.sample_id)
    for _, row in incorrect.iterrows()
}

col1, col2 = st.columns(2)
with col1:
    st.markdown("Correct samples")
    st.selectbox(
        "Select correct sample",
        ["None"] + list(correct_opts.keys()),
        key="correct_sel",
        on_change=on_correct_change,
    )
with col2:
    st.markdown("Incorrect samples")
    st.selectbox(
        "Select incorrect sample",
        ["None"] + list(incorrect_opts.keys()),
        key="incorrect_sel",
        on_change=on_incorrect_change,
    )

if invalid_count:
    st.caption(f"Invalid generations for this sentence: {invalid_count}")

selected_sample_id: Optional[int] = None
if st.session_state.correct_sel != "None":
    selected_sample_id = correct_opts.get(st.session_state.correct_sel)
elif st.session_state.incorrect_sel != "None":
    selected_sample_id = incorrect_opts.get(st.session_state.incorrect_sel)


st.subheader("Generation Viewer")
st.caption(
    "This view shows the fixed prefix together with the selected continuation. Blue text "
    "comes before the final sentence in the fixed prefix, black text is the final sentence "
    "in the fixed prefix, and green text is the sampled continuation."
)

if selected_sample_id is None:
    st.info("Select a correct or incorrect sample above to view the generation.")
    st.stop()

selected_rows = subset[subset["sample_id"] == selected_sample_id]
if selected_rows.empty:
    st.info("Selected sample is not available for the current sentence.")
    st.stop()

row = selected_rows.iloc[0]
gen_text = strip_code_fences(row.get("gen_text") or "")

if resolved_end is not None:
    prefix_text = raw_text[:resolved_end]
else:
    prefix_text = sentence_text or ""

prefix_mode_key = "full"
display_selected_idx, _ = build_generation_sentence_labels(
    prefix_text,
    prefix_mode_key,
    resolved_sentence_idx,
)
generation_sentence_labels = {}

st.markdown(
    f"Correct: {row.get('is_correct')} | Predicted option: {row.get('predicted_option')} | "
    f"Final answer: {row.get('final_answer')} | Parse error: {row.get('parse_error')}"
)

st.markdown(
    render_prefix_generation_with_sentence_indices_html(
        prefix_text,
        gen_text,
        display_selected_idx,
        sentence_labels=generation_sentence_labels,
    ),
    unsafe_allow_html=True,
)

with st.expander("Full generation text", expanded=False):
    full_generation_text = row.get("full_generation_text") or ""
    if full_generation_text:
        st.text(full_generation_text)
    else:
        st.info("No full generation text stored.")

with st.expander("Sample metadata", expanded=False):
    sample_meta = {
        "is_correct": row.get("is_correct"),
        "predicted_option": row.get("predicted_option"),
        "final_answer": row.get("final_answer"),
        "parse_success": row.get("parse_success"),
        "parse_error": row.get("parse_error"),
        "matched_final_answer_block": row.get("matched_final_answer_block"),
    }
    st.json(sample_meta, expanded=False)

with st.expander("Schema overview", expanded=False):
    top_level_df, history_df, generation_df = summarize_record_schema(result)
    st.markdown("Top-level record")
    st.dataframe(top_level_df, use_container_width=True)
    st.markdown("History keys")
    st.dataframe(history_df, use_container_width=True)
    st.markdown("Generation keys")
    st.dataframe(generation_df, use_container_width=True)
