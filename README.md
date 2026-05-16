# Commitment Pipeline

This folder contains the commitment / answer-accuracy localization pipeline for MathQA.

The workflow is:

1. `commitment mining`
2. `build sentence dataset`
3. `commitment localization`

These commands assume the `deception` conda environment and the current folder layout under `/playpen-ssd/smerrill/commitment`.

## Paths

- Source code: `commitment/src`
- Utilities: `commitment/utils`
- Shell scripts: `commitment/shell_scripts`
- Notebooks: `commitment/notebooks`
- Results root: `commitment/results`

## 1. Commitment Mining

What it does:

- Loads MathQA examples.
- Samples model completions for each question.
- Keeps sampling until it finds one `correct` reasoning trace and one `incorrect` reasoning trace for the same question.
- Writes those mined traces to `commitment_samples.jsonl`.

Main script:

- `commitment/src/commitment_miner.py`

Recommended launcher:

- `commitment/shell_scripts/run_commitment_miner_single_gpu.sh`

Example: run on GPU 7

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_commitment_miner_single_gpu.sh \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu 7
```

Example: larger run

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_commitment_miner_single_gpu.sh \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu 7 \
  --num_questions 200 \
  --max_samples_per_question 60
```

Outputs:

- `commitment/results/mining/<MODEL_TAG>/<RUN_TAG>/commitment_samples.jsonl`
- `commitment/results/mining/<MODEL_TAG>/<RUN_TAG>/run_config.json`
- `commitment/results/mining/<MODEL_TAG>/<RUN_TAG>/run_summary.json`

Notes:

- By default this uses `math_qa` and `test`.
- `RUN_TAG` is the timestamp-named run directory created under the model folder.

## 2. Build Sentence Dataset

What it does:

- Reads the mined commitment traces.
- Extracts the reasoning text from each mined example.
- Splits the reasoning into sentence spans.
- Writes sentence-level JSONL files used by localization.

Main script:

- `commitment/src/build_sentence_dataset.py`

Example:

```bash
RUN_TAG=2026-05-16_18-49-30
/playpen-ssd/smerrill/conda_envs/deception/bin/python /playpen-ssd/smerrill/commitment/src/build_sentence_dataset.py \
  --input_root /playpen-ssd/smerrill/commitment/results/mining/DeepSeek-R1-Distill-Qwen-7B/$RUN_TAG \
  --out_dir /playpen-ssd/smerrill/commitment/results/sentence_datasets/DeepSeek-R1-Distill-Qwen-7B/$RUN_TAG
```

Outputs:

- `commitment/results/sentence_datasets/<MODEL_TAG>/<RUN_TAG>/examples.jsonl`
- `commitment/results/sentence_datasets/<MODEL_TAG>/<RUN_TAG>/sentences.jsonl`

Notes:

- This step uses the improved sentence splitter in `commitment/utils/sentence_pipeline.py`.
- You can filter to only correct or only incorrect traces with `--label_filter correct_only` or `--label_filter incorrect_only`.

## 3. Commitment Localization

What it does:

- Takes the mined sentence dataset.
- Fixes the model prefix at different sentence boundaries.
- Resamples completions from each prefix.
- Measures the `correct answer rate` from that prefix onward.
- Produces per-example localization JSON files showing where accuracy changes.

Main script:

- `commitment/src/sentence_localization.py`

Recommended launcher:

- `commitment/shell_scripts/run_sentence_localization_multi_gpu.sh`

Example: run localization on GPUs `4 5 6 7`

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_sentence_localization_multi_gpu.sh \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu_ids "4 5 6 7" \
  --run_tag 2026-05-16_18-49-30
```

Example: point directly at a miner output directory

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_sentence_localization_multi_gpu.sh \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --gpu_ids "4 5 6 7" \
  --miner_output_dir /playpen-ssd/smerrill/commitment/results/mining/DeepSeek-R1-Distill-Qwen-7B/2026-05-16_18-49-30
```

Outputs:

- `commitment/results/localization/<MODEL_TAG>/<RUN_TAG>/localization/*.json`
- `commitment/results/localization/<MODEL_TAG>/<RUN_TAG>/run_gpu_<GPU>.log`
- optionally sharded JSONL outputs if `--write_jsonl` is used

Notes:

- The localization launcher shards examples across the GPUs you pass in `--gpu_ids`.
- If the sentence dataset does not already exist, the launcher will build it automatically.
- The default localization mode is `--method adaptive --mode prefix`.

## End-to-End Shortcut

If you want to run mining first and then localization in one go:

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_commitment_pipeline.sh \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --miner_gpu 7 \
  --localization_gpus "4 5 6 7"
```

This runs:

1. `run_commitment_miner_single_gpu.sh`
2. `run_sentence_localization_multi_gpu.sh`

## Streamlit Dashboard

What it is:

- A local browser for the commitment localization results.
- Lets you choose a model, run tag, and example from `commitment/results/localization`.
- Plots `correct rate` by sentence index.
- Lets you inspect fixed-prefix probes and compare `correct` vs `incorrect` sampled continuations.

Main app:

- `commitment/src/app.py`

Recommended launcher:

- `commitment/shell_scripts/run_dashboard.sh`

Run it:

```bash
bash /playpen-ssd/smerrill/commitment/shell_scripts/run_dashboard.sh
```

Then open:

- `http://localhost:8765`

If you want a different port:

```bash
PORT=8766 bash /playpen-ssd/smerrill/commitment/shell_scripts/run_dashboard.sh
```

Direct Streamlit command:

```bash
/playpen-ssd/smerrill/conda_envs/deception/bin/streamlit run /playpen-ssd/smerrill/commitment/src/app.py --server.headless true --server.address 0.0.0.0 --server.port 8765
```
