# GlobalOpinionQA Australia/New Zealand and India evaluation

This directory is a self-contained GlobalOpinionQA evaluation package for the
M0 collaboration. It does not import Slakshna, Bhaskera, `monash_exps`, or any
other code from the repository that contains it. The package can be copied to a
different project and run there as long as its internal directory structure is
preserved.

The included dataset contains only questions with at least one valid human
response distribution for Australia, New Zealand, or India. Human distributions
for every other country have been removed. The model prompt is not conditioned
on a country identity: the same model distribution is compared separately with
the available human distributions.

## Contents

| File | Purpose |
|---|---|
| `data/goqa_au_nz_india.jsonl` | Filtered 1,106-question evaluation dataset |
| `data/manifest.json` | Source hash, filtering rule, target counts, and output hash |
| `run_inference.py` | Five-prompt vLLM inference with optional unmerged LoRA |
| `run_trajectory.py` | Persistent-engine inference over a base model and multiple LoRAs |
| `score_predictions.py` | Dependency-free Jensen–Shannon scoring and reports |
| `validate_package.py` | Dataset, manifest, and prediction integrity checks |
| `run_evaluation.sh` | One-command validation, inference, and scoring workflow |
| `run_trajectory.sh` | One-command multi-adapter inference and scoring workflow |
| `stage_model.sh` | Optional staging of model files onto node-local storage |
| `build_dataset.py` | Reproducible subset builder; not needed for normal evaluation |

See `DATASET_NOTICE.md` for the upstream dataset license and citation.

## Environment

Python 3.10 or newer and a CUDA-compatible vLLM installation are required for
inference. The scoring and validation scripts use only the Python standard
library. The code has been tested with Python 3.11, vLLM 0.13.0, PyTorch 2.9.0,
and Transformers 4.57.6.

Install into an existing GPU environment:

```bash
python -m pip install -r requirements.txt
```

The exact PyTorch and vLLM build should be selected for the CUDA driver on the
target cluster. Installing `requirements.txt` is optional when a working vLLM
environment already exists.

Before allocating a GPU, validate the package on a CPU or login node:

```bash
python validate_package.py
```

## One-command evaluation

Evaluate a base model:

```bash
bash run_evaluation.sh /path/to/base-model /path/to/output
```

Evaluate an unmerged Hugging Face-format LoRA adapter:

```bash
bash run_evaluation.sh /path/to/base-model /path/to/output /path/to/lora-adapter
```

To select a particular Python environment or tune vLLM without editing code:

```bash
GOQA_PYTHON=/path/to/python \
GOQA_TENSOR_PARALLEL_SIZE=1 \
GOQA_REQUEST_BATCH_SIZE=8192 \
GOQA_GPU_MEMORY_UTILIZATION=0.90 \
bash run_evaluation.sh /path/to/base-model /path/to/output /path/to/lora-adapter
```

The workflow is resumable. Re-running the same command skips question IDs that
are already present in `predictions.jsonl`. It refuses to resume if the model,
adapter, dataset, prompt, or inference configuration recorded in the prediction
manifest has changed.

## Multi-adapter trajectory evaluation

When several LoRA checkpoints share one base model, use the trajectory entry
point. It loads the base model once, switches unmerged adapters inside the same
vLLM engine, and scores every completed prediction file after inference:

```bash
GOQA_PYTHON=/path/to/python \
bash run_trajectory.sh /path/to/base-model /path/to/output \
  base \
  round_01=/path/to/round-1-adapter \
  round_02=/path/to/round-2-adapter
```

On clusters with slow shared storage, the model can be copied once to local
NVMe before vLLM starts:

```bash
GOQA_PYTHON=/path/to/python \
GOQA_STAGE_MODEL_DIR=/tmp/goqa-base-model \
bash run_trajectory.sh /path/to/base-model /path/to/output \
  base round_01=/path/to/round-1-adapter
```

The default scheduler limits are tuned for the checked-in dataset: its 5,530
prompt variants have a maximum encoded length of 448 tokens. The evaluator uses
`max_model_len=512`, `max_num_batched_tokens=32768`, and `max_num_seqs=512`.
Override these with `GOQA_MAX_MODEL_LEN`, `GOQA_MAX_NUM_BATCHED_TOKENS`, and
`GOQA_MAX_NUM_SEQS` when evaluating a different dataset.

## Separate inference and scoring

Inference can be run independently:

```bash
python run_inference.py \
  --model /path/to/base-model \
  --adapter /path/to/lora-adapter \
  --run-name m0-model \
  --output /path/to/output/predictions.jsonl \
  --request-batch-size 8192 \
  --tensor-parallel-size 1
```

Omit `--adapter` to evaluate the unchanged base model. After inference, scoring
does not require a GPU or vLLM:

```bash
python score_predictions.py \
  --predictions /path/to/output/predictions.jsonl \
  --output-dir /path/to/output/scores
```

The score directory contains machine-readable JSON, regional and disaggregated
CSV tables, per-question/group distances, and a short Markdown report.

## Evaluation protocol

Each question is presented under five deterministic option orders: the source
order and four SHA-256-seeded permutations. For every prompt, inference is
restricted to the valid one-token option labels. Their first-token log
probabilities are normalized to form a distribution, mapped back to source
option order, and averaged across the five prompts.

vLLM is configured to return processed log probabilities after applying the
valid-label mask. This is mathematically equivalent to normalizing the raw
label logits, while ensuring that every valid label is returned in the first
request. It avoids additional forced-label generation requests and does not
change the evaluation metric.

The averaged model distribution is compared with each available target human
distribution using base-2 Jensen–Shannon distance:

```text
M = (P + Q) / 2
JS_distance(P, Q) = sqrt((KL2(P || M) + KL2(Q || M)) / 2)
```

The distance is bounded by zero and one, and lower is better. A missing country
distribution is not imputed and contributes no pair. A distribution whose
source values sum to zero is excluded when the subset is built.

The primary regional metrics are:

| Metric | Aggregation |
|---|---|
| Australia/New Zealand macro | Mean of the Australia question mean and New Zealand question mean |
| India sample-frame macro | Mean of Current national, Non-national, and Old national question means |
| Two-region macro | Equal mean of the two regional metrics above |

The India value is deliberately called a sample-frame macro rather than a
country macro. The three India labels describe different survey samples. Their
individual results are always reported alongside the regional value. Pair-
weighted regional means are also emitted as secondary diagnostics.

## Dataset coverage

| Target group | Valid country-question pairs |
|---|---:|
| Australia | 626 |
| New Zealand | 273 |
| India — Current national sample | 470 |
| India — Non-national sample | 340 |
| India — Old national sample | 122 |

The subset contains 1,106 unique questions. Australia/New Zealand covers 626
unique questions, India covers 766, and 286 questions occur in both regional
views. Original GOQA row indices and question IDs are retained so that prompt
permutations exactly match the existing M0 baseline evaluation.

## Rebuilding the subset

Normal users should evaluate the checked-in dataset rather than rebuilding it.
Given the original `global_opinions.csv`, the checked-in artifact can be
reproduced with:

```bash
python build_dataset.py \
  --source-csv /path/to/global_opinions.csv \
  --output data/goqa_au_nz_india.jsonl \
  --manifest data/manifest.json
python validate_package.py
```
