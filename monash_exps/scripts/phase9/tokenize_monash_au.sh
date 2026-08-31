#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
slakshna_root="${workspace}/Slakshna"
bhaskera_root="${slakshna_root}/Bhaskera"
python_bin="${M0_FL_PYTHON:-${workspace}/monash_exps/.runtime/venvs/primary/bin/python}"
config_template="${workspace}/monash_exps/configs/phase9/tokenize_monash_au.yaml"
config="${workspace}/monash_exps/.runtime/configs/phase9/tokenize_monash_au.yaml"
source_data="${workspace}/monash_exps/.runtime/data/m0/prepared/australia_nz/train.jsonl"
cache_root="${JOINT_TOKENIZED_ROOT:-/fs04/scratch2/da33/minghanw/tapestry_runtime/cross_country_fl/tokenized/monash_au_seq2048_packed_20260830}"
log_root="${workspace}/monash_exps/.runtime/logs/cross_country_fl"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -x "${python_bin}" ]] || die "Python environment not found: ${python_bin}"
[[ -f "${config_template}" ]] || die "Tokenization config template not found: ${config_template}"
[[ -f "${source_data}" ]] || die "AU training data not found: ${source_data}"

rows="$(wc -l < "${source_data}")"
[[ "${rows}" == "9337" ]] || die "Expected 9,337 AU/NZ rows, found ${rows}"
source_sha="$(sha256sum "${source_data}" | awk '{print $1}')"
[[ "${source_sha}" == "9f6535cc196db6f514cce1f9e04307148e79f74a575e067d1567bc1d7d5b9237" ]] || \
    die "AU/NZ source hash does not match the prepared V1 view"

mkdir -p "${cache_root}" "${log_root}" "$(dirname "${config}")"
"${python_bin}" - "${config_template}" "${config}" "${source_data}" "${cache_root}" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination, train_path, cache_dir = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text())
config["data"]["train_path"] = str(train_path.resolve())
config["data"]["cache_dir"] = str(cache_dir.resolve())
destination.write_text(yaml.safe_dump(config, sort_keys=False))
PY
export HF_HOME="${HF_HOME:-/fs04/scratch2/da33/minghanw/tapestry_runtime/cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export TOKENIZERS_PARALLELISM=false

stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/tokenize-monash-au-${stamp}.log"

echo "Monash AU offline tokenization"
echo "  Slakshna : $(git -C "${slakshna_root}" rev-parse --short HEAD)"
echo "  Bhaskera : $(git -C "${bhaskera_root}" rev-parse --short HEAD)"
echo "  source   : ${source_data}"
echo "  rows     : ${rows}"
echo "  config   : ${config}"
echo "  cache    : ${cache_root}"
echo "  log      : ${log_path}"

cd "${workspace}"
"${python_bin}" -m bhaskera.launcher.tokenize \
    --config "${config}" \
    --split train \
    2>&1 | tee "${log_path}"

tokenized_path="$("${python_bin}" - "${cache_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = []
for metadata_path in root.glob("*/metadata.json"):
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("model_name") == "allenai/OLMo-2-1124-7B-Instruct"
        and metadata.get("seq_len") == 2048
        and metadata.get("dataset_name") == "local_train"
        and metadata.get("pack_sequences") is True
        and metadata.get("train_on_inputs") is False
    ):
        candidates.append(metadata_path.parent)
if len(candidates) != 1:
    raise SystemExit(
        f"expected exactly one matching packed AU cache below {root}, found {candidates}"
    )
print(candidates[0].resolve())
PY
)"

"${python_bin}" - "${tokenized_path}" "${rows}" <<'PY'
import json
import sys
from pathlib import Path

import pyarrow.dataset as ds

cache = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
metadata = json.loads((cache / "metadata.json").read_text())
parquet_files = sorted(cache.glob("*.parquet"))
if not parquet_files:
    raise SystemExit(f"no Parquet shards found under {cache}")
table = ds.dataset([str(path) for path in parquet_files], format="parquet")
actual_rows = table.count_rows()
if actual_rows != metadata.get("num_rows"):
    raise SystemExit(
        f"metadata/parquet row mismatch: metadata={metadata.get('num_rows')} parquet={actual_rows}"
    )
if not 0 < actual_rows <= expected_rows:
    raise SystemExit(
        f"invalid packed row count: source={expected_rows}, tokenized={actual_rows}"
    )
if metadata.get("pack_sequences") is not True:
    raise SystemExit("cache metadata does not declare pack_sequences=true")
if metadata.get("train_on_inputs") is not False:
    raise SystemExit("cache metadata does not declare assistant-only labels")
required = {"input_ids", "attention_mask", "labels"}
missing = required.difference(table.schema.names)
if missing:
    raise SystemExit(f"tokenized schema is missing: {sorted(missing)}")
sample = table.head(min(64, actual_rows), columns=["input_ids", "attention_mask", "labels"])
for row_index, (input_ids, attention_mask, labels) in enumerate(
    zip(sample["input_ids"].to_pylist(), sample["attention_mask"].to_pylist(), sample["labels"].to_pylist())
):
    if not (len(input_ids) == len(attention_mask) == len(labels) == 2048):
        raise SystemExit(f"row {row_index} is not a fixed 2048-token sequence")
    if not any(label != -100 for label in labels):
        raise SystemExit(f"row {row_index} has no supervised assistant tokens")
print(f"Validated {actual_rows} packed rows from {expected_rows} source records")
print(f"Schema: {table.schema.names}")
print(f"Metadata: {metadata}")
PY

echo
echo "AU TOKENIZATION PASSED"
echo "Use this in the training configuration:"
echo "  tokenized_path: \"${tokenized_path}\""
