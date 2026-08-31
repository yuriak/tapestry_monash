#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
slakshna_root="${workspace}/Slakshna"
bhaskera_root="${slakshna_root}/Bhaskera"
python_bin="${M0_FL_PYTHON:-${workspace}/monash_exps/.runtime/venvs/primary/bin/python}"
config_template="${workspace}/monash_exps/configs/phase9/tokenize_south_asia.yaml"
config="${workspace}/monash_exps/.runtime/configs/phase9/tokenize_south_asia.yaml"
source_data="${bhaskera_root}/dataset/South_Asia.jsonl"
cache_root="${JOINT_TOKENIZED_ROOT:-${workspace}/monash_exps/.runtime/data/phase9/tokenized/south_asia_seq2048_packed_20260830}"
log_root="${workspace}/monash_exps/.runtime/logs/cross_country_fl"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ -x "${python_bin}" ]] || die "Python environment not found: ${python_bin}"
[[ -f "${config_template}" ]] || die "Tokenization config template not found: ${config_template}"
[[ -f "${source_data}" ]] || die "South Asia training data not found: ${source_data}"

rows="$(wc -l < "${source_data}")"
[[ "${rows}" == "15331" ]] || die "Expected 15,331 South Asia rows, found ${rows}"
source_sha="$(sha256sum "${source_data}" | awk '{print $1}')"
[[ "${source_sha}" == "5d4b90d35c8692a9e5db5b2aafe1cbf07e484f3bbd296320cf2fa9069a231429" ]] || \
    die "South Asia source hash does not match the expected dataset"

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
export XDG_CACHE_HOME="${workspace}/monash_exps/.runtime/cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
unset RAY_ADDRESS RAY_HEAD_SERVICE_HOST RAY_HEAD_SERVICE_PORT

stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/tokenize-south-asia-${stamp}.log"

echo "South Asia offline tokenization"
echo "  Slakshna : $(git -C "${slakshna_root}" rev-parse --short HEAD)"
echo "  Bhaskera : $(git -C "${bhaskera_root}" rev-parse --short HEAD)"
echo "  source   : ${source_data}"
echo "  rows     : ${rows}"
echo "  config   : ${config}"
echo "  cache    : ${cache_root}"
echo "  log      : ${log_path}"

tokenized_path="${cache_root}/local_train_079a0daaff30bc4e"
if [[ -f "${tokenized_path}/metadata.json" ]]; then
    echo "Existing South Asia cache found; validating without retokenizing."
else
    cd "${workspace}"
    "${python_bin}" -m bhaskera.launcher.tokenize \
        --config "${config}" \
        --split train \
        2>&1 | tee "${log_path}"
fi

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
echo "SOUTH ASIA TOKENIZATION PASSED"
echo "Use this in the India runtime view:"
echo "  tokenized_path: \"${tokenized_path}\""
