#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
slakshna_root="${workspace_root}/Slakshna"
runtime_root="${experiment_root}/.runtime"
venv="${M0_UV_ENVIRONMENT:-${slakshna_root}/Bhaskera/.venv-phase9}"
python_bin="${venv}/bin/python"
hf_bin="${venv}/bin/hf"

local_data_root="${M0_LOCAL_DATA_ROOT:-${workspace_root}/local_data}"
incoming_root="${M0_INCOMING_ROOT:-${local_data_root}/m0_incoming}"
model_root="${M0_MODEL_ROOT:-${runtime_root}/models/m0}"
benchmark_root="${M0_BENCHMARK_ROOT:-${runtime_root}/data/m0/benchmarks/raw}"
hf_cache="${M0_HF_CACHE:-${runtime_root}/cache/huggingface}"
manifest_root="${runtime_root}/manifests/m0"
log_root="${runtime_root}/logs/m0"

western_europe_url="https://drive.google.com/file/d/1qsnk7WblcUx_m8WljUzxcsqpqpw3c12A/view"
continent_folder_url="https://drive.google.com/drive/u/2/folders/1-sxaE2tVCqkB53dxQ_OpLJLI9Uq5U6yS"
model_7b="allenai/OLMo-2-1124-7B-Instruct"
model_1b="allenai/OLMo-2-0425-1B-Instruct"
benchmark_cultural="kellycyy/CulturalBench"
benchmark_goqa="Anthropic/llm_global_opinions"

download_culture=1
download_model=1
download_benchmarks=1
download_full_1b="${M0_DOWNLOAD_FULL_1B:-0}"
parallel="${M0_DOWNLOAD_PARALLEL:-1}"

usage() {
    cat <<'EOF'
Usage: bash monash_exps/scripts/m0/02_download_assets.sh [options]

Download the long-running M0 assets. Existing complete Hugging Face files and
gdown outputs are reused, so the script is safe to rerun after interruption.

Options:
  --skip-culture       Skip the Western Europe and continent Google Drive data
  --skip-model         Skip OLMo 2 model downloads
  --skip-benchmarks    Skip CulturalBench and GlobalOpinionQA
  --full-1b            Download full OLMo 2 1B weights (metadata only by default)
  --sequential         Run the three download groups sequentially
  -h, --help           Show this help

Environment overrides:
  M0_LOCAL_DATA_ROOT, M0_INCOMING_ROOT, M0_MODEL_ROOT, M0_BENCHMARK_ROOT
  M0_HF_MAX_WORKERS (default 4), M0_DOWNLOAD_PARALLEL (default 1)
  M0_GDRIVE_AUTHUSER (default 2 for the supplied /u/2 folder)
  HF_TOKEN (only if Hugging Face access requires it)
  GDOWN_COOKIE_PATH (default ~/.cache/gdown/cookies.txt)
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --skip-culture) download_culture=0 ;;
        --skip-model) download_model=0 ;;
        --skip-benchmarks) download_benchmarks=0 ;;
        --full-1b) download_full_1b=1 ;;
        --sequential) parallel=0 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

die() { echo "ERROR: $*" >&2; exit 1; }
section() { echo; echo "=== $* ==="; }

[[ -x "${python_bin}" ]] || die "Missing upgraded environment: ${python_bin}"
[[ -x "${hf_bin}" ]] || die "Missing Hugging Face CLI: ${hf_bin}"
"${python_bin}" -c 'import gdown, huggingface_hub' || \
    die "gdown/huggingface_hub missing; run 01_upgrade_stack.sh first"

mkdir -p "${incoming_root}" "${model_root}" "${benchmark_root}" \
    "${hf_cache}" "${manifest_root}" "${log_root}"
stamp="$(date +%Y%m%dT%H%M%S)"
log_path="${log_root}/download-assets-${stamp}.log"
exec > >(tee -a "${log_path}") 2>&1

available_kb="$(df -Pk "${workspace_root}" | awk 'NR==2 {print $4}')"
minimum_kb=$(( 60 * 1024 * 1024 ))
if (( available_kb < minimum_kb )) && [[ "${M0_ALLOW_LOW_DISK:-0}" != "1" ]]; then
    die "Less than 60 GiB free under ${workspace_root}; set storage roots or M0_ALLOW_LOW_DISK=1"
fi
echo "Free space: $(( available_kb / 1024 / 1024 )) GiB"
echo "Log       : ${log_path}"

hf_args=(--cache-dir "${hf_cache}" --max-workers "${M0_HF_MAX_WORKERS:-4}")
if [[ -n "${HF_TOKEN:-}" ]]; then hf_args+=(--token "${HF_TOKEN}"); fi

download_culture_group() {
    section "Download data-team CultureInstruct partitions"
    cookie_path="${GDOWN_COOKIE_PATH:-${HOME}/.cache/gdown/cookies.txt}"
    [[ -s "${cookie_path}" ]] || \
        die "Missing gdown cookies: ${cookie_path}"
    cookie_mode="$(stat -c '%a' "${cookie_path}")"
    [[ "${cookie_mode}" == "600" ]] || \
        echo "WARNING: cookie mode is ${cookie_mode}; 600 is recommended"

    western_dir="${incoming_root}/western_europe"
    continent_dir="${incoming_root}/continent_splits"
    mkdir -p "${western_dir}" "${continent_dir}"

    drive_helper="${experiment_root}/src/m0/download_authenticated_drive.py"
    [[ -f "${drive_helper}" ]] || die "Missing Drive helper: ${drive_helper}"
    western_id="$(sed -E 's#^.*/d/([^/]+)/.*#\1#' <<< "${western_europe_url}")"
    continent_id="$(sed -E 's#^.*/folders/([^/?]+).*$#\1#' <<< "${continent_folder_url}")"
    drive_args=(
        --cookie-file "${cookie_path}"
        --authuser "${M0_GDRIVE_AUTHUSER:-2}"
    )
    "${python_bin}" "${drive_helper}" file --id "${western_id}" \
        --output "${western_dir}" "${drive_args[@]}"
    "${python_bin}" "${drive_helper}" folder --id "${continent_id}" \
        --output "${continent_dir}" "${drive_args[@]}"
}

download_model_group() {
    section "Download formal OLMo 2 7B Instruct checkpoint"
    "${hf_bin}" download "${model_7b}" \
        --local-dir "${model_root}/OLMo-2-1124-7B-Instruct" \
        "${hf_args[@]}"

    if [[ "${download_full_1b}" == "1" ]]; then
        section "Download full OLMo 2 1B Instruct smoke checkpoint"
        "${hf_bin}" download "${model_1b}" \
            --local-dir "${model_root}/OLMo-2-0425-1B-Instruct" \
            "${hf_args[@]}"
    else
        section "Download OLMo 2 1B tokenizer/config only"
        "${hf_bin}" download "${model_1b}" \
            --include '*.json' '*.model' '*.txt' \
            --local-dir "${model_root}/OLMo-2-0425-1B-Instruct-metadata" \
            "${hf_args[@]}"
    fi
}

download_benchmark_group() {
    section "Download CulturalBench"
    "${hf_bin}" download "${benchmark_cultural}" --repo-type dataset \
        --local-dir "${benchmark_root}/CulturalBench" \
        "${hf_args[@]}"
    section "Download GlobalOpinionQA"
    "${hf_bin}" download "${benchmark_goqa}" --repo-type dataset \
        --local-dir "${benchmark_root}/GlobalOpinionQA" \
        "${hf_args[@]}"
}

declare -a pids=()
declare -a labels=()
launch_group() {
    local label="$1"
    shift
    if [[ "${parallel}" == "1" ]]; then
        "$@" &
        pids+=("$!")
        labels+=("${label}")
    else
        "$@"
    fi
}

if (( download_culture )); then launch_group culture download_culture_group; fi
if (( download_model )); then launch_group model download_model_group; fi
if (( download_benchmarks )); then launch_group benchmarks download_benchmark_group; fi

failures=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "Download group passed: ${labels[$index]}"
    else
        echo "Download group failed: ${labels[$index]}" >&2
        failures=$(( failures + 1 ))
    fi
done
(( failures == 0 )) || die "${failures} download group(s) failed; rerun to resume"

section "Write acquired-file inventory"
inventory_path="${manifest_root}/downloaded-assets.json"
M0_INVENTORY_PATH="${inventory_path}" \
M0_INCOMING_ROOT="${incoming_root}" \
M0_MODEL_ROOT="${model_root}" \
M0_BENCHMARK_ROOT="${benchmark_root}" \
M0_DOWNLOAD_LOG="${log_path}" \
"${python_bin}" - <<'PY'
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

roots = {
    "culture_incoming": Path(os.environ["M0_INCOMING_ROOT"]),
    "models": Path(os.environ["M0_MODEL_ROOT"]),
    "benchmarks": Path(os.environ["M0_BENCHMARK_ROOT"]),
}
groups = {}
for name, root in roots.items():
    files = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if not path.is_file() or ".cache" in path.parts:
                continue
            files.append({
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    groups[name] = {"root": str(root.resolve()), "files": files}

payload = {
    "schema_version": 1,
    "created_at": datetime.now(timezone.utc).isoformat(),
    "groups": groups,
    "log": os.environ["M0_DOWNLOAD_LOG"],
}
output = Path(os.environ["M0_INVENTORY_PATH"])
temporary = output.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(output)
print(f"Inventory: {output}")
for name, group in groups.items():
    print(f"{name}: {len(group['files'])} files, {sum(x['bytes'] for x in group['files'])} bytes")
PY

echo
echo "M0 ASSET DOWNLOAD PASSED"
echo "Incoming data : ${incoming_root}"
echo "Models        : ${model_root}"
echo "Benchmarks    : ${benchmark_root}"
echo "Inventory     : ${inventory_path}"
echo "Log           : ${log_path}"
