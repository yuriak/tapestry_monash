#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
slakshna_root="${workspace}/Slakshna"
python_bin="${workspace}/monash_exps/.runtime/venvs/primary/bin/python"
uv_bin="${workspace}/monash_exps/.runtime/tools/uv/bin/uv"
rust_binary="${workspace}/monash_exps/.runtime/cargo-target/slakshna/release/iiitd"
expected_slakshna="a73287fd56b1d1e935482c2f76771a33d2f05b0c"
expected_bhaskera="75a2698b60313aa6b26124312c3329cb72083b9b"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

[[ "$(git -C "${slakshna_root}" rev-parse HEAD)" == "${expected_slakshna}" ]] || \
    die "Slakshna revision does not match the second-joint-training revision"
[[ "$(git -C "${slakshna_root}/Bhaskera" rev-parse HEAD)" == "${expected_bhaskera}" ]] || \
    die "Bhaskera revision does not match the Slakshna-pinned revision"
[[ -x "${python_bin}" ]] || die "missing Python environment: ${python_bin}"
[[ -x "${uv_bin}" ]] || die "missing project-local uv: ${uv_bin}"
[[ -x "${rust_binary}" ]] || die "missing Rust release binary: ${rust_binary}"

export UV_CACHE_DIR="${workspace}/monash_exps/.runtime/cache/uv"
"${uv_bin}" pip check --python "${python_bin}"

"${python_bin}" - \
    "${slakshna_root}/node_template.yaml" \
    "${slakshna_root}/node_monash.toml" \
    "${workspace}/monash_exps/.runtime/data/m0/prepared/australia_nz/train.jsonl" \
    "${JOINT_REQUIRE_PEER:-0}" <<'PY'
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path

import pyarrow.dataset as ds
import torch
import yaml

template_path, node_path, source_path = map(Path, sys.argv[1:4])
require_peer = sys.argv[4] == "1"
template = yaml.safe_load(template_path.read_text())
node = tomllib.loads(node_path.read_text())

assert template["model"]["name"] == "allenai/OLMo-2-1124-7B"
assert template["model"]["dtype"] == "bfloat16"
assert template["data"]["pack_sequences"] is True
assert template["data"]["seq_len"] == 2048
assert template["lora"]["enabled"] is True
assert template["lora"]["rank"] == 16
assert template["lora"]["target_modules"] == ["q_proj", "v_proj"]
assert template["training"]["batch_size"] == 4
assert template["training"]["gradient_accumulation_steps"] == 8
assert template["training"]["max_steps"] == 50
assert template["training"]["distributed"]["strategy"] == "fsdp"
assert template["checkpoint"]["local_interval"] == 1
assert node["training"]["expected_peers"] == 2
assert node["node"]["num_gpus"] == 2
assert not node["discovery"]["mdns"]

source = Path(template["data"]["train_path"])
assert source.resolve() == source_path.resolve()
assert sum(1 for _ in source.open()) == 9337
digest = hashlib.sha256(source.read_bytes()).hexdigest()
assert digest == "9f6535cc196db6f514cce1f9e04307148e79f74a575e067d1567bc1d7d5b9237"

cache = Path(template["data"]["tokenized_path"])
metadata_path = cache / "metadata.json"
assert metadata_path.is_file(), f"missing packed token cache metadata: {metadata_path}"
metadata = json.loads(metadata_path.read_text())
assert metadata["model_name"] == "allenai/OLMo-2-1124-7B-Instruct"
assert metadata["seq_len"] == 2048
assert metadata["pack_sequences"] is True
assert metadata["train_on_inputs"] is False
parquet = sorted(cache.glob("*.parquet"))
assert parquet, f"no Parquet files found under {cache}"
row_count = ds.dataset([str(path) for path in parquet], format="parquet").count_rows()
assert row_count == metadata["num_rows"] and row_count > 0

peers = node["network"].get("peers", [])
allowed = node["network"].get("allowed_peers", [])
if require_peer and not peers:
    raise AssertionError("meeting peer information has not been entered")
if peers:
    assert len(peers) == 1 and len(allowed) == 1
    match = re.fullmatch(r"([0-9a-f]{64})@([^:]+):(\d+)", peers[0])
    assert match, f"invalid peer entry: {peers[0]}"
    assert allowed[0] == match.group(1)
    assert 1 <= int(match.group(3)) <= 65535
    peer_status = peers[0]
else:
    peer_status = "pending meeting handoff"

state_dir = Path(node["node"]["data_dir"])
assert not (state_dir / "network_deltas").exists(), "new runtime contains cached peer deltas"

print(f"torch={torch.__version__} cuda_build={torch.version.cuda}")
print(f"source_rows=9337 source_sha256={digest}")
print(f"packed_rows={row_count} cache={cache}")
print(f"peer={peer_status}")
print(f"fresh_state={state_dir}")
PY

manifest="${workspace}/monash_exps/.runtime/manifests/primary/slakshna-build.txt"
[[ -f "${manifest}" ]] || die "missing Rust build manifest"
grep -q "revision=${expected_slakshna}" "${manifest}" || die "Rust binary revision is stale"

runtime_ld_library_path=""
IFS=':' read -r -a ld_entries <<< "${LD_LIBRARY_PATH:-}"
for entry in "${ld_entries[@]}"; do
    if [[ -z "${entry}" || "${entry}" == "/usr/local/gcc/10.2.0/lib64" ]]; then
        continue
    fi
    [[ -z "${runtime_ld_library_path}" ]] || runtime_ld_library_path+=":"
    runtime_ld_library_path+="${entry}"
done
if LD_LIBRARY_PATH="${runtime_ld_library_path}" ldd "${rust_binary}" | grep -q 'not found'; then
    die "Rust binary has unresolved shared libraries"
fi

[[ -d "${slakshna_root}/hf_cache/hub/models--allenai--OLMo-2-1124-7B/snapshots" ]] || \
    die "local OLMo 2 7B base snapshot is missing"
[[ ! -e "${slakshna_root}/ml_models" ]] || \
    [[ -z "$(find "${slakshna_root}/ml_models" -mindepth 1 -print -quit)" ]] || \
    die "Slakshna/ml_models is not empty before the new run"
[[ ! -e "${slakshna_root}/ml_states" ]] || \
    [[ -z "$(find "${slakshna_root}/ml_states" -mindepth 1 -print -quit)" ]] || \
    die "Slakshna/ml_states is not empty before the new run"

echo
echo "SECOND JOINT TRAINING PREFLIGHT PASSED"
echo "Slakshna : ${expected_slakshna}"
echo "Bhaskera : ${expected_bhaskera}"
echo "Launcher  : ${slakshna_root}/run_monash.sh"
echo "Node TOML : ${slakshna_root}/node_monash.toml"
echo "Training  : ${slakshna_root}/node_template.yaml"
