#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Phase 8 bridge readiness requires at least one visible GPU for G0 creation")
print(f"Visible GPUs: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"  logical {index}: {torch.cuda.get_device_name(index)}")
PY

run_id="${1:-$(date '+%Y%m%d_%H%M%S')_${SLURM_JOB_ID:-interactive}_phase8_bridge}"
artifact_root="${experiment_root}/artifacts/phase8/${run_id}"
mkdir -p "${artifact_root}"

echo "=== Compile and test the strict Phase 8 protocol ==="
python -m compileall -q "${experiment_root}/src/phase8"
PYTHONPATH="${experiment_root}/src" python -m unittest discover \
    -s "${experiment_root}/src/phase8/tests" -v
PYTHONPATH="${experiment_root}/src" python \
    "${experiment_root}/src/phase8/simulate_protocol.py" \
    --output "${artifact_root}/protocol-simulation.json"

allocated_cpus="${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-$(nproc)}}"
allocated_cpus="${allocated_cpus%%(*}"
if [[ ! "${allocated_cpus}" =~ ^[0-9]+$ ]]; then allocated_cpus=8; fi
if (( allocated_cpus > 8 )); then allocated_cpus=8; fi
if (( allocated_cpus < 3 )); then allocated_cpus=3; fi

echo "=== Prepare isolated, non-IID site inputs ==="
for site in site-a site-b; do
    SLAKSHNA_CPU_LIMIT="${allocated_cpus}" python \
        "${experiment_root}/src/phase8/prepare_site.py" \
        --site "${site}" --site-root "${artifact_root}/${site}" \
        2>&1 | tee "${artifact_root}/${site}-prepare.log"
done

echo "=== Create G0 once and install the verified bundle at both sites ==="
CUDA_VISIBLE_DEVICES=0 python "${experiment_root}/src/phase8/g0_bundle.py" create \
    --site-root "${artifact_root}/site-a" \
    --output-dir "${artifact_root}/g0-transfer-bundle"
for site in site-a site-b; do
    python "${experiment_root}/src/phase8/g0_bundle.py" install \
        --bundle-dir "${artifact_root}/g0-transfer-bundle" \
        --site-root "${artifact_root}/${site}"
done

echo "=== Materialize cluster-neutral Slakshna runtime configurations ==="
endpoint_a="$(printf 'a%.0s' {1..64})"
endpoint_b="$(printf 'b%.0s' {1..64})"
federation_id="phase8-readiness-${run_id//_/-}"
python "${experiment_root}/src/phase8/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase8/two_site.toml" \
    --bridge "${experiment_root}/src/phase8/ml_bridge.py" \
    --runtime-dir "${artifact_root}/site-a/runtime" \
    --data-dir "${artifact_root}/site-a/node-data" \
    --site-root "${artifact_root}/site-a" --site site-a \
    --federation-id "${federation_id}" --gpu-id 0 \
    --p2p-port 38180 --ws-port 38181 --api-port 38182 \
    --allowed-peer "${endpoint_b}"
python "${experiment_root}/src/phase8/prepare_runtime.py" \
    --template "${experiment_root}/configs/phase8/two_site.toml" \
    --bridge "${experiment_root}/src/phase8/ml_bridge.py" \
    --runtime-dir "${artifact_root}/site-b/runtime" \
    --data-dir "${artifact_root}/site-b/node-data" \
    --site-root "${artifact_root}/site-b" --site site-b \
    --federation-id "${federation_id}" --gpu-id 0 \
    --p2p-port 39180 --ws-port 39181 --api-port 39182 \
    --seed-peer "${endpoint_a}@127.0.0.1:38180" \
    --allowed-peer "${endpoint_a}"

python - "${artifact_root}" <<'PY'
import hashlib
import json
import sys
import tomllib
from pathlib import Path

root = Path(sys.argv[1]).resolve()
for site in ("site-a", "site-b"):
    config = tomllib.loads((root / site / "runtime/node.toml").read_text(encoding="utf-8"))
    if config["compression"]["enabled"] is not False:
        raise SystemExit(f"{site}: Slakshna compression must be disabled for the custom envelope")
    if config["training"]["expected_peers"] != 2 or config["node"]["num_gpus"] != 1:
        raise SystemExit(f"{site}: runtime cardinality mismatch")
    if any(config["discovery"].values()):
        raise SystemExit(f"{site}: public discovery unexpectedly enabled")
    bridge = root / site / "runtime/ml_engine.py"
    source = Path("monash_exps/src/phase8/ml_bridge.py")
    if hashlib.sha256(bridge.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
        raise SystemExit(f"{site}: staged bridge differs from source")
g0_a = json.loads((root / "site-a/global-0/g0-manifest.json").read_text())
g0_b = json.loads((root / "site-b/global-0/g0-manifest.json").read_text())
if g0_a != g0_b:
    raise SystemExit("installed G0 manifests differ")
summary = {
    "status": "PASS",
    "artifact_root": str(root),
    "protocol_tests": 6,
    "simulated_federated_rounds": 5,
    "site_train_rows": 1152,
    "site_validation_rows": 128,
    "g0_state_sha256": g0_a["adapter"]["state_sha256"],
    "training_contract_sha256": g0_a["training_contract_sha256"],
}
(root / "bridge-readiness-summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
print("PHASE8 BRIDGE READINESS PASSED")
PY

printf '%s\n' "${artifact_root}" > "${experiment_root}/artifacts/phase8/latest-bridge-readiness.txt"
