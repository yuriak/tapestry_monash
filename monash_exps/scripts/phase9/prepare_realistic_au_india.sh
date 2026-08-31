#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
slakshna_root="${workspace}/Slakshna"
runtime_root="${slakshna_root}/.runtime_views/realistic_au_india_20260830"
india_cache="${workspace}/monash_exps/.runtime/data/phase9/tokenized/south_asia_seq2048_packed_20260830/local_train_079a0daaff30bc4e"

[[ -f "${india_cache}/metadata.json" ]] || {
    echo "ERROR: validated South Asia cache is missing: ${india_cache}" >&2
    exit 1
}

for site in au india; do
    view="${runtime_root}/${site}"
    mkdir -p "${view}/logs" "${view}/data" "${view}/ml_models" \
        "${view}/ml_states" "${view}/cache" "${view}/rust_state"
    ln -sfn "${slakshna_root}/ml_engine.py" "${view}/ml_engine.py"
    ln -sfn "${slakshna_root}/federated_communication" "${view}/federated_communication"
    ln -sfn "${slakshna_root}/hf_cache" "${view}/hf_cache"
done

cp "${slakshna_root}/node_template_realistic_au.yaml" \
    "${runtime_root}/au/node_template.yaml"
cp "${slakshna_root}/node_template_realistic_india.yaml" \
    "${runtime_root}/india/node_template.yaml"
cp "${slakshna_root}/node_realistic_au.toml" \
    "${runtime_root}/au/node.toml"
cp "${slakshna_root}/node_realistic_india.toml" \
    "${runtime_root}/india/node.toml"

echo "Prepared isolated runtime views:"
echo "  AU    ${runtime_root}/au"
echo "  India ${runtime_root}/india"
