#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
runtime_store="${M0_FL_SPARTAN_OUTPUT_ROOT:-/data/gpfs/projects/punim2961/yebai1/tapestry_monash/m0_fl_runtime}"
runtime_link="${workspace}/Slakshna/m0_runtime"
slakshna_exclude="$(git -C "${workspace}/Slakshna" rev-parse --git-path info/exclude)"

cd "${workspace}"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    echo "Tracked repository state is dirty; finish Git synchronization first." >&2
    git status --short >&2
    exit 1
}
mkdir -p "${runtime_store}/slurm" "${runtime_store}/local_fl"
if [[ -e "${runtime_link}" || -L "${runtime_link}" ]]; then
    [[ "$(readlink -f "${runtime_link}")" == "$(readlink -f "${runtime_store}")" ]] || {
        echo "Existing runtime link points elsewhere: ${runtime_link}" >&2
        exit 1
    }
else
    ln -s "${runtime_store}" "${runtime_link}"
fi
# The output symlink is intentionally machine-local. Keep the upstream
# Slakshna submodule clean without changing its tracked .gitignore.
if ! grep -Fxq '/m0_runtime' "${slakshna_exclude}" 2>/dev/null; then
    printf '%s\n' '/m0_runtime' >> "${slakshna_exclude}"
fi

export SLAKSHNA_CLUSTER=spartan
export SLAKSHNA_SPARTAN_COMPILER_MODULE="${SLAKSHNA_SPARTAN_COMPILER_MODULE:-GCCcore/13.3.0}"
export SLAKSHNA_SPARTAN_LLVM_MODULE="${SLAKSHNA_SPARTAN_LLVM_MODULE:-LLVM/18.1.8}"
export SLAKSHNA_LIBCLANG_PATH="${SLAKSHNA_LIBCLANG_PATH:-/apps/easybuild-2022/easybuild/software/Compiler/GCCcore/13.3.0/LLVM/18.1.8/lib}"
bash monash_exps/scripts/m0_fl/01_upgrade_environment.sh --skip-gpu-check

for view in australia_nz south_asia; do
    test -n "$(find "monash_exps/.runtime/data/m0/tokenized/olmo2-7b-chatml-seq1024/${view}" -name '*.parquet' -print -quit)"
done
for site in au india; do
    test -s "monash_exps/.runtime/data/m0/fl_round_shards/${site}/round-shards-manifest.json"
    test "$(find "monash_exps/.runtime/data/m0/fl_round_shards/${site}" -name '*.parquet' | wc -l)" -eq 10
done
test -s monash_exps/.runtime/artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth
test -d monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct
test -x monash_exps/.runtime/tools/playit/bin/playitd
test -x monash_exps/.runtime/cargo-target/slakshna/release/iiitd

echo "Spartan environment and assets are ready."
echo "Next: configure the India ingress with the reused Phase 9 Playit agent."
