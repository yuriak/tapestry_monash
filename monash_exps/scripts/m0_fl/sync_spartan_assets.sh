#!/usr/bin/env bash
set -euo pipefail

mode="${1:---dry-run}"
case "${mode}" in
    --dry-run) dry_run=(--dry-run) ;;
    --execute) dry_run=() ;;
    *) echo "Usage: bash monash_exps/scripts/m0_fl/sync_spartan_assets.sh --dry-run|--execute" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
remote="${M0_FL_SPARTAN_HOST:-spartan}"
remote_root="${M0_FL_SPARTAN_ROOT:-/home/yebai1/workspace/tapestry_monash}"
cache_rel="monash_exps/.runtime/data/m0/tokenized/olmo2-7b-chatml-seq1024"
round_shards_rel="monash_exps/.runtime/data/m0/fl_round_shards"

cd "${workspace}"
for view in australia_nz south_asia; do
    test -d "${cache_rel}/${view}"
done
test -s monash_exps/.runtime/artifacts/m0/g0/olmo2-7b-r16-qv-seed20260820.pth
test -d monash_exps/.runtime/models/m0/OLMo-2-1124-7B-Instruct
for site in au india; do
    test -s "${round_shards_rel}/${site}/round-shards-manifest.json"
    test "$(find "${round_shards_rel}/${site}" -name '*.parquet' | wc -l)" -eq 10
done

if [[ "${mode}" == "--execute" ]]; then
    ssh -o BatchMode=yes "${remote}" \
        "mkdir -p '${remote_root}/${cache_rel}' '${remote_root}/${round_shards_rel}' '${remote_root}/monash_exps/.runtime/downloads/m0_fl'"
fi

rsync -a --checksum --info=stats2,progress2 "${dry_run[@]}" \
    "${cache_rel}/australia_nz" "${cache_rel}/south_asia" \
    "${remote}:${remote_root}/${cache_rel}/"

rsync -a --checksum --delete --info=stats2,progress2 "${dry_run[@]}" \
    "${round_shards_rel}/au" "${round_shards_rel}/india" \
    "${remote}:${remote_root}/${round_shards_rel}/"

wheel="monash_exps/.runtime/downloads/m0_fl/flash_attn-2.8.3+cu128torch2.9-cp311-cp311-linux_x86_64.whl"
if [[ -s "${wheel}" ]]; then
    rsync -a --checksum --info=stats2,progress2 "${dry_run[@]}" \
        "${wheel}" "${remote}:${remote_root}/monash_exps/.runtime/downloads/m0_fl/"
fi

echo "Model and G0 were deliberately not copied: verified copies already exist on Spartan."
echo "Repository code and Slakshna were not copied: synchronize them with Git after review."
