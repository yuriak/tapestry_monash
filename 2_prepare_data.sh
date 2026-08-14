#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${workspace_root}/monash_exps/scripts/phase9/prepare_data.sh" "$@"
