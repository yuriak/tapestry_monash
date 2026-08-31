#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 2 ]] || {
    echo "Usage: $0 claim|status|stop india" >&2
    echo "       $0 write-config INDIA_HOST INDIA_PORT" >&2
    exit 2
}
action="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
runtime="${workspace}/monash_exps/.runtime"
daemon="${runtime}/tools/playit/bin/playitd"
cli="${runtime}/tools/playit/bin/playit"
secret_root="${runtime}/secrets/playit"
state_root="${runtime}/run/m0_fl_playit"
config_root="${runtime}/configs/m0_fl"
mkdir -p "${secret_root}" "${state_root}" "${config_root}"
chmod 700 "${secret_root}" "${state_root}" "${config_root}"

site_paths() {
    site="$1"
    [[ "${site}" == "india" ]] || {
        echo "Only the India site runs the single Playit ingress agent." >&2
        exit 2
    }
    secret="${secret_root}/phase9-agent.toml"
    socket="/tmp/m0-fl-playit-${UID}-setup-${site}.sock"
    pidfile="${state_root}/${site}.pid"
    log="${state_root}/${site}.log"
}

start_agent() {
    rm -f -- "${socket}"
    : > "${log}"
    nohup "${daemon}" --secret-path "${secret}" --socket-path "${socket}" \
        --log-path "${log}" >/dev/null 2>&1 &
    printf '%s\n' "$!" > "${pidfile}"
    for _ in $(seq 1 60); do
        [[ -S "${socket}" ]] && return 0
        kill -0 "$!" 2>/dev/null || { echo "Playit daemon exited; inspect ${log}" >&2; exit 1; }
        sleep 1
    done
    echo "Playit socket was not created: ${socket}" >&2
    exit 1
}

stop_agent() {
    if [[ -s "${pidfile}" ]]; then
        pid="$(<"${pidfile}")"
        kill "${pid}" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
    fi
    rm -f -- "${pidfile}" "${socket}"
}

case "${action}" in
    claim)
        site_paths "$2"
        [[ ! -s "${secret}" ]] || { echo "${site} already has a secret: ${secret}"; exit 0; }
        start_agent
        "${cli}" --socket-path "${socket}" setup
        test -s "${secret}"
        chmod 600 "${secret}"
        stop_agent
        echo "Claimed ${site}. Assign one UDP tunnel targeting 127.0.0.1:38080."
        ;;
    status)
        site_paths "$2"
        start_agent
        "${cli}" --socket-path "${socket}" status
        ;;
    stop)
        site_paths "$2"
        stop_agent
        ;;
    write-config)
        [[ $# -eq 3 ]] || { echo "write-config requires INDIA_HOST INDIA_PORT" >&2; exit 2; }
        [[ -s "${secret_root}/phase9-agent.toml" ]] || {
            echo "Phase 9 Playit secret is missing." >&2; exit 1;
        }
        config_path="${config_root}/spartan_playit.toml"
        cat > "${config_path}" <<EOF
[india]
public_host = "$2"
public_port = $3
local_port = 38080
secret_path = "monash_exps/.runtime/secrets/playit/phase9-agent.toml"
EOF
        chmod 600 "${config_path}"
        echo "Wrote ${config_path}"
        ;;
    *) echo "Unknown action: ${action}" >&2; exit 2 ;;
esac
