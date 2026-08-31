#!/usr/bin/env bash
set -euo pipefail

[[ $# -ge 1 ]] || {
    echo "Usage: $0 claim|start|status|stop" >&2
    echo "       $0 write-config PUBLIC_IPV4 PUBLIC_PORT LOCAL_PORT" >&2
    exit 2
}
action="$1"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
workspace="$(cd "${script_dir}/../../.." && pwd -P)"
runtime="${workspace}/monash_exps/.runtime"
daemon="${runtime}/tools/playit/bin/playitd"
cli="${runtime}/tools/playit/bin/playit"
secret_root="${runtime}/secrets/playit"
state_root="${runtime}/run/m0_fl_m3_playit"
config_root="${runtime}/configs/m0_fl"
secret="${secret_root}/m0-fl-m3-backup-agent.toml"
socket="/tmp/m0-fl-playit-${UID}-m3-backup.sock"
pidfile="${state_root}/agent.pid"
log="${state_root}/agent.log"

umask 077
mkdir -p "${secret_root}" "${state_root}" "${config_root}"
chmod 700 "${secret_root}" "${state_root}" "${config_root}"

running_pid() {
    [[ -s "${pidfile}" ]] || return 1
    local pid
    pid="$(<"${pidfile}")"
    [[ "${pid}" =~ ^[0-9]+$ ]] || return 1
    kill -0 "${pid}" 2>/dev/null || return 1
    printf '%s\n' "${pid}"
}

start_agent() {
    local pid
    if pid="$(running_pid)"; then
        echo "M3 backup agent is already running (pid=${pid})."
        return 0
    fi
    rm -f -- "${pidfile}" "${socket}"
    : > "${log}"
    nohup "${daemon}" --secret-path "${secret}" --socket-path "${socket}" \
        --log-path "${log}" >/dev/null 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" > "${pidfile}"
    for _ in $(seq 1 60); do
        kill -0 "${pid}" 2>/dev/null || {
            echo "Playit daemon exited; inspect ${log}" >&2
            exit 1
        }
        [[ -S "${socket}" ]] && return 0
        sleep 1
    done
    echo "Playit IPC socket was not created: ${socket}" >&2
    exit 1
}

stop_agent() {
    local pid
    if pid="$(running_pid)"; then
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
        [[ ! -s "${secret}" ]] || {
            echo "The independent M3 backup agent is already claimed: ${secret}"
            exit 0
        }
        start_agent
        echo "Open the claim URL printed below in the Playit account."
        "${cli}" --socket-path "${socket}" setup
        test -s "${secret}"
        chmod 600 "${secret}"
        echo "Agent claimed and left online for dashboard tunnel assignment."
        echo "Assign a UDP tunnel to this agent, then record it with write-config."
        ;;
    start)
        [[ -s "${secret}" ]] || { echo "Claim the agent first." >&2; exit 1; }
        start_agent
        echo "M3 backup Playit agent is running."
        ;;
    status)
        [[ -s "${secret}" ]] || { echo "Claim the agent first." >&2; exit 1; }
        start_agent
        sleep 5
        "${cli}" --socket-path "${socket}" status
        ;;
    stop)
        stop_agent
        echo "M3 backup Playit agent stopped."
        ;;
    write-config)
        [[ $# -eq 4 ]] || {
            echo "write-config requires PUBLIC_IPV4 PUBLIC_PORT LOCAL_PORT" >&2
            exit 2
        }
        [[ -s "${secret}" ]] || { echo "Claim the agent first." >&2; exit 1; }
        stop_agent
        config_path="${config_root}/m3_playit.toml"
        cat > "${config_path}" <<EOF
[india]
public_host = "$2"
public_port = $3
local_port = $4
secret_path = "monash_exps/.runtime/secrets/playit/m0-fl-m3-backup-agent.toml"
EOF
        chmod 600 "${config_path}"
        echo "Wrote ${config_path}"
        ;;
    *) echo "Unknown action: ${action}" >&2; exit 2 ;;
esac
