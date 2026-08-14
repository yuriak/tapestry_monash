#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: playit_agent.sh claim|start|status|stop" >&2
    exit 2
fi
action="$1"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
experiment_root="$(cd "${script_dir}/../.." && pwd)"
workspace_root="$(cd "${experiment_root}/.." && pwd)"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

playit_cli="${SLAKSHNA_PLAYIT_BIN}"
playit_daemon="${SLAKSHNA_PLAYIT_DAEMON}"
secret_dir="${SLAKSHNA_RUNTIME_ROOT}/secrets/playit"
secret_path="${PHASE9_PLAYIT_SECRET_PATH:-${secret_dir}/phase9-agent.toml}"
legacy_secret="${secret_dir}/phase8-m3-agent.toml"
state_dir="${SLAKSHNA_RUNTIME_ROOT}/run/playit"
pid_path="${state_dir}/phase9-agent.pid"
log_dir="${SLAKSHNA_RUNTIME_ROOT}/logs/playit"
log_path="${log_dir}/phase9-agent.log"
socket_path="${TMPDIR:-/tmp}/slakshna-playit-${UID}-phase9.sock"
required_tunnels="${PHASE9_PLAYIT_REQUIRED_TUNNELS:-1}"

umask 077
mkdir -p "${secret_dir}" "${state_dir}" "${log_dir}"
chmod 700 "${secret_dir}" "${state_dir}" "${log_dir}"
[[ -x "${playit_cli}" && -x "${playit_daemon}" ]] || {
    echo "Pinned playit binaries are missing; run monash_exps/scripts/setup.sh first." >&2
    exit 1
}
if [[ ! "${required_tunnels}" =~ ^[1-9][0-9]*$ ]]; then
    echo "PHASE9_PLAYIT_REQUIRED_TUNNELS must be a positive integer" >&2
    exit 2
fi

migrate_legacy_secret() {
    if [[ ! -s "${secret_path}" && -s "${legacy_secret}" ]]; then
        install -m 0600 "${legacy_secret}" "${secret_path}"
        echo "Reused the previously claimed Playit agent secret for Phase 9."
    fi
}

read_pid() {
    if [[ -s "${pid_path}" ]]; then
        local value
        value="$(< "${pid_path}")"
        [[ "${value}" =~ ^[0-9]+$ ]] && printf '%s\n' "${value}"
    fi
}

is_our_daemon() {
    local pid="$1" executable
    [[ "${pid}" =~ ^[0-9]+$ && -e "/proc/${pid}/exe" ]] || return 1
    executable="$(readlink -f "/proc/${pid}/exe" 2>/dev/null || true)"
    [[ "${executable}" == "$(readlink -f "${playit_daemon}")" ]] || return 1
    tr '\0' '\n' < "/proc/${pid}/cmdline" | grep -Fx -- "${socket_path}" >/dev/null
}

running_pid() {
    local pid
    pid="$(read_pid || true)"
    if [[ -n "${pid}" ]] && is_our_daemon "${pid}"; then
        printf '%s\n' "${pid}"
        return 0
    fi
    return 1
}

status_text() {
    "${playit_cli}" --socket-path "${socket_path}" status 2>&1
}

start_daemon() {
    local pid
    if pid="$(running_pid)"; then
        echo "Phase 9 playitd is already running (pid=${pid})."
        return 0
    fi
    rm -f -- "${pid_path}"
    if [[ -e "${socket_path}" ]]; then
        if "${playit_cli}" --socket-path "${socket_path}" status >/dev/null 2>&1; then
            echo "A live Playit service owns ${socket_path}; refusing to replace it." >&2
            exit 1
        fi
        rm -f -- "${socket_path}"
    fi
    : > "${log_path}"
    nohup "${playit_daemon}" \
        --secret-path "${secret_path}" \
        --socket-path "${socket_path}" \
        --log-path "${log_path}" \
        >/dev/null 2>&1 &
    pid=$!
    printf '%s\n' "${pid}" > "${pid_path}"
    for _ in $(seq 1 60); do
        if ! is_our_daemon "${pid}"; then
            echo "playitd exited during startup; inspect ${log_path}" >&2
            exit 1
        fi
        [[ -S "${socket_path}" ]] && return 0
        sleep 1
    done
    echo "playitd did not create its IPC socket" >&2
    exit 1
}

wait_for_tunnels() {
    local count="" stable_samples=0 pid
    for _ in $(seq 1 90); do
        if ! pid="$(running_pid)"; then
            echo "Managed playitd exited while waiting for its tunnels; inspect ${log_path}" >&2
            return 1
        fi
        # A single structured event also contains nested connection counters.
        # Select the first tunnel_count after "tunnels loaded" rather than the
        # final (usually zero) nested counter on that same line.
        count="$(sed -E $'s/\\x1B\\[[0-9;]*[mK]//g' "${log_path}" | awk '
            /playit connected; tunnels loaded/ {
                line = $0
                sub(/^.*playit connected; tunnels loaded/, "", line)
                if (match(line, /tunnel_count=[0-9]+/)) {
                    value = substr(line, RSTART + 13, RLENGTH - 13)
                    print value
                }
            }
        ' | tail -1)"
        if [[ "${count:-0}" =~ ^[0-9]+$ ]] && (( count >= required_tunnels )); then
            ((stable_samples += 1))
            if (( stable_samples >= 3 )); then
                echo "Playit agent online with ${count} active tunnel(s) (pid=${pid})."
                status_text || true
                return 0
            fi
        else
            stable_samples=0
        fi
        sleep 1
    done
    echo "Playit loaded ${count:-0} tunnel(s), but ${required_tunnels} are required." >&2
    echo "Create/assign the missing UDP tunnel(s) in the Playit dashboard." >&2
    echo "Daemon diagnostics are stored locally at ${log_path}." >&2
    return 1
}

stop_daemon() {
    local pid
    if ! pid="$(running_pid)"; then
        echo "No managed Phase 9 playitd process is running."
        return 0
    fi
    kill -TERM "${pid}"
    for _ in $(seq 1 30); do
        if ! is_our_daemon "${pid}"; then
            rm -f -- "${pid_path}" "${socket_path}"
            echo "Phase 9 playitd stopped."
            return 0
        fi
        sleep 1
    done
    echo "playitd did not stop after SIGTERM" >&2
    return 1
}

case "${action}" in
    claim)
        migrate_legacy_secret
        start_daemon
        if [[ -s "${secret_path}" ]]; then
            chmod 600 "${secret_path}"
            wait_for_tunnels
            exit 0
        fi
        echo "Open and approve the claim URL printed below."
        "${playit_cli}" --socket-path "${socket_path}" setup
        [[ -s "${secret_path}" ]] || {
            echo "Playit setup did not create ${secret_path}" >&2
            exit 1
        }
        chmod 600 "${secret_path}"
        wait_for_tunnels
        ;;
    start)
        migrate_legacy_secret
        [[ -s "${secret_path}" ]] || {
            echo "Agent is not claimed. Run: bash $0 claim" >&2
            exit 1
        }
        chmod 600 "${secret_path}"
        start_daemon
        wait_for_tunnels
        ;;
    status)
        running_pid >/dev/null || {
            echo "No managed Phase 9 playitd process is running." >&2
            exit 1
        }
        status_text
        ;;
    stop)
        stop_daemon
        ;;
    *)
        echo "Unknown action: ${action}" >&2
        exit 2
        ;;
esac
