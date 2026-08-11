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
cd "${workspace_root}"

# shellcheck source=../cluster/activate.sh
source "${experiment_root}/scripts/cluster/activate.sh"

playit_cli="${SLAKSHNA_PLAYIT_BIN}"
playit_daemon="${SLAKSHNA_PLAYIT_DAEMON}"
secret_dir="${SLAKSHNA_RUNTIME_ROOT}/secrets/playit"
secret_path="${secret_dir}/phase8-m3-agent.toml"
state_dir="${SLAKSHNA_RUNTIME_ROOT}/run/playit"
pid_path="${state_dir}/phase8-m3-agent.pid"
log_dir="${SLAKSHNA_RUNTIME_ROOT}/logs/playit"
log_path="${log_dir}/phase8-m3-agent.log"
socket_path="${TMPDIR:-/tmp}/slakshna-playit-${UID}-phase8.sock"

umask 077
mkdir -p "${secret_dir}" "${state_dir}" "${log_dir}"
chmod 700 "${secret_dir}" "${state_dir}" "${log_dir}"
[[ -x "${playit_cli}" && -x "${playit_daemon}" ]] || {
    echo "Pinned playit binaries are missing; run monash_exps/scripts/setup.sh first." >&2
    exit 1
}

read_pid() {
    if [[ -s "${pid_path}" ]]; then
        local value
        value="$(< "${pid_path}")"
        if [[ "${value}" =~ ^[0-9]+$ ]]; then printf '%s\n' "${value}"; fi
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

start_daemon() {
    local pid
    if pid="$(running_pid)"; then
        echo "playitd is already running (pid=${pid})."
        return 0
    fi
    rm -f -- "${pid_path}"
    if [[ -e "${socket_path}" ]]; then
        # The name is allocation/user scoped. Refuse to unlink a live endpoint.
        if "${playit_cli}" --socket-path "${socket_path}" status >/dev/null 2>&1; then
            echo "A live playit service already owns ${socket_path}; refusing to replace it." >&2
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
            tail -n 80 "${log_path}" >&2 || true
            exit 1
        fi
        if [[ -S "${socket_path}" ]]; then
            echo "playitd started (pid=${pid}, socket=${socket_path})."
            return 0
        fi
        sleep 1
    done
    echo "playitd did not create its IPC socket; inspect ${log_path}" >&2
    exit 1
}

status_text() {
    "${playit_cli}" --socket-path "${socket_path}" status 2>&1
}

is_healthy_status() {
    local status="$1"
    grep -Fq "Secret configured: true" <<< "${status}" || return 1
    grep -Fq "Last error:" <<< "${status}" && return 1
    grep -Eq "Phase: (running|connected|online|ready)" <<< "${status}"
}

wait_for_healthy() {
    local status=""
    for _ in $(seq 1 30); do
        status="$(status_text || true)"
        if is_healthy_status "${status}"; then
            printf '%s\n' "${status}"
            return 0
        fi
        sleep 1
    done
    printf '%s\n' "${status}" >&2
    echo "playitd did not reach a healthy online state; inspect ${log_path}" >&2
    tail -n 40 "${log_path}" >&2 || true
    return 1
}

stop_daemon() {
    local pid
    if ! pid="$(running_pid)"; then
        echo "No managed Phase 8 playitd process is running."
        return 0
    fi
    kill -TERM "${pid}"
    for _ in $(seq 1 30); do
        if ! is_our_daemon "${pid}"; then
            rm -f -- "${pid_path}" "${socket_path}"
            echo "playitd stopped."
            return 0
        fi
        sleep 1
    done
    echo "playitd did not stop after SIGTERM; refusing to send a broader signal." >&2
    exit 1
}

case "${action}" in
    claim)
        start_daemon
        existing_status="$(status_text || true)"
        if [[ -s "${secret_path}" ]]; then
            chmod 600 "${secret_path}"
            if is_healthy_status "${existing_status}"; then
                echo "The M3 Phase 8 playit agent is already claimed and online."
                printf '%s\n' "${existing_status}"
                exit 0
            fi
            echo "The existing agent secret is present but the daemon is not online."
            echo "Provisioning a replacement secret through a new browser claim."
        fi
        echo
        echo "Open the claim URL printed below in the registered playit account."
        echo "Approve the agent, then return here; the CLI will finish automatically."
        echo "The agent secret will not be printed or written to experiment logs."
        echo
        if ! "${playit_cli}" --socket-path "${socket_path}" setup; then
            stop_daemon || true
            exit 1
        fi
        [[ -s "${secret_path}" ]] || {
            echo "playit setup returned without creating the secret file." >&2
            stop_daemon || true
            exit 1
        }
        chmod 600 "${secret_path}"
        echo
        echo "Agent claim completed; secret file mode=$(stat -c '%a' "${secret_path}")."
        wait_for_healthy
        ;;
    start)
        [[ -s "${secret_path}" ]] || {
            echo "The agent is not claimed. Run: bash $0 claim" >&2
            exit 1
        }
        chmod 600 "${secret_path}"
        start_daemon
        wait_for_healthy
        ;;
    status)
        if running_pid >/dev/null; then
            "${playit_cli}" --socket-path "${socket_path}" status
        else
            echo "No managed Phase 8 playitd process is running."
            exit 1
        fi
        ;;
    stop)
        stop_daemon
        ;;
    *)
        echo "Unknown action: ${action}; expected claim|start|status|stop" >&2
        exit 2
        ;;
esac
