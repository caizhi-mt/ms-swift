#!/bin/bash
set -euo pipefail

bash build.sh

HOSTFILE="${HOSTFILE:-./hostfile}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-$(pwd)}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-run_qwen3_5_27b_node.sh}"
MASTER_PORT="${MASTER_PORT:-7788}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
SSH_USER="${SSH_USER:-}"
SSH_OPTS="${SSH_OPTS:--o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10}"
DRYRUN="${DRYRUN:-0}"

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "[ERROR] hostfile not found: ${HOSTFILE}"
    exit 1
fi

mapfile -t HOSTS < <(awk '/^[[:space:]]*#/ {next} NF {print $1}' "${HOSTFILE}")

NNODES="${#HOSTS[@]}"
if [[ "${NNODES}" -lt 1 ]]; then
    echo "[ERROR] no valid hosts found in ${HOSTFILE}"
    exit 1
fi

MASTER_ADDR="${HOSTS[0]}"
TIMESTAMP="$(date +"%Y%m%d_%H%M%S")"
LAUNCH_LOG_DIR="${REMOTE_WORKDIR}/logs/multinode_launch_${TIMESTAMP}"
mkdir -p "${LAUNCH_LOG_DIR}"

echo "[INFO] HOSTFILE=${HOSTFILE}"
echo "[INFO] NNODES=${NNODES}"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] REMOTE_WORKDIR=${REMOTE_WORKDIR}"
echo "[INFO] REMOTE_SCRIPT=${REMOTE_SCRIPT}"
echo "[INFO] NPROC_PER_NODE=${NPROC_PER_NODE}"
echo "[INFO] LAUNCH_LOG_DIR=${LAUNCH_LOG_DIR}"
echo

build_ssh_target() {
    local host="$1"
    if [[ -n "${SSH_USER}" ]]; then
        echo "${SSH_USER}@${host}"
    else
        echo "${host}"
    fi
}

for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    target="$(build_ssh_target "${host}")"
    echo "[CHECK] ssh ${target}"
    if ! ssh ${SSH_OPTS} "${target}" "echo connected: \$(hostname)" >/dev/null 2>&1; then
        echo "[ERROR] ssh failed for ${target}"
        exit 1
    fi
done

echo "[INFO] all ssh connectivity checks passed"
echo

PIDS=()

for i in "${!HOSTS[@]}"; do
    host="${HOSTS[$i]}"
    node_rank="${i}"
    target="$(build_ssh_target "${host}")"
    per_host_log="${LAUNCH_LOG_DIR}/launch_node${node_rank}_$(echo "${host}" | tr '/:' '__').log"

    remote_cmd=$(cat <<EOF
set -euo pipefail
cd "${REMOTE_WORKDIR}"

if [[ ! -f "${REMOTE_SCRIPT}" ]]; then
    echo "[ERROR] remote script not found: ${REMOTE_WORKDIR}/${REMOTE_SCRIPT}"
    exit 1
fi

chmod +x "${REMOTE_SCRIPT}"

export NNODES="${NNODES}"
export NODE_RANK="${node_rank}"
export MASTER_ADDR="${MASTER_ADDR}"
export MASTER_PORT="${MASTER_PORT}"
export NPROC_PER_NODE="${NPROC_PER_NODE}"

echo "================ REMOTE ENV ================"
echo "HOSTNAME=\$(hostname)"
echo "NNODES=\${NNODES}"
echo "NODE_RANK=\${NODE_RANK}"
echo "MASTER_ADDR=\${MASTER_ADDR}"
echo "MASTER_PORT=\${MASTER_PORT}"
echo "NPROC_PER_NODE=\${NPROC_PER_NODE}"
echo "PWD=\$(pwd)"
echo "============================================"

bash "${REMOTE_SCRIPT}"
EOF
)

    echo "[LAUNCH] node_rank=${node_rank} host=${host}"

    if [[ "${DRYRUN}" == "1" ]]; then
        echo "ssh ${SSH_OPTS} ${target} '${remote_cmd}'"
        continue
    fi

    ssh ${SSH_OPTS} "${target}" "${remote_cmd}" > "${per_host_log}" 2>&1 &
    PIDS+=("$!")
done

if [[ "${DRYRUN}" == "1" ]]; then
    echo "[INFO] dryrun done"
    exit 0
fi

echo
echo "[INFO] all launch commands submitted"
echo "[INFO] launch logs are under: ${LAUNCH_LOG_DIR}"
echo

FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        FAIL=1
    fi
done

if [[ "${FAIL}" -ne 0 ]]; then
    echo "[ERROR] one or more remote launch commands failed"
    exit 1
fi

echo "[INFO] all remote ssh sessions finished"
