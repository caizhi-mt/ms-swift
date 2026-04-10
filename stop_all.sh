#!/usr/bin/env bash
set -euo pipefail

HOSTFILE="./hostfile"

if [[ ! -f "${HOSTFILE}" ]]; then
    echo "[ERROR] hostfile not found: ${HOSTFILE}"
    exit 1
fi

mapfile -t HOSTS < <(
    awk '
        NF > 0 && $1 !~ /^#/ {
            print $1
        }
    ' "${HOSTFILE}" | awk '!seen[$0]++'
)

if [[ ${#HOSTS[@]} -eq 0 ]]; then
    echo "[ERROR] no valid hosts found in ${HOSTFILE}"
    exit 1
fi

echo "[INFO] hosts to stop: ${HOSTS[*]}"

for host in "${HOSTS[@]}"; do
    echo "[WARN] killing ALL python processes on ${host} ..."
    ssh "${host}" "pkill -9 python || true" || echo "[WARN] failed to kill python on ${host}"
done

echo "Stopping, please wait..."
sleep 5

echo "[INFO] done."
