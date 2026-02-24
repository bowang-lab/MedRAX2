#!/bin/bash

set -u -o pipefail

CHUNK_SIZE=2
BENCHMARK_JOBS=2
MAX_ATTEMPTS=2
POLL_INTERVAL_SECONDS=5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SET_DIR="$SCRIPT_DIR/benchmarking/eval_sets"

current_medgemma_job_id=""
temp_csv_link=""

usage() {
  echo "Usage: $0 <csv_file_or_name>"
}

cleanup_current_medgemma_job() {
  if [[ -n "${current_medgemma_job_id}" ]]; then
    scancel "${current_medgemma_job_id}" >/dev/null 2>&1 || true
    current_medgemma_job_id=""
  fi
}

cleanup() {
  cleanup_current_medgemma_job
  if [[ -n "${temp_csv_link}" && -L "${temp_csv_link}" ]]; then
    rm -f "${temp_csv_link}"
  fi
}

trap cleanup EXIT INT TERM

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

input_csv="$1"
if [[ -f "${input_csv}" ]]; then
  csv_path="${input_csv}"
elif [[ -f "${EVAL_SET_DIR}/${input_csv}" ]]; then
  csv_path="${EVAL_SET_DIR}/${input_csv}"
else
  echo "Error: CSV file not found: ${input_csv}" >&2
  exit 1
fi

csv_name="$(basename "${csv_path}")"
csv_dir="$(cd "$(dirname "${csv_path}")" && pwd -P)"
eval_dir="$(cd "${EVAL_SET_DIR}" && pwd -P)"

if [[ "${csv_dir}" != "${eval_dir}" ]]; then
  temp_csv_link="${EVAL_SET_DIR}/.dispatch_${csv_name%.*}_$$.${csv_name##*.}"
  ln -sf "${csv_path}" "${temp_csv_link}"
  csv_name="$(basename "${temp_csv_link}")"
fi

mapfile -t pending_rows < <(python3 - "${csv_path}" <<'PY'
import csv
import sys

csv_path = sys.argv[1]

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if reader.fieldnames is None:
        raise SystemExit(f"CSV '{csv_path}' has no header row.")
    if "score" not in reader.fieldnames:
        raise SystemExit(f"CSV '{csv_path}' is missing required column: score")

    for idx, row in enumerate(reader, start=1):
        if not any((v or "").strip() for v in row.values()):
            continue
        if not (row.get("score", "") or "").strip():
            print(idx)
PY
)

if [[ ${#pending_rows[@]} -eq 0 ]]; then
  echo "No rows with empty score in ${csv_path}. Nothing to dispatch."
  exit 0
fi

echo "Found ${#pending_rows[@]} row(s) with empty score in ${csv_path}."

run_chunk_attempt() {
  local rows_selector="$1"
  local med_job_id=""
  local state_and_node=""
  local med_state=""
  local med_nodelist=""
  local med_node=""
  local med_api_url=""
  local benchmark_rc=0

  med_job_id="$(sbatch --parsable "${SCRIPT_DIR}/medgemma.sh" "${SCRIPT_DIR}")"
  if [[ -z "${med_job_id}" ]]; then
    echo "Failed to submit medgemma.sh for rows ${rows_selector}." >&2
    return 1
  fi
  current_medgemma_job_id="${med_job_id}"
  echo "Submitted MedGemma job ${med_job_id} for rows ${rows_selector}."

  while true; do
    state_and_node="$(squeue -h -j "${med_job_id}" -o "%T|%N" | head -n1 || true)"

    if [[ -z "${state_and_node}" ]]; then
      echo "MedGemma job ${med_job_id} no longer appears in queue before RUNNING." >&2
      cleanup_current_medgemma_job
      return 1
    fi

    med_state="${state_and_node%%|*}"
    med_nodelist="${state_and_node#*|}"

    if [[ "${med_state}" == "RUNNING" ]]; then
      if [[ -n "${med_nodelist}" && "${med_nodelist}" != "(null)" && "${med_nodelist}" != "None" ]]; then
        med_node="$(scontrol show hostnames "${med_nodelist}" | head -n1)"
        if [[ -n "${med_node}" ]]; then
          break
        fi
      fi
    fi

    case "${med_state}" in
      FAILED*|CANCELLED*|TIMEOUT*|PREEMPTED*|BOOT_FAIL*|NODE_FAIL*|OUT_OF_MEMORY*|DEADLINE*|REVOKED*)
        echo "MedGemma job ${med_job_id} entered terminal state ${med_state} before RUNNING." >&2
        cleanup_current_medgemma_job
        return 1
        ;;
    esac

    sleep "${POLL_INTERVAL_SECONDS}"
  done

  med_api_url="http://${med_node}:8002"
  echo "MedGemma job ${med_job_id} is RUNNING on ${med_node}."
  echo "Submitting benchmark rows ${rows_selector} with MEDGEMMA_API_URL=${med_api_url}."

  sbatch --wait \
    --export="ALL,MEDGEMMA_API_URL=${med_api_url}" \
    "${SCRIPT_DIR}/benchmark_from_csv.sh" \
    "${csv_name}" \
    "${rows_selector}" \
    "${BENCHMARK_JOBS}"
  benchmark_rc=$?

  cleanup_current_medgemma_job

  if [[ ${benchmark_rc} -ne 0 ]]; then
    echo "Benchmark submission failed for rows ${rows_selector} (exit ${benchmark_rc})." >&2
    return 1
  fi
  return 0
}

failed_chunks=()
total_rows=${#pending_rows[@]}

for ((start=0; start<total_rows; start+=CHUNK_SIZE)); do
  chunk_rows=("${pending_rows[@]:start:CHUNK_SIZE}")
  rows_selector="$(IFS=,; echo "${chunk_rows[*]}")"

  echo "=============================="
  echo "Dispatching chunk rows: ${rows_selector}"
  echo "=============================="

  chunk_ok=0
  for ((attempt=1; attempt<=MAX_ATTEMPTS; attempt++)); do
    echo "Attempt ${attempt}/${MAX_ATTEMPTS} for rows ${rows_selector}"
    if run_chunk_attempt "${rows_selector}"; then
      chunk_ok=1
      break
    fi
    echo "Attempt ${attempt} failed for rows ${rows_selector}."
  done

  if [[ ${chunk_ok} -eq 0 ]]; then
    failed_chunks+=("${rows_selector}")
    echo "Chunk failed after ${MAX_ATTEMPTS} attempts: ${rows_selector}" >&2
  fi
done

if [[ ${#failed_chunks[@]} -gt 0 ]]; then
  echo "Completed with failed chunks (${#failed_chunks[@]}):" >&2
  for chunk in "${failed_chunks[@]}"; do
    echo "  - ${chunk}" >&2
  done
  exit 1
fi

echo "All chunks completed successfully."
