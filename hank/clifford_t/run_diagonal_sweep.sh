#!/usr/bin/env bash
# run_diagonal_sweep.sh
#
# Runs gen_diagonal_un.py with Dim=2 over values 1/1, 1/2, ..., 1/120.
#
# Usage:
#   ./run_diagonal_sweep.sh              # sequential
#   ./run_diagonal_sweep.sh 8            # up to 8 jobs in parallel

MAX_JOBS=${1:-1}     # parallel job limit (1 = sequential)
DIM=2
N_STEPS=120

echo "Starting sweep: Dim=${DIM}, values 1/1 down to 1/${N_STEPS}"
echo "Max parallel jobs: ${MAX_JOBS}"
echo "-------------------------------------------"

job_count=0

for i in $(seq 1 ${N_STEPS}); do
    # Compute value = 1/i as a float using awk
    value=$(awk "BEGIN { printf \"%.10f\", 1.0 / ${i} }")

    echo "  [${i}/${N_STEPS}] python3 gen_diagonal_un.py ${DIM} 0 ${value}"
    python3 gen_diagonal_un.py ${DIM} 0 ${value} &

    job_count=$((job_count + 1))

    # If we've reached the job limit, wait for all current jobs to finish
    if [ "${job_count}" -ge "${MAX_JOBS}" ]; then
        wait
        job_count=0
    fi
done

# Wait for any remaining background jobs
wait

echo "-------------------------------------------"
echo "Sweep complete."
