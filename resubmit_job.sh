#!/bin/bash

# This script repeatedly submits a Slurm job, waits for it to start,
# and then sleeps for 6 hours before resubmitting.

if [ -z "$1" ]; then
    echo "Usage: $0 <command_to_run...>"
    exit 1
fi

while true
do
    # Create slurm_output directory if it doesn't exist
    mkdir -p slurm_output

    # Submit the job and capture the output
    sbatch_output=$("$@")

    # Check if the job was submitted successfully
    if [ $? -ne 0 ]; then
        echo "Failed to submit sbatch job. Retrying in 1 minute."
        sleep 5
        continue
    fi

    # Extract the job ID from the sbatch output
    job_id=$(echo "$sbatch_output" | awk '{print $4}')

    # Check if job_id is a number
    if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
        echo "Could not parse job ID from sbatch output: $sbatch_output"
        echo "Retrying in 1 minute."
        sleep 5
        continue
    fi

    echo "Submitted job with ID: $job_id"

    # Define the expected output and error files
    out_file="slurm_output/tr_${job_id}.out"
    err_file="slurm_output/tr_${job_id}.err"

    echo "Waiting for job ${job_id} to start (checking for ${out_file} and ${err_file})..."

    # Wait until both the .out and .err files are created
    while ! [ -f "$out_file" ] || ! [ -f "$err_file" ]; do
        sleep 5 # Check every minute
    done

    echo "Job ${job_id} has started."
    echo "Sleeping for 6 hours before resubmitting."

    # Sleep for 6 hours
    sleep 6h
    sleep 300 # Additional sleep to ensure the job has time to run
done
