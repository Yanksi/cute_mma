jid1=$(sbatch run_build.sh)
if [ $? -ne 0 ]; then
    echo "Failed to submit run_build.sh"
    exit 1
fi

# Extract the job ID from the output
jid1=$(echo $jid1 | awk '{print $4}')
if [ -z "$jid1" ]; then
    echo "Failed to capture job ID for run_build.sh"
    exit 1
fi

jid2=$(sbatch --dependency=afterok:$jid1 run_perf.sh)