#!/bin/bash

#SBATCH --job-name=bench-picotron
#SBATCH --time=00:30:00
#SBATCH --partition=hopper-prod
#SBATCH --nodes={{ nodes }}
#SBATCH --gres=gpu:{{ n_proc_per_node }}
#SBATCH --qos={{ qos }}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --output={{ root_path }}/log_%j.out
#SBATCH --error={{ root_path }}/log_%j.out

# Function to update status based on squeue output
update_status() {
    job_id=$1
    status_file=$2
    # For unknown reasons, it doenst update status for pending. It only works for running 
    while true; do
        job_status=$(squeue --job $job_id --noheader --format=%T)
        echo "Job status: $job_status"
        if [ -z "$job_status" ]; then
            # Job has finished or is not found
            break
        elif [ "$job_status" = "RUNNING" ]; then
            printf "running" > $status_file
            break
        fi
        sleep 10
    done
}

# Misc initializations.
echo "========================"
echo "START TIME: $(date)"
source /etc/profile.d/modules.sh
source /fsx/ferdinandmom/miniforge3/etc/profile.d/conda.sh
conda activate /fsx/ferdinandmom/miniforge3/envs/env-picotron
echo python3 version = $(python3 --version)
echo "========================"

# Slurm stuff
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((1024 + RANDOM % 64511))

export TMPDIR=/scratch
export TORCH_HOME=/fsx/$USER/.cache/torch
export HF_HOME=/fsx/$USER/.cache/huggingface
export WANDB_DIR=/fsx/$USER/.cache/wandb
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"

module load cuda/12.1

GIT_REPO="/fsx/ferdinandmom/ferdinand-hf/picotron/"
CMD="$GIT_REPO/train.py --config {{ config }}"

LAUNCHER="torchrun --nproc_per_node={{ n_proc_per_node }} --nnode={{ nodes }} --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT"

# Checkout the bench_cluster branch
cd $GIT_REPO
# Get the current job ID
job_id=${SLURM_JOB_ID}

# Update status to "pending" or "running" in the background
update_status $job_id {{ root_path }}/status.txt &

# Run the main command
echo "Running command: $CMD"
srun -u $LAUNCHER $CMD
exit_status=$?

job_id=$SLURM_JOB_ID

# Update status based on the exit status of `srun`
if [ $exit_status -eq 0 ]; then
    printf "completed" > {{ root_path }}/status.txt
else
    if grep -q "OutOfMemoryError" {{ root_path }}/log_${job_id}.out; then
        printf "oom" > {{ root_path }}/status.txt
    elif grep -q " CUDA error: an illegal memory access" {{ root_path }}/log_${job_id}.out; then
        printf "oom" > {{ root_path }}/status.txt
    elif grep -q "Timeout" {{ root_path }}/log_${job_id}.out; then
        printf "timeout" > {{ root_path }}/status.txt
    else
        printf "fail" > {{ root_path }}/status.txt
    fi 
fi