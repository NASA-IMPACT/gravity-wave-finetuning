#!/usr/bin/sh

###################################################################

#   This is a bash script called by pbs script cluster_nas.pbs 

### Placeholders:

#  <ENVIRONMENT_PATH>: Path to your environment initialization script.
#  <MODULE_PATH>: Path to the module files directory.
#  <MODULE_NAME>: Name of the module to load (e.g., miniconda3/v4).
#  <CONDA_ENV_PATH>: Path to your Conda environments.
#  <YOUR_PATH>: Any specific path you want to add to the Python path.
#  <SCRIPT_NAME>: Name of the script you are running (e.g., finetune_gravity_wave).

###################################################################

function ExitErr {
  echo "Error: $1"
  exit 1
}

if [ "$#" -ne 6 ]; then
    ExitErr "Usage: $0 <data_args> <node_rank> <num_nodes> <master_addr> <master_port> <job_id>"
fi

# Get worker info
data_args=$1
export NODE_RANK=$2
export NUM_NODES=$3
export RDZV_ADDR=$4
export RDZV_PORT=$5
export RDZV_ID=$6

#export WORLD_SIZE=$3

# reload the module and environment on slave node
if [[ $NODE_RANK -ne 0 ]]; then
    ENV=<ENVIRONMENT_PATH>; export ENV
    . $ENV
fi

module purge
module use -a <MODULE_PATH>
module load <MODULE_NAME>
export CONDA_ENVS_PATH=<CONDA_ENV_PATH>
source activate pt24
sleep 5

# Set MPI_LAUNCH_TIMEOUT to longer values so code has time to load
export MPI_LAUNCH_TIMEOUT=60

# Set std out buffer to have no limit
export MPI_UNBUFFERED_STDIO=true

# Set NCCL to use IP-over-Infiniband rather than default RoCE
export NCCL_IB_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Used for debugging, helps print which nodes are used in code
export TORCH_DISTRIBUTED_DEBUG=INFO

export PYTHONPATH=$PWD:$PWD/<YOUR_PATH>:$PYTHONPATH
echo "Python path: $PYTHONPATH"

# Run the script with torchrun
echo "Torchrun on node $NODE_RANK, total nodes: $NUM_NODES" >&2

torchrun \
    --nproc_per_node=4 \
    --nnodes=$NUM_NODES \
    --node_rank $NODE_RANK \
    --rdzv_id=$RDZV_ID \
    --rdzv_endpoint "$RDZV_ADDR:$RDZV_PORT" \
    --rdzv_backend=c10d \
    ./prithviwxc/gravitywave/<SCRIPT_NAME>.py --split $data_args

conda deactivate