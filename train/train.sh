#!/bin/bash
set -x
# ========== Configurable Parameters ==========
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA=ib7s
export NCCL_DEBUG=WARN
echo nproc_per_node=$nproc_per_node
echo nnodes=$nnodes
echo node_rank=$node_rank
echo master_addr=$master_addr
echo master_port=$master_port
CONFIG_FILE="complexnet_config.yaml"    #
num_processes=$((nnodes * nproc_per_node))
echo num_processes=${num_processes}
echo NCCL_TIMEOUT=${NCCL_TIMEOUT}

# ========== Set Environment Variables ==========

source /root/.bashrc
conda activate newcomplex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ========== Launch Training ==========
accelerate launch \
  --config-file ${CONFIG_FILE} \
  --main_process_ip $master_addr \
  --main_process_port $master_port \
  --machine_rank $node_rank \
  --num_machines $nnodes \
  --num_processes ${num_processes} \
  ./train.py \
  --quant_method complex_phase_v2 \
  # --skip_lm_head \ (default False, i.e., lm_head will be replaced)


