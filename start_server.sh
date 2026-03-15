#!/bin/bash
source activate lmdeploy 2>/dev/null || conda activate lmdeploy 2>/dev/null
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,8,7,9
echo "Starting lmdeploy server..."
lmdeploy serve api_server /mnt/data3/MiniMax-M2.1-UD-Q6_K_XL-00001-of-00004.gguf --backend turbomind --tp 8 --server-port 23333 --session-len 4096 --cache-max-entry-count 0.3 2>&1
