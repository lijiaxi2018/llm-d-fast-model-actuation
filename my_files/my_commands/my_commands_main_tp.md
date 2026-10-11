### GPUs
export INSTANCE_ID=qwen3-4b-tp2-local
export GPU_UUID_0=GPU-3e270923-d620-c3e7-83a4-d74d0f793006
export GPU_UUID_1=GPU-0c4d3824-10df-e7f5-0fc3-3c60f7c7549e

### Directory where you built NCCL and its checkpoint shim.
export FMA_NCCL_SOURCE=/absolute/path/to/nccl-tp-checkpoint
export FMA_NCCL_LIB="$FMA_NCCL_SOURCE/build/lib/libnccl.so.2"
export FMA_NCCL_CHECKPOINT_LIB="$FMA_NCCL_SOURCE/contrib/nccl_checkpoint/build/lib/libnccl-checkpoint-shim.so"

curl --fail-with-body -sS -X PUT \
    "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" \
    -H 'Content-Type: application/json' \
    -d "{
    \"options\": \"--model Qwen/Qwen3-4B --host 127.0.0.1 --port 8000 --tensor-parallel-size 2 --enable-sleep-mode --max-model-len 1024 --max-num-seqs 1 --kv-cache-memory-bytes 268435456\",
    \"gpu_uuids\": [\"${GPU_UUID_0}\", \"${GPU_UUID_1}\"],
    \"env_vars\": {
        \"FMA_NCCL_LIB\": \"${FMA_NCCL_LIB}\",
        \"FMA_NCCL_CHECKPOINT_LIB\": \"${FMA_NCCL_CHECKPOINT_LIB}\",
        \"VLLM_SERVER_DEV_MODE\": \"1\",
        \"VLLM_LOGGING_LEVEL\": \"INFO\",
        \"CUDA_HOME\": \"/usr/local/cuda-12.8\",
        \"CUDA_PATH\": \"/usr/local/cuda-12.8\",
        \"CUDACXX\": \"/usr/local/cuda-12.8/bin/nvcc\"
    },
    \"annotations\": {
        \"inference-port\": \"8000\",
        \"isc-name\": \"qwen3-4b\"
    }
}"