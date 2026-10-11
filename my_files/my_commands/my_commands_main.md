## io_uring
### Check
sysctl kernel.io_uring_disabled

### Disable
sudo sysctl kernel.io_uring_disabled=2

### Enable
sudo sysctl kernel.io_uring_disabled=0

## Export Model Path
export HF_HOME=/nvme/jiaxili3/hf

## Configure nvcc env
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

## Export cuda-checkpoint Path
export PATH=/nvme/jiaxili3/Research/snapshotting/cuda-checkpoint/bin/x86_64_Linux:$PATH

## Export criu paths
export CUDA_CHECKPOINT_DIR=/nvme/jiaxili3/Research/snapshotting/cuda-checkpoint/bin/x86_64_Linux
export VENV_PYTHON="$(command -v python3)"
export VENV_BIN="$(dirname "$VENV_PYTHON")"
export FMA_CRIU_ROOT=/nvme/jiaxili3/Research/fma/checkpoints

## Start the launcher
sudo -E env \
    PATH="$CUDA_CHECKPOINT_DIR:$VENV_BIN:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    FMA_CRIU_ROOT="$FMA_CRIU_ROOT" \
    "$VENV_PYTHON" launcher.py \
        --host 127.0.0.1 \
        --port 8001 \
        --log-level info

curl -sS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8001/v2/vllm/instances

## Start a vLLM
export INSTANCE_ID=qwen3-4b-local
export GPU_UUID=GPU-3e270923-d620-c3e7-83a4-d74d0f793006

curl -sS -X PUT \
    "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" \
    -H 'Content-Type: application/json' \
    -d "{
    \"options\": \"--model Qwen/Qwen3-4B --enable-sleep-mode --max-model-len 1024 --max-num-seqs 1 --kv-cache-memory-bytes 256M --port 8000\",
    \"gpu_uuids\": [\"${GPU_UUID}\"],
    \"env_vars\": {
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

curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/log" | tail -n 200

curl -sS \
    http://127.0.0.1:8000/v1/chat/completions \
        -H 'Content-Type: application/json' \
        -d '{
        "model": "Qwen/Qwen3-4B",
        "messages": [
            {
            "role": "user",
            "content": "Who is Aqours?"
            }
        ],
        "max_tokens": 128,
        "temperature": 0
    }'

curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}"

curl -sS -X DELETE "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}"
sudo du -sh "$CHECKPOINT_DIR"
sudo rm -r $CHECKPOINT_DIR
sudo du -sh "$CHECKPOINT_DIR"

## nvidia-smi
nvidia-smi
nvidia-smi -L

## Checkpoint
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq
nvidia-smi

### cuda-checkpoint --checkpoint
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/cuda/checkpoint" | jq
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq
nvidia-smi

### criu --dump
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/disk/checkpoint" | jq
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq

### check snapshot
CHECKPOINT_DIR=$(curl -sS \
  "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" |
  jq -er 'select(.status == "checkpointed_disk") | .checkpoint_dir')

sudo du -sh "$CHECKPOINT_DIR"
sudo tail -5 "$CHECKPOINT_DIR/dump.log"

### criu --restore only
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/disk/restore" | jq
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq

### criu --restore with cuda-checkpoint --restore
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/disk/restore?resume=true" | jq
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq

### cuda-checkpoint --restore
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/cuda/restore" | jq
curl -sS "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}" | jq