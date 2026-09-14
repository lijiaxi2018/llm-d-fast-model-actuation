## Export Model Path
export HF_HOME=/nvme/jiaxili3/hf

## Configure nvcc env
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

## Export cuda-checkpoint Path
export PATH=/nvme/jiaxili3/Research/swcr/SWCR/cuda-checkpoint/bin/x86_64_Linux:$PATH

## Export criu paths
export CUDA_CHECKPOINT_DIR=/nvme/jiaxili3/Research/swcr/SWCR/cuda-checkpoint/bin/x86_64_Linux
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

nvidia-smi
nvidia-smi -L

curl -sS http://127.0.0.1:8001/health
curl -sS http://127.0.0.1:8001/v2/vllm/instances

## Drop Page Cache
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

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

## cuda-checkpoint
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/cuda-checkpoint/toggle"

## criu
time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/criu/checkpoint"

time curl -sS -X POST "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/criu/restore"

### Level 1 Sleep
time curl -X POST 'http://127.0.0.1:8000/sleep?level=1'
time curl -X POST 'http://127.0.0.1:8000/wake_up'

curl -sS 'http://127.0.0.1:8000/is_sleeping'

### Level 2 Sleep
time curl -X POST 'http://127.0.0.1:8000/sleep?level=2'
time curl -X POST 'http://127.0.0.1:8000/wake_up?tags=weights'
time curl -X POST 'http://127.0.0.1:8000/collective_rpc' -H 'Content-Type: application/json' -d '{"method":"reload_weights"}'
time curl -X POST 'http://127.0.0.1:8000/wake_up?tags=kv_cache'

## General DRAM Usage Check
PROCESS_PID=2352732

echo PROCESS_PID=$PROCESS_PID

sudo awk '
/^Pss:/ {pss=$2}
/^Rss:/ {rss=$2}
END {
    printf "PSS: %.3f GiB\n", pss / 1048576
    printf "RSS: %.3f GiB\n", rss / 1048576
}
' "/proc/$PROCESS_PID/smaps_rollup"

INSTANCE_ID=qwen3-8b-local
FMA_CRIU_ROOT="${FMA_CRIU_ROOT:-/var/lib/fma-criu}"

## vllm DRAM Usage Check
VLLM_ROOT_PID=2390166
ENGINE_PID=2390231

for PID in "$VLLM_ROOT_PID" "$ENGINE_PID"; do
    echo "PID $PID"

    sudo awk '
    /^Pss:/ {pss=$2}
    /^Rss:/ {rss=$2}
    END {
        printf "PSS: %.3f GiB\n", pss / 1048576
        printf "RSS: %.3f GiB\n", rss / 1048576
    }
    ' "/proc/$PID/smaps_rollup"
done

## Check Raw Model Size
sudo du -sh "/nvme/jiaxili3/hf/hub/models--Qwen--Qwen3-4B"

## Check Image Size
INSTANCE_ID=qwen3-4b-local
sudo du -sh "/nvme/jiaxili3/Research/fma/checkpoints/$INSTANCE_ID/images"

## Find Root PID
ENGINE_PID=2390231

VLLM_ROOT_PID=$(
  ps -o ppid= -p "$ENGINE_PID" |
  tr -d ' '
)

echo "engine_pid=$ENGINE_PID"
echo "vllm_root_pid=$VLLM_ROOT_PID"

ps -o pid,ppid,pgid,stat,cmd \
  -p "$VLLM_ROOT_PID","$ENGINE_PID"