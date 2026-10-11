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

### debug mode
sudo -E env \
    PATH="$CUDA_CHECKPOINT_DIR:$VENV_BIN:/usr/local/cuda-12.8/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    FMA_CRIU_ROOT="$FMA_CRIU_ROOT" \
    "$VENV_PYTHON" launcher.py \
        --host 127.0.0.1 \
        --port 8001 \
        --log-level debug

### 751
####
curl --noproxy '*' --fail --max-time 2 -i \
  http://127.0.0.1:8001/health

####
(
  while true; do
    for endpoint in /health /v2/vllm/instances; do
      printf '%s %-20s ' "$(date +%T)" "$endpoint"

      curl --noproxy '*' -sS \
        --connect-timeout 1 \
        --max-time 1 \
        -o /dev/null \
        -w 'HTTP=%{http_code} time=%{time_total}s\n' \
        "http://127.0.0.1:8001${endpoint}" || true
    done

    sleep 0.2
  done
) > /tmp/fma-pr751-probes.log 2>&1 &

FMA_REPRO_PROBE_PID=$!

####
tail -n 10 /tmp/fma-pr751-probes.log

####
ps -eo pid,ppid,pgid,args | grep '[l]auncher.py'

####
sudo kill -STOP 646998

####
curl --max-time 60 -sS -X DELETE \
  http://127.0.0.1:8001/v2/vllm/instances/qwen3-4b-local \
  -w '\nHTTP=%{http_code} elapsed=%{time_total}s\n'

####
tail -n 600 /tmp/fma-pr751-probes.log