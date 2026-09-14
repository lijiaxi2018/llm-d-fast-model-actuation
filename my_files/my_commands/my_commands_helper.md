## CLose FDs
export VLLM_ROOT_PID=$(
  curl -sS \
    "http://127.0.0.1:8001/v2/vllm/instances/${INSTANCE_ID}/log" |
  sed -n 's/.*VLLM process (PID: \([0-9][0-9]*\)) started.*/\1/p' |
  tail -n 1
)

sudo kill -USR1 "$VLLM_ROOT_PID"
sleep 1

sudo find \
  "/proc/$VLLM_ROOT_PID/fd" \
  -maxdepth 1 \
  -type l \
  -printf '%f -> %l\n' |
grep nvidia || true

export ENGINE_PID=$(
  pgrep -P "$VLLM_ROOT_PID" -f EngineCore |
  head -n 1
)

echo "$ENGINE_PID"

sudo find \
  "/proc/$ENGINE_PID/fd" \
  -maxdepth 1 \
  -type l \
  -printf '%f -> %l\n' |
grep nvidia |
head

## Kill Launcher
sudo fuser -k -9 8001/tcp