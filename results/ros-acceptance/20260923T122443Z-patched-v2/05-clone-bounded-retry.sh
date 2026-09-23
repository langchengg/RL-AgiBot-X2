set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
timeout --signal=INT --kill-after=10s 120s git -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 clone --depth 1 --filter=blob:none --no-checkout https://github.com/langchengg/RL-AgiBot-X2.git "$WS"
git -C "$WS" sparse-checkout set src
timeout --signal=INT --kill-after=10s 120s git -C "$WS" -c http.version=HTTP/1.1 -c http.lowSpeedLimit=1024 -c http.lowSpeedTime=30 checkout --detach 71037f7d5530a65da220345f9037ac4d29f32908
cd "$WS"
git rev-parse HEAD
git rev-parse origin/main
git status --short

