set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
git clone --depth 1 --filter=blob:none --no-checkout https://github.com/langchengg/RL-AgiBot-X2.git "$WS"
git -C "$WS" sparse-checkout set src
git -C "$WS" checkout --detach 71037f7d5530a65da220345f9037ac4d29f32908
cd "$WS"
git rev-parse HEAD
git rev-parse origin/main
git status --short
git sparse-checkout list
uname -a
cat /etc/os-release
systemd-detect-virt
df -h /tmp
free -h

