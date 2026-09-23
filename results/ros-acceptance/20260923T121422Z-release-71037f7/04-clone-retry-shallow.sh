set -eo pipefail
git clone --depth 1 https://github.com/langchengg/RL-AgiBot-X2.git "$WS"
cd "$WS"
git checkout --detach 71037f7d5530a65da220345f9037ac4d29f32908
git rev-parse HEAD
git rev-parse origin/main
git status --short
uname -a
cat /etc/os-release
systemd-detect-virt
df -h /tmp
free -h

