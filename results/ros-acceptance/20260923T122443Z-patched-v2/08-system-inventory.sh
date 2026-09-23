set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
uname -a
cat /etc/os-release
systemd-detect-virt
df -h /tmp
free -h
dpkg-query -W -f='${binary:Package} ${Version} ${Architecture} ${db:Status-Status}\n' git ca-certificates python3 python3-minimal python3-venv python3-pip python3-setuptools python3-colcon-common-extensions python3-colcon-core python3-colcon-python-setup-py python3-colcon-ros python3-colcon-package-information python3-numpy python3-pil python3-matplotlib python3-cffi-backend python3-nacl libosmesa6 ros-jazzy-ros-base ros-jazzy-rclpy ros-jazzy-sensor-msgs ros-jazzy-std-msgs ros-jazzy-std-srvs ros-jazzy-launch ros-jazzy-launch-ros ros-jazzy-rmw-fastrtps-cpp ros-jazzy-ament-index-python ros-jazzy-ament-package ros-jazzy-rcl-interfaces
apt-cache policy python3 python3-numpy python3-pil python3-matplotlib libosmesa6 python3-colcon-common-extensions ros-jazzy-rclpy ros-jazzy-rmw-fastrtps-cpp

