set -eo pipefail
apt-cache policy python3 python3-venv python3-pip python3-setuptools python3-colcon-common-extensions python3-numpy python3-pil python3-matplotlib libosmesa6 ros-jazzy-rclpy ros-jazzy-rmw-fastrtps-cpp
# python3-cffi was an inventory query, not a required apt package: requirements installs cffi in the venv.
dpkg-query -W -f='${binary:Package} ${Version} ${Architecture} ${db:Status-Status}\n' python3-cffi-backend python3-colcon-core python3-colcon-python-setup-py python3-colcon-ros ros-jazzy-ament-package ros-jazzy-ament-index-python ros-jazzy-rcl-interfaces

