set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd "$WS"
git apply --check "$ACCEPT_ROOT/evidence/runner-evidence.patch"
git apply "$ACCEPT_ROOT/evidence/runner-evidence.patch"
sha256sum src/x2_recovery/test/test_ros_integration.py "$ACCEPT_ROOT/evidence/runner-evidence.patch"
git diff --check
git diff --stat
git diff -- README.md src/x2_recovery/test/test_ros_integration.py > "$ACCEPT_ROOT/evidence/tested.patch"
sha256sum "$ACCEPT_ROOT/evidence/tested.patch"

