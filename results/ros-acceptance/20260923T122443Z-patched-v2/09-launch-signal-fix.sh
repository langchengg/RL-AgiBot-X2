set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd "$WS"
git apply --check "$ACCEPT_ROOT/evidence/runner-evidence-v1-to-v2.patch"
git apply "$ACCEPT_ROOT/evidence/runner-evidence-v1-to-v2.patch"
sha256sum src/x2_recovery/test/test_ros_integration.py
git diff --check
git diff -- README.md src/x2_recovery/test/test_ros_integration.py > "$ACCEPT_ROOT/evidence/tested-v2.patch"
sha256sum "$ACCEPT_ROOT/evidence/tested-v2.patch"

