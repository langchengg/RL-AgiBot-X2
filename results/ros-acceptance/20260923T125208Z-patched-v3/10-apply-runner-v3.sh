set -eo pipefail
PS4='+ ${EPOCHREALTIME} cwd=${PWD} '; set -x
cd "$WS"
git apply --check "$ACCEPT_ROOT/evidence/runner-evidence-v3.patch"
git apply "$ACCEPT_ROOT/evidence/runner-evidence-v3.patch"
git diff --check
git diff -- README.md src/x2_recovery/test/test_ros_integration.py > "$ACCEPT_ROOT/evidence/tested-v3.patch"
sha256sum README.md src/x2_recovery/test/test_ros_integration.py "$ACCEPT_ROOT/evidence/tested-v3.patch"
git status --short

