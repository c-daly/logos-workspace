#!/usr/bin/env bash
# Unit tests for bootstrap.sh pure logic. Sources the script (main is guarded).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/." || exit 1
cd ..
# shellcheck disable=SC1091
source ./bootstrap.sh

fail=0
assert_eq() { # $1=desc $2=expected $3=actual
  if [ "$2" = "$3" ]; then printf 'ok   - %s\n' "$1"
  else printf 'FAIL - %s\n      expected: %q\n      actual:   %q\n' "$1" "$2" "$3"; fail=1; fi
}

WITH_ML=false
assert_eq "apollo extras"        "--no-interaction -E otel" "$(repo_install_args apollo)"
assert_eq "sophia extras"        "--no-interaction -E otel" "$(repo_install_args sophia)"
assert_eq "hermes extras"        "--no-interaction -E otel" "$(repo_install_args hermes)"
assert_eq "logos no extras"      "--no-interaction"         "$(repo_install_args logos)"
assert_eq "talos no extras"      "--no-interaction"         "$(repo_install_args talos)"

# shellcheck disable=SC2034  # WITH_ML is read by repo_install_args in the sourced script
WITH_ML=true
assert_eq "sophia ml is a group" "--no-interaction -E otel --with ml" "$(repo_install_args sophia)"
assert_eq "hermes ml is an extra" "--no-interaction -E otel -E ml"    "$(repo_install_args hermes)"

exit "$fail"
