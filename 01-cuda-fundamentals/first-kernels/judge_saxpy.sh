#!/usr/bin/env bash
# Judge for saxpy.cu, with its own independent reference check.
# Usage: bash first-kernels/judge_saxpy.sh first-kernels/saxpy.cu   (topic root)
#        bash judge_saxpy.sh                                        (defaults to
#        the sibling saxpy.cu; or pass any path to a saxpy source file)
# Env vars: ARCH=sm_80 sets the target arch (default native, i.e. this machine's
#           GPU; set it explicitly when compiling on a GPU-less node)
#           WMHPC_RESULT=1 additionally emits a machine-readable ##RESULT line
# Builds the given source, then runs 7 cases (n = 0 1 31 1024 1025 1048576
# 1048579). Each case checks the program's contract: ./saxpy <n> generates x, y
# from the fixed formula, computes y = 2x + y on the GPU, prints one line
# SUM=<sum of all y[i]>, exit code 0. The expected SUM is computed independently
# on the CPU with the same formula -- all formula values are small integers or
# half-integers, exactly representable in float, so integer comparison suffices.
# Prints one PASS line per case and "all passed" at the end; exit code 0 only
# when every case passes.
set -u
SRC="${1:-$(dirname "$0")/saxpy.cu}"
ARCH="${ARCH:-native}"
BIN="$(mktemp -u /tmp/saxpy.XXXXXX)"

emit_result() {  # $1=status $2=metrics_json
    [[ "${WMHPC_RESULT:-0}" == "0" || -z "${WMHPC_RESULT:-}" ]] && return 0
    local dev
    dev="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
    printf '##RESULT {"name":"saxpy","status":"%s","metrics":%s,"device":"%s"}\n' \
        "$1" "$2" "${dev:-unknown}"
}

if [[ ! -f "$SRC" ]]; then
    echo "file not found: $SRC"
    emit_result "not_attempted" '{}'
    exit 1
fi

if ! nvcc -O2 -std=c++20 -arch="$ARCH" -o "$BIN" "$SRC"; then
    echo "compile failed"
    emit_result "compile_error" '{}'
    exit 1
fi

expected_sum() {
    awk -v n="$1" 'BEGIN {
        s = 0;
        for (i = 0; i < n; i++) {
            x = (i % 2048 - 1024) * 0.5;
            y = i % 1024 - 512;
            s += 2 * x + y;
        }
        printf "%.0f", s;
    }'
}

fail=0
npass=0
ntotal=0
for n in 0 1 31 1024 1025 1048576 1048579; do
    ntotal=$((ntotal + 1))
    out="$("$BIN" "$n" 2>&1)"
    rc=$?
    got="$(printf '%s\n' "$out" | grep -o 'SUM=[-0-9]*' | tail -1 | cut -d= -f2)"
    want="$(expected_sum "$n")"
    if [[ $rc -eq 0 && -n "$got" && "$got" == "$want" ]]; then
        echo "n=$n  PASS  (SUM=$got)"
        npass=$((npass + 1))
    else
        echo "n=$n  FAIL  (expected SUM=$want, exit code $rc, your output below)"
        printf '%s\n' "$out" | head -5
        fail=1
    fi
done
rm -f "$BIN"

metrics="$(printf '{"cases_total":%d,"cases_passed":%d}' "$ntotal" "$npass")"
if [[ $fail -eq 0 ]]; then
    echo "all passed"
    emit_result "pass" "$metrics"
else
    emit_result "fail" "$metrics"
    exit 1
fi
