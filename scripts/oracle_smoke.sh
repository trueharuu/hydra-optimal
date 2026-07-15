#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
GRAPH_PATH=${1:-"$ROOT_DIR/graph.bin"}
BINARY=${2:-"$ROOT_DIR/target/release/zxcl-optimal-solver"}
WEIGHTS_PATH="$ROOT_DIR/weights.txt"
VSTAR_PATH="$ROOT_DIR/vstar_l0_f32.bin"

usage() {
    printf 'Usage: %s [GRAPH_PATH [BINARY]]\n' "$0"
}

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! -f "$GRAPH_PATH" ]]; then
    printf 'error: graph not found: %s\n' "$GRAPH_PATH" >&2
    usage >&2
    exit 2
fi

if [[ ! -x "$BINARY" ]]; then
    printf 'error: executable not found: %s\n' "$BINARY" >&2
    printf 'build it first with: cargo build --release --locked\n' >&2
    usage >&2
    exit 2
fi

if [[ ! -f "$WEIGHTS_PATH" ]]; then
    printf 'error: weights not found: %s\n' "$WEIGHTS_PATH" >&2
    exit 2
fi

if [[ ! -f "$VSTAR_PATH" ]]; then
    printf 'error: V* table not found: %s\n' "$VSTAR_PATH" >&2
    exit 2
fi

# Decision mode writes to the current directory. Resolve user-supplied relative paths before that
# test changes directories.
GRAPH_PATH=$(cd -- "$(dirname -- "$GRAPH_PATH")" && pwd)/$(basename -- "$GRAPH_PATH")
BINARY=$(cd -- "$(dirname -- "$BINARY")" && pwd)/$(basename -- "$BINARY")

TEMP_WORKDIR=
cleanup() {
    if [[ -n "$TEMP_WORKDIR" ]]; then
        if [[ -e "$TEMP_WORKDIR/tree_data.js" ]]; then
            unlink -- "$TEMP_WORKDIR/tree_data.js"
        fi
        rmdir -- "$TEMP_WORKDIR" 2>/dev/null || true
    fi
}
trap cleanup EXIT

run_case() {
    local name=$1
    local input=$2
    local expected=$3
    shift 3

    local actual
    if ! actual=$(
        printf '%s\n' "$input" |
            "$BINARY" \
                --graph "$GRAPH_PATH" \
                --weights "$WEIGHTS_PATH" \
                -m 1 \
                -o \
                "$@" \
                2>/dev/null
    ); then
        printf 'not ok - %s (solver exited with an error)\n' "$name" >&2
        return 1
    fi

    if [[ "$actual" != "$expected" ]]; then
        printf 'not ok - %s\n' "$name" >&2
        printf '  expected: %q\n' "$expected" >&2
        printf '  actual:   %q\n' "$actual" >&2
        return 1
    fi

    printf 'ok - %s\n' "$name"
}

run_rejected_case() {
    local name=$1
    shift

    if "$BINARY" \
        --graph "$GRAPH_PATH" \
        --weights "$WEIGHTS_PATH" \
        --vstar "$VSTAR_PATH" \
        "$@" \
        </dev/null \
        >/dev/null \
        2>/dev/null; then
        printf 'not ok - %s (invalid option combination was accepted)\n' "$name" >&2
        return 1
    fi

    printf 'ok - %s\n' "$name"
}

run_decision_case() {
    local expected_result='1'
    local expected_tree
    expected_tree=$'init_hash=0\ndata=[[0],[4103,1],[12351,2],[209151,3],[25424127,4],[528740607,0],[535035135,6],[129384054015,0],[136901295359,1],[274072600575,2],[1099511627775,3]]'

    TEMP_WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/zxcl-optimal-smoke.XXXXXX")

    local actual_result
    if ! actual_result=$(
        cd -- "$TEMP_WORKDIR"
        printf '%s\n' 'IJLOSTZIJLO IJLOSTZ' |
            "$BINARY" \
                --graph "$GRAPH_PATH" \
                --weights "$WEIGHTS_PATH" \
                -m 1 \
                -s 11 \
                -d \
                -o \
                2>/dev/null
    ); then
        printf 'not ok - decision mode (solver exited with an error)\n' >&2
        return 1
    fi

    if [[ "$actual_result" != "$expected_result" ]]; then
        printf 'not ok - decision mode result\n' >&2
        printf '  expected: %q\n' "$expected_result" >&2
        printf '  actual:   %q\n' "$actual_result" >&2
        return 1
    fi

    local tree_path="$TEMP_WORKDIR/tree_data.js"
    if [[ ! -f "$tree_path" ]]; then
        printf 'not ok - decision mode did not create tree_data.js\n' >&2
        return 1
    fi

    local actual_tree
    actual_tree=$(<"$tree_path")
    local actual_size
    actual_size=$(wc -c <"$tree_path")
    actual_size=${actual_size//[[:space:]]/}
    if [[ "$actual_tree" != "$expected_tree" || "$actual_size" != "${#expected_tree}" ]]; then
        printf 'not ok - decision tree serialization\n' >&2
        printf '  expected bytes: %s\n' "${#expected_tree}" >&2
        printf '  actual bytes:   %s\n' "$actual_size" >&2
        return 1
    fi

    unlink -- "$tree_path"
    rmdir -- "$TEMP_WORKDIR"
    TEMP_WORKDIR=
    printf 'ok - decision mode and exact tree serialization\n'
}

run_optimal_case() {
    local expected_result='4350.43798828125'
    local expected_tree
    expected_tree=$'init_hash=274072600575\nobjective="expected_pc"\ndata=[1099511627775,3,4350.43798828125,[[[4350.34326171875]],[[4350.35888671875]],[[4350.36865234375]],[[4350.4248046875]],[[4350.65576171875]],[[4350.2919921875]],[[4350.62255859375]]]]\nsurvival_success=1\nsurvival_total=1'

    TEMP_WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/zxcl-optimal-smoke.XXXXXX")
    local stderr_path="$TEMP_WORKDIR/stderr.txt"

    local actual_result
    if ! actual_result=$(
        cd -- "$TEMP_WORKDIR"
        printf '%s\n' 'OIJLSTZ IJLOSTZ' |
            "$BINARY" \
                --graph "$GRAPH_PATH" \
                --weights "$WEIGHTS_PATH" \
                --vstar "$VSTAR_PATH" \
                -m 1 \
                -d \
                --optimal \
                -f 274072600575 \
                -o \
                2>"$stderr_path"
    ); then
        printf 'not ok - V*-optimal mode (solver exited with an error)\n' >&2
        return 1
    fi

    if [[ "$actual_result" != "$expected_result" ]]; then
        printf 'not ok - V*-optimal result\n' >&2
        printf '  expected: %q\n' "$expected_result" >&2
        printf '  actual:   %q\n' "$actual_result" >&2
        return 1
    fi

    local stderr_text
    stderr_text=$(<"$stderr_path")
    if [[ "$stderr_text" != *'Survival: 1/1'* ]]; then
        printf 'not ok - V*-optimal decision survival probability\n' >&2
        return 1
    fi

    local tree_path="$TEMP_WORKDIR/tree_data.js"
    if [[ ! -f "$tree_path" ]]; then
        printf 'not ok - V*-optimal mode did not create tree_data.js\n' >&2
        return 1
    fi

    local actual_tree
    actual_tree=$(<"$tree_path")
    local actual_size
    actual_size=$(wc -c <"$tree_path")
    actual_size=${actual_size//[[:space:]]/}
    if [[ "$actual_tree" != "$expected_tree" || "$actual_size" != "${#expected_tree}" ]]; then
        printf 'not ok - V*-optimal tree serialization\n' >&2
        printf '  expected bytes: %s\n' "${#expected_tree}" >&2
        printf '  actual bytes:   %s\n' "$actual_size" >&2
        return 1
    fi

    unlink -- "$tree_path"
    unlink -- "$stderr_path"
    rmdir -- "$TEMP_WORKDIR"
    TEMP_WORKDIR=
    printf 'ok - V*-optimal scalar, classic survival metadata, and exact tree serialization\n'
}

run_optimal_score_only_case() {
    local expected_result='4350.43798828125'
    TEMP_WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/zxcl-optimal-score-only.XXXXXX")
    local stderr_path="$TEMP_WORKDIR/stderr.txt"

    local actual_result
    if ! actual_result=$(
        cd -- "$TEMP_WORKDIR"
        printf '%s\n' 'OIJLSTZ IJLOSTZ' |
            "$BINARY" \
                --graph "$GRAPH_PATH" \
                --vstar "$VSTAR_PATH" \
                -m 1 \
                --optimal \
                -f 274072600575 \
                -o \
                2>"$stderr_path"
    ); then
        printf 'not ok - score-only V*-optimal mode (solver exited with an error)\n' >&2
        return 1
    fi
    if [[ "$actual_result" != "$expected_result" ]]; then
        printf 'not ok - score-only V*-optimal result\n' >&2
        return 1
    fi
    local stderr_text
    stderr_text=$(<"$stderr_path")
    if [[ "$stderr_text" != *'Survival: 1/1'* ]]; then
        printf 'not ok - score-only V*-optimal survival probability\n' >&2
        return 1
    fi
    if [[ -e "$TEMP_WORKDIR/tree_data.js" ]]; then
        printf 'not ok - score-only V*-optimal mode wrote tree_data.js\n' >&2
        return 1
    fi

    unlink -- "$stderr_path"
    rmdir -- "$TEMP_WORKDIR"
    TEMP_WORKDIR=
    printf 'ok - score-only V*-optimal value and classic survival probability\n'
}

run_full_optimal_case() {
    local expected_result='4353.563114239728'
    local expected_size='33123419'
    local expected_hash='e9871a1a7dfa2edd48bae73b0f0956408a7061b7f8b62bcf0cf96143a570f1f9'

    TEMP_WORKDIR=$(mktemp -d "${TMPDIR:-/tmp}/zxcl-optimal-full-oracle.XXXXXX")

    local actual_result
    if ! actual_result=$(
        cd -- "$TEMP_WORKDIR"
        printf '%s\n' 'SIZTLOJ IJLOSTZ' |
            "$BINARY" \
                --graph "$GRAPH_PATH" \
                --vstar "$VSTAR_PATH" \
                -m "${ZXCL_FULL_THREADS:-1}" \
                -d \
                --optimal \
                -o \
                2>/dev/null
    ); then
        printf 'not ok - full V*-optimal oracle (solver exited with an error)\n' >&2
        return 1
    fi
    if [[ "$actual_result" != "$expected_result" ]]; then
        printf 'not ok - full V*-optimal root\n' >&2
        printf '  expected: %s\n' "$expected_result" >&2
        printf '  actual:   %s\n' "$actual_result" >&2
        return 1
    fi

    local tree_path="$TEMP_WORKDIR/tree_data.js"
    local actual_size
    actual_size=$(wc -c <"$tree_path")
    actual_size=${actual_size//[[:space:]]/}
    local actual_hash
    if command -v sha256sum >/dev/null 2>&1; then
        read -r actual_hash _ < <(sha256sum "$tree_path")
    elif command -v shasum >/dev/null 2>&1; then
        read -r actual_hash _ < <(shasum -a 256 "$tree_path")
    else
        printf 'not ok - full V*-optimal oracle (no SHA-256 tool found)\n' >&2
        return 1
    fi
    if [[ "$actual_size" != "$expected_size" || "$actual_hash" != "$expected_hash" ]]; then
        printf 'not ok - full V*-optimal tree\n' >&2
        printf '  expected: %s bytes, %s\n' "$expected_size" "$expected_hash" >&2
        printf '  actual:   %s bytes, %s\n' "$actual_size" "$actual_hash" >&2
        return 1
    fi

    unlink -- "$tree_path"
    rmdir -- "$TEMP_WORKDIR"
    TEMP_WORKDIR=
    printf 'ok - full empty-field V*-optimal root and 33 MB tree oracle\n'
}

run_case \
    'normal, inferred bag, and runtime see command' \
    $'IJLOSTZ IJLOSTZ\nOTJLISO 1\n-s 11\nIJLOSTZIJLO IJLOSTZ' \
    $'838\n206\n1'

run_case \
    'boolean mode' \
    'IJLOSTZ IJLOSTZ' \
    '0' \
    -b

run_case \
    'weighted mode' \
    'IJLOSTZ IJLOSTZ' \
    '8589934592' \
    -w

run_case \
    'two-line terminal field' \
    'IJLOSTZ IJLOSTZ' \
    '1' \
    -t -f 1048575

run_decision_case

run_rejected_case \
    'optimal mode requires see 7' \
    --optimal -s 6

run_rejected_case \
    'optimal mode rejects weighted mode' \
    --optimal -w

run_rejected_case \
    'optimal mode rejects boolean mode' \
    --optimal -b

run_rejected_case \
    'optimal mode rejects explicit two-line mode' \
    --optimal -t

run_optimal_score_only_case
run_optimal_case

if [[ ${ZXCL_FULL_ORACLE:-0} == 1 ]]; then
    run_full_optimal_case
else
    printf 'skip - full empty-field V* oracle (set ZXCL_FULL_ORACLE=1)\n'
fi

if command -v node >/dev/null 2>&1; then
    node "$ROOT_DIR/scripts/viewer_smoke.js" "$ROOT_DIR/main.js"
    printf 'ok - zero-step optimal terminal renders in the viewer\n'
else
    printf 'skip - viewer smoke (node not found)\n'
fi

printf 'all oracle smoke cases passed\n'
