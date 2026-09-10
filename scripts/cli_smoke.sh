#!/usr/bin/env bash
# End-to-end CLI smoke test.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MY_WEIGHT="${REPO_ROOT}/my_weight"
[[ -d "${MY_WEIGHT}" ]] || { echo "no checkpoints under ${MY_WEIGHT}" >&2; exit 1; }

_pick_python() {
    local candidates=()
    [[ -n "${PYTHON:-}" ]] && candidates+=("${PYTHON}")
    candidates+=("${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/../vllm/.venv/bin/python" "python3")
    for c in "${candidates[@]}"; do
        [[ -x "$(command -v "$c" 2>/dev/null)" || -x "$c" ]] || continue
        "$c" -c 'import torch,sys;sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null && { echo "$c"; return 0; }
    done
    return 1
}

PY="$(_pick_python)" || { echo "ERROR: no Python with torch.cuda" >&2; exit 1; }
echo "using torch=$("${PY}" -c 'import torch;print(torch.__version__)') driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1||echo ?)"

_is_mm() { "${PY}" -c "import json,sys;c=json.load(open('$1/config.json'));sys.exit(0 if c.get('model_type') in ('llava','qwen3_vl') else 1)"; }

_smoke_one() {
    local name="$1" md="${MY_WEIGHT}/$1" log err
    [[ -f "${md}/config.json" ]] && compgen -G "${md}/*.safetensors" >/dev/null || { echo "  [skip] $name"; return; }
    log=$(mktemp); err=$(mktemp)
    local ok=1
    if _is_mm "${md}"; then
        local img
        for c in "${REPO_ROOT}/images/llava_test/"*.{jpeg,jpg,png}; do [[ -f "$c" ]] && { img="$c"; break; }; done
        [[ -n "${img}" ]] || { echo "  [skip] $name: no test image"; rm -f "$log" "$err"; return; }
        echo "  [vl  ] $name"
        "${PY}" -m rapid_llm.cli vl-chat --model-dir "${md}" --image "${img}" \
            --prompt 'Describe the animal in one sentence.' \
            --temperature 0.0 --top-p 1.0 --max-gen-len 16 --max-seq-len 2048 >"${log}" 2>"${err}" || ok=0
    else
        echo "  [chat] $name"
        RAPID_LLM_MODEL_DIR="${md}" "${PY}" -m rapid_llm.cli chat \
            --temperature 0.0 --top-p 1.0 --max-gen-len 16 --max-seq-len 512 \
            <<< $'The capital of France is\nexit\n' >"${log}" 2>"${err}" || ok=0
    fi
    if [[ $ok -ne 1 ]]; then echo "    FAIL"; sed 's/^/      /' "${err}" >&2; rm -f "$log" "$err"; return 1; fi
    local gen; gen=$(grep -v '^Loaded' "${log}" | tr -d '\r' | tr -s '[:space:]' ' ' | sed 's/^ *//;s/ *$//;s/^>>> *//')
    [[ -n "$gen" ]] || { echo "    FAIL: no output"; tail -5 "$err" >&2; rm -f "$log" "$err"; return 1; }
    [[ "$gen" != *$'\ufffd'* ]] || { echo "    FAIL: garbled"; rm -f "$log" "$err"; return 1; }
    echo "    ok: ${gen:0:90}"
    rm -f "$log" "$err"
}

names=("$@"); [[ ${#names[@]} -eq 0 ]] && mapfile -t names < <(find "${MY_WEIGHT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
failed=0
for n in "${names[@]}"; do _smoke_one "$n" || ((failed++)); done
[[ $failed -eq 0 ]] || { echo "FAIL: ${failed} checkpoint(s)" >&2; exit 1; }
echo "OK: all checkpoints passed"