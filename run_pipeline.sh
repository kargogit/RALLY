#!/bin/bash
set -euo pipefail

# ============================
# Usage: ./run_pipeline.sh <input.asm>
# Example: ./run_pipeline.sh asm/golden.asm
# ============================

if [ $# -eq 0 ]; then
    echo "❌ Error: No ASM file provided!"
    echo "Usage: $0 <path/to/yourfile.asm>"
    exit 1
fi

ASM_FILE="$1"
BASE=$(basename "$ASM_FILE" .asm)   # e.g. "golden" from "asm/golden.asm"

echo "=== ASM → Protobuf pipeline ==="
python3 asmParser.py "$ASM_FILE" | \
    python3 extractAST.py | \
    python3 partitionAST.py | \
    python3 cfgAnnotator.py | \
    python3 cfgRefiner.py | \
    python3 symbolMapper.py | \
    python3 abiRecovery.py | \
    python3 typeInference.py | \
    python3 interprocAnalysis.py > "${BASE}_9.pb"

echo "=== LLVM lowering pipeline ==="

run_stage() {
    local tool="$1"
    shift
    local args=("$@")

    echo "→ Running $tool..."

    # Normal run
    "./$tool" "${args[@]}"

    # --print run (auto-generates .ll + .json)
    local len=${#args[@]}
    local print_args=("${args[@]:0:len-2}")
    local bc="${args[len-2]}"
    local pb="${args[len-1]}"

    "./$tool" --print "${print_args[@]}" "${bc%.bc}.ll" "${pb%.pb}.json"
}

# One clean line per stage — filenames derived automatically from your ASM file
run_stage llvmInit         "${BASE}_9.pb"           "${BASE}_10.bc" "${BASE}_10.pb"
run_stage llvmLowerInteger "${BASE}_10.bc" "${BASE}_10.pb" "${BASE}_11.bc" "${BASE}_11.pb"
run_stage llvmLowerFpAtomic "${BASE}_11.bc" "${BASE}_11.pb" "${BASE}_12.bc" "${BASE}_12.pb"
run_stage llvmLowerControlFlow "${BASE}_12.bc" "${BASE}_12.pb" "${BASE}_13.bc" "${BASE}_13.pb"
run_stage llvmLowerVectorSSE "${BASE}_13.bc" "${BASE}_13.pb" "${BASE}_14.bc" "${BASE}_14.pb"

echo "   Final output: ${BASE}_14.pb"
echo "   All debug files (.ll + .json) generated automatically"
