#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: ${0##*/} [--dry-run | --commit [message]]"
}

mode=${1:---dry-run}
case "$mode" in
    --help|-h) usage; exit 0 ;;
    --dry-run) if (( $# > 1 )); then usage >&2; exit 2; fi ;;
    --commit)  if (( $# > 2 )); then usage >&2; exit 2; fi ;;
    *) usage >&2; exit 2 ;;
esac

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
files=(
    BasicTensor.cpp BasicTensor.h
    Module.cpp Module.h Parallel.h
    base/ThreadJob.h base/ThreadPool.cpp base/ThreadPool.h
    mnist.cpp tests/AutogradTest.cpp tests/ThreadPoolTest.cpp
    CMakeLists.txt README.md checkin.sh
)

if [[ $mode == --dry-run ]]; then
    git --no-optional-locks status --short --untracked-files=all -- "${files[@]}"
    exit 0
fi

message=${2:-Add multithreaded C++ autograd library and MNIST example}
git add -A -- "${files[@]}"
git commit --only -m "$message" -- "${files[@]}"
