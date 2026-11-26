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
    base/ForkJoinPool.cpp base/ForkJoinPool.h
    mnist.cpp tests/AutogradTest.cpp tests/ForkJoinPoolTest.cpp
    CMakeLists.txt README.md checkin.sh
)
# Replaced by base/ForkJoinPool. Deleted from the repository by the next --commit.
removed=(
    base/ThreadJob.h base/ThreadPool.cpp base/ThreadPool.h tests/ThreadPoolTest.cpp
)
# The removed paths that HEAD still contains. Empty once the deletion is committed.
mapfile -t stale < <(git ls-tree -r --name-only HEAD -- "${removed[@]}")

if [[ $mode == --dry-run ]]; then
    git --no-optional-locks status --short --untracked-files=all -- "${files[@]}" "${stale[@]}"
    exit 0
fi

message=${2:-Replace mutex ThreadPool with lock-free ForkJoinPool and fused dense layer}
git add -A -- "${files[@]}"
if (( ${#stale[@]} )); then git rm -q --ignore-unmatch -- "${stale[@]}"; fi
git commit --only -m "$message" -- "${files[@]}" "${stale[@]}"
