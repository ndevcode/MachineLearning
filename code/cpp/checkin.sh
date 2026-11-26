#!/usr/bin/env bash
# Copyright 2026 N Dev (ndevcode)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
