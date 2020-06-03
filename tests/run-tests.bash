#!/usr/bin/env bash

set -ex
cd "$(dirname "$0")"/..

find tests/ -name '*.py' -print0 | xargs -0 --max-args 1 --verbose python
