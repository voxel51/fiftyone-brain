#!/bin/sh
# Installs the `fiftyone-brain` package and its dependencies.
#
# Usage:
#   sh install.sh
#
# Copyright 2017-2026, Voxel51, Inc.
# voxel51.com
#

# Show usage information
usage() {
    echo "Usage:  sh $0 [-h] [-d]

Getting help:
-h      Display this help message.

Custom installations:
-d      Install developer dependencies.
"
}

# Parse flags
SHOW_HELP=false
DEV_INSTALL=false
while getopts "hd" FLAG; do
    case "${FLAG}" in
        h) SHOW_HELP=true ;;
        d) DEV_INSTALL=true ;;
        *)
            usage
            exit 1
            ;;
    esac
done
[ "$SHOW_HELP" = true ] && usage && exit 0

set -e

if command -v python >/dev/null 2>&1; then
    PYTHON=python
elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
else
    echo "ERROR: Neither python3 nor python found in PATH."
    exit 1
fi

# Ensure pip targets this Python interpreter
if command -v uv >/dev/null 2>&1; then
    PIP="uv pip"
else
    PIP="$PYTHON -m pip"
fi

echo "***** INSTALLING FIFTYONE-BRAIN *****"
if [ "$DEV_INSTALL" = true ]; then
    echo "Performing dev install"
    # Compat mode puts the source tree on sys.path, which lets static tools
    # like pylint resolve `fiftyone.brain` alongside the core `fiftyone`
    $PIP install -e ".[dev]" --config-settings editable_mode=compat
    pre-commit install
else
    $PIP install .
fi

echo "***** INSTALLATION COMPLETE *****"
