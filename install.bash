#!/bin/bash
# Installs the `fiftyone-brain` package and its dependencies.
#
# Usage:
#   bash install.bash
#
# Copyright 2017-2023, Voxel51, Inc.
# voxel51.com
#

# Show usage information
set -e
usage() {
    echo "Usage:  bash $0 [-h] [-d]

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
        *) usage ;;
    esac
done
[ ${SHOW_HELP} = true ] && usage && exit 0
OS=$(uname -s)

echo "***** INSTALLING FIFTYONE-BRAIN *****"
if [ ${DEV_INSTALL} = true ]; then
    echo "Performing dev install"
    pip install -r requirements/dev.txt
    pre-commit install
else
    pip install -r requirements.txt
fi
pip install -e .

echo "***** INSTALLATION COMPLETE *****"
