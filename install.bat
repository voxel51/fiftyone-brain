@echo off
:: Installs the `fiftyone-brain` package and its dependencies.
::
:: Usage:
:: .\install.bat
::
:: Copyright 2017-2024, Voxel51, Inc.
:: voxel51.com
::
:: Commands:
:: -h      Display help message
:: -d      Install developer dependencies.

set SHOW_HELP=false
set DEV_INSTALL=false

:parse
IF "%~1"=="" GOTO endparse
IF "%~1"=="-h" GOTO helpmessage
IF "%~1"=="-d" set DEV_INSTALL=true
SHIFT
GOTO parse
:endparse

echo ***** INSTALLING FIFTYONE-BRAIN *****
IF %DEV_INSTALL%==true (
  echo Performing dev install
  pip install -r requirements/dev.txt
  pre-commit install
  pip install -e .
) else (
  pip install -r requirements.txt
  pip install .
)

echo ***** INSTALLATION COMPLETE *****
exit /b

:helpmessage
echo Additional Arguments:
echo -h      Display help message
echo -d      Install developer dependencies.
exit /b