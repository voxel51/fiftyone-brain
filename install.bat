@echo off
:: Installs the `fiftyone-brain` package and its dependencies.
::
:: Usage:
:: .\install.bat
::
:: Copyright 2017-2026, Voxel51, Inc.
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

:: Resolve pip backend
where uv >nul 2>&1
IF NOT ERRORLEVEL 1 (
  set PIP=uv pip
) else (
  where pip >nul 2>&1
  IF ERRORLEVEL 1 (
    echo ERROR: Neither 'uv' nor 'pip' found in PATH. Please install one before running this script.
    exit /b 1
  )
  set PIP=pip
)

echo ***** INSTALLING FIFTYONE-BRAIN *****
IF %DEV_INSTALL%==true (
  echo Performing dev install
  REM Compat mode puts the source tree on sys.path, which lets static tools
  REM like pylint resolve `fiftyone.brain` alongside the core `fiftyone`
  %PIP% install -e .[dev] --config-settings editable_mode=compat
  pre-commit install
) else (
  %PIP% install .
)

echo ***** INSTALLATION COMPLETE *****
exit /b

:helpmessage
echo Additional Arguments:
echo -h      Display help message
echo -d      Install developer dependencies.
exit /b