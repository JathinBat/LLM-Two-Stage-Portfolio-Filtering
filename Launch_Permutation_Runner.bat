@echo off
setlocal
cd /d "%~dp0"

set "LOCAL_PY=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"

echo Starting Investment Strategy Permutation Runner...

if exist "%LOCAL_PY%" (
    "%LOCAL_PY%" permutation_runner.py
) else (
    python permutation_runner.py
)

echo.
echo Permutation runner exited.
pause
