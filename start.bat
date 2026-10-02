@echo off
setlocal
cd /d %~dp0
if not defined OPENBLAS_NUM_THREADS set OPENBLAS_NUM_THREADS=1
if not defined OMP_NUM_THREADS set OMP_NUM_THREADS=1
if not defined VECTORRF_PORT set VECTORRF_PORT=8765
if not exist .venv\Scripts\python.exe (
  py -3 -m venv .venv
)
call .venv\Scripts\activate.bat
python -c "import vectorrf" >nul 2>nul
if errorlevel 1 python -m pip install -e .
echo VectorRF Lab: http://127.0.0.1:%VECTORRF_PORT%
python -m uvicorn vectorrf.app:app --app-dir backend --host 127.0.0.1 --port %VECTORRF_PORT%
