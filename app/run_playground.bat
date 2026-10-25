@echo off
cd /d "%~dp0.."
if /i "%~1"=="memory" (
    python app\memory_engine.py %2 %3 %4 %5 %6 %7 %8 %9
) else (
    python app\parallel_playground.py
)
if errorlevel 1 pause
