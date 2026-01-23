@echo off
setlocal
title Vietnamese RVC By Anh

if exist "runtime\python.exe" (
    runtime\python.exe main\app\app.py --open --allow_all_disk
) else if exist "env\\Scripts\\python.exe" (
    env\Scripts\python.exe main\app\app.py --open --allow_all_disk
)

echo.
pause