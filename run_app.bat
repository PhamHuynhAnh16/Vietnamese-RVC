@echo off
chcp 65001 > nul

setlocal
title Vietnamese RVC

if exist "runtime\python.exe" (
    runtime\python.exe main\app\app.py --open --allow_all_disk
) else if exist "env\\Scripts\\python.exe" (
    env\Scripts\python.exe main\app\app.py --open --allow_all_disk
)

echo.
pause