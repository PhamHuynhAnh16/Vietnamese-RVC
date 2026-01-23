@echo off
setlocal
title Vietnamese RVC Tensorboard

if exist "runtime\\python.exe" (
    runtime\python.exe main\app\run_tensorboard.py --open
) else if exist "env\\Scripts\\python.exe" (
    env\Scripts\python.exe main\app\run_tensorboard.py --open
)

echo.
pause