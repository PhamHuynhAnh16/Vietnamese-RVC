@echo off
setlocal
title Vietnamese RVC Tensorboard

env\\Scripts\\python.exe main/app/tensorboard.py --open
echo.
pause