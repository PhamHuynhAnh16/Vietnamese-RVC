@echo off
chcp 65001 > nul

setlocal
title Vietnamese RVC Install

set ZIP_PATH=%~dp0assets\runtime.zip
set DEST_PATH=%~dp0


if exist "%ZIP_PATH%" (
    echo Giải nén runtime...
    powershell -command "Expand-Archive -Path '%ZIP_PATH%' -DestinationPath '%DEST_PATH%' -Force"
    echo Giải nén runtime hoàn tất!
) else (
    echo Không tìm thấy runtime!
    pause
)

if exist "runtime\python.exe" (
    echo Bắt đầu cài đặt gói pip...
    runtime\python.exe runtime\get-pip.py

    cls
    echo Hoàn tất cài gói pip, bắt đầu cài thư viện...

    runtime\python.exe main\app\install.py

    cls
    echo Hoàn tất!
) else (
    echo Không tìm thấy phần runtime đã giải nén!
    pause
)

echo Bắt đầu khởi động giao diện!
call run_app.bat
pause