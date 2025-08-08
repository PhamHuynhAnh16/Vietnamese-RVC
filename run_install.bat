@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

set GPU_NVIDIA=0
set GPU_AMD=0

set FILES=requirements.txt run_app.bat tensorboard.bat docker-compose-amd.yaml docker-compose-cpu.yaml docker-compose-cuda118.yaml docker-compose-cuda128.yaml Dockerfile Dockerfile.amd Dockerfile.cuda118 Dockerfile.cuda128
set BASE_URL=https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main
set FFMPEG=%BASE_URL%/ffmpeg/ffmpeg.exe
set FFPROBE=%BASE_URL%/ffmpeg/ffprobe.exe
set DIR=%~dp0

for /f "tokens=*" %%i in ('wmic path win32_VideoController get Name') do (
    echo %%i | find /i "GTX" >nul && set GPU_NVIDIA=1
    echo %%i | find /i "RTX" >nul && set GPU_NVIDIA=1
    echo %%i | find /i "NVIDIA" >nul && set GPU_NVIDIA=1
    echo %%i | find /i "Quadro" >nul && set GPU_NVIDIA=1
    echo %%i | find /i "GeForce" >nul && set GPU_NVIDIA=1
    echo %%i | find /i "RX" >nul && set GPU_AMD=1
    echo %%i | find /i "AMD" >nul && set GPU_AMD=1
    echo %%i | find /i "Vega" >nul && set GPU_AMD=1
    echo %%i | find /i "Radeon" >nul && set GPU_AMD=1
    echo %%i | find /i "FirePro" >nul && set GPU_AMD=1
)

if %GPU_NVIDIA%==1 if %GPU_AMD%==1 (
    echo Phát hiện cả NVIDIA và AMD. Sẽ ưu tiên GPU của Nvidia.
    set GPU_NVIDIA=1
    set GPU_AMD=0
)

if %GPU_NVIDIA%==0 if %GPU_AMD%==0 (
    echo Không phát hiện GPU. Sẽ sử dụng môi trường chạy của CPU.
    set GPU_TYPE=CPU
)

if %GPU_NVIDIA%==1 (
    echo Sẽ sử dụng môi trường chạy của Nvidia.
    set GPU_TYPE=NVIDIA
)

if %GPU_AMD%==1 (
    echo Sẽ sử dụng môi trường chạy của AMD.
    set GPU_TYPE=AMD
)

echo GPU được chọn: %GPU_TYPE%

if not exist "runtime" (
    cd /d "%~dp0"
    echo Bắt đầu xóa các tệp cũ và không cần thiết...

    for %%f in (%FILES%) do (
        if exist "%%f" (
            del /f /q "%%f"
            echo Xóa tệp %%f...
        )
    )

    echo Hoàn tất xóa!
    echo Bắt đầu cài đặt ffmpeg...

    if not exist ffmpeg.exe (
        curl.exe -L "%FFMPEG%" -o ffmpeg.exe
        if not exist ffmpeg.exe (
            echo Đã xảy ra lỗi khi cài tải xuống ffmpeg...
            pause
            exit /b 1
        )
    )

    if not exist ffprobe.exe (
        curl.exe -L "%FFPROBE%" -o ffprobe.exe

        if not exist ffprobe.exe (
            echo Đã xảy ra lỗi khi cài tải xuống ffprobe...
            pause
            exit /b 1
        )
    )
    
    echo Đã hoàn thành cài dặt ffmpeg. Tiến hành cài đặt môi trường chạy...

    if "%GPU_TYPE%"=="NVIDIA" (
        set RUNTIME_ZIP=runtime-nvidia.zip
    ) else if "%GPU_TYPE%"=="AMD" (
        set RUNTIME_ZIP=runtime-amd.zip
    ) else (
        set RUNTIME_ZIP=runtime-cpu.zip
    )

    set DOWNLOAD_URL=!BASE_URL!/runtime/!RUNTIME_ZIP!
    echo Đường dẫn liên kết: !DOWNLOAD_URL!

    if not exist runtime.zip (
        curl.exe -L "!DOWNLOAD_URL!" -o runtime.zip

        if not exist runtime.zip (
            echo Đã xảy ra lỗi khi cài tải xuống môi trường chạy...
            pause
            exit /b 1
        )
    )

    echo Bắt đầu giải nén tệp runtime.zip...
    tar -xf runtime.zip
    del runtime.zip

    echo Hoàn tất giải nén. Tiến hành tạo lại tệp khởi chạy...

    echo @echo off > run_app.bat
    echo setlocal >> run_app.bat
    echo title Vietnamese RVC By Anh >> run_app.bat
    echo. >> run_app.bat
    echo set "scriptDir=%%~dp0" >> run_app.bat
    echo set "runtimeFolder=%%scriptDir%%runtime" >> run_app.bat
    echo. >> run_app.bat
    echo runtime\\python.exe main\\app\\app.py --open --allow_all_disk >> run_app.bat
    echo. >> run_app.bat
    echo pause >> run_app.bat

    echo @echo off > tensorboard.bat
    echo setlocal >> tensorboard.bat
    echo title Vietnamese RVC Tensorboard >> tensorboard.bat
    echo. >> tensorboard.bat
    echo set "scriptDir=%%~dp0" >> tensorboard.bat
    echo set "runtimeFolder=%%scriptDir%%runtime" >> tensorboard.bat
    echo. >> tensorboard.bat
    echo runtime\\python.exe main/app/run_tensorboard.py --open >> tensorboard.bat
    echo. >> tensorboard.bat
    echo pause >> tensorboard.bat

    "%DIR%runtime\python.exe" -m pip install six packaging python-dateutil platformdirs pywin32 onnxconverter_common
    echo Đã cài đặt hoàn tất. Bạn có thể tiến hành sử dụng!
) else (
    echo Đã có thư mục môi trường chạy không cần cài đặt lại.
)

pause