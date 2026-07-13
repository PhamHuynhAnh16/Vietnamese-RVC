#!/bin/bash
clear
echo "Vietnamese RVC Install (Linux)"

echo "Đang cài đặt Python 3.12 và thư viện hệ thống..."
sudo apt update
sudo apt install -y python3.12 python3.12-venv

echo "Khởi tạo môi trường ảo..."
python3.12 -m venv venv
source "./venv/bin/activate"

echo "Bắt đầu cài đặt thư viện..."
python "./main/app/install.py"

echo "Hoàn tất! Bắt đầu khởi động giao diện..."
bash "./run_app.sh"
