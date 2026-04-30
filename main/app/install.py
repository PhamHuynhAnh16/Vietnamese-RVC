import os
import sys
import subprocess

sys.path.append(os.getcwd())

os.makedirs(os.path.join("assets", ".uv"), exist_ok=True)
os.environ["UV_CACHE_DIR"] = os.path.join("assets", ".uv")

python_dir = os.path.join("runtime", "python.exe")

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi thực thi lệnh: {e}")

def install_dependencies():
    print("Cài đặt các thư viện cơ bản...")

    run_command(f"{python_dir} -m pip install uv")
    run_command(f"{python_dir} -m uv pip install six packaging python-dateutil platformdirs pywin32 wget")

def install_ffmpeg():
    if not os.path.exists("ffmpeg.exe"):
        print("Cài đặt ffmpeg...")
        run_command('curl -L -o ffmpeg.exe https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/ffmpeg/ffmpeg.exe')

    if not os.path.exists("ffprobe.exe"):
        print("Cài đặt ffprobe...")
        run_command('curl -L -o ffprobe.exe https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/ffmpeg/ffprobe.exe')

def remove_onnxruntime():
    if not os.path.exists("requirements.txt"): return

    with open("requirements.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.writelines(l for l in lines if "onnxruntime" not in l)

def install_cpu():
    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện cpu...")

    run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu")
    run_command(f"{python_dir} -m uv pip install onnxruntime")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_cuda(version = "cu118"):
    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện cuda...")

    if version in ("cu118", "cu121"): run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")
    run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/{version}")
    run_command(f"{python_dir} -m uv pip install onnxruntime-gpu" + ("==1.20.1" if version in ("cu118", "cu121") else ""))
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")
    if version in ("cu128", "cu130"): run_command(f"{python_dir} -m uv pip install faiss-cpu==1.13.2")

def install_directml():
    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện directml...")

    run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")
    run_command(f"{python_dir} -m uv pip install torch==2.4.1 torchaudio==2.4.1 torchvision")
    run_command(f"{python_dir} -m uv pip install torch-directml==0.2.5.dev240914")
    run_command(f"{python_dir} -m uv pip install onnxruntime-directml")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_opencl():
    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện opencl...")

    run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")
    run_command(f"{python_dir} -m uv pip install torch==2.6.0 torchaudio==2.6.0 torchvision")
    run_command(f"{python_dir} -m uv pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp311-none-win_amd64.whl")
    run_command(f"{python_dir} -m uv pip install onnxruntime-directml")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_xpu(provider = "openvino"):
    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện xpu...")

    run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu")
    run_command(f"{python_dir} -m uv pip install openvino==2025.4.1")
    run_command(f"{python_dir} -m uv pip install triton-windows")
    run_command(f"{python_dir} -m uv pip install " + ("onnxruntime-openvino==1.24.1" if provider == "openvino" else "onnxruntime-directml"))
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")
    run_command(f"{python_dir} -m uv pip install faiss-cpu==1.13.2")

def install_requirements():
    print("Các phiên bản Backend: \n - 1. CPU (Đối với CPU) / 2. CUDA (Đối với GPU NVIDIA) / 3. DIRECTML (Đối với GPU NVIDIA / AMD / INTEL) / 4. OPENCL (Đối với GPU AMD) / 5. XPU (Đối với GPU INTEL)")

    while 1:
        selected = str(input("Hãy chọn phiên bản Backend (Nhập 1-5): ")).strip().upper()

        if selected in ("1", "CPU"):
            print("Cài đặt dành cho CPU...")
            install_cpu()
            break
        elif selected in ("2", "CUDA"):
            print("Cài đặt dành cho CUDA...")
            print("Các phiên bản CUDA: \n Lưu ý: Hãy chọn đúng phiên bản nếu như chọn sai có thể gây ra lỗi suy luận hoặc huấn luyện. \n - 1. Phiên bản 118 dành cho các GPU GTX 10-Series trở lên. \n - 2. Phiên bản 121 dành cho các GPU RTX 20-Series đến 30-Series trở lên. \n - 3. Phiên bản 128 dành cho các GPU RTX 30-Series đến RTX 40-Series trở lên. \n - 4. Phiên bản 130 dành cho các GPU RTX 50-Series trở lên.")

            cuda_selected = None

            while 1:
                cuda_selected = str(input("Hãy chọn phiên bản CUDA (Nhập 1-4): ")).strip().upper()
                
                if cuda_selected in ("1", "118"):
                    install_cuda("cu118")
                    break
                elif cuda_selected in ("2", "121"):
                    install_cuda("cu121")
                    break
                elif cuda_selected in ("3", "128"):
                    install_cuda("cu128")
                    break
                elif cuda_selected in ("4", "130"):
                    install_cuda("cu130")
                    break
                else:
                    print("Vui lòng chọn phiên bản CUDA hợp lệ!")
            
            if cuda_selected in ("cu128", "cu130"):
                print("Cài đặt gói phụ thuộc TensorRT (Cực Kỳ Không Khuyên Dùng)")

                while 1:
                    tensorrt_selected = str(input("Bạn có muốn sử dụng TensorRT không (Y/N): ")).strip().upper()

                    if tensorrt_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install tensorrt")
                        break
                    elif tensorrt_selected in ("N", "NO"):
                        break
                    else:
                        print("Vui lòng chọn giá trị hợp lệ!")
                
                print("Cài đặt gói phụ thuộc Triton (Không Khuyên Dùng)")

                while 1:
                    triton_selected = str(input("Bạn có muốn sử dụng Triton không (Y/N): ")).strip().upper()

                    if triton_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install triton-windows")
                        break
                    elif triton_selected in ("N", "NO"):
                        break
                    else:
                        print("Vui lòng chọn giá trị hợp lệ!")

            break
        elif selected in ("3", "DIRECTML"):
            print("Cài đặt dành cho DIRECTML...")
            install_directml()
            break
        elif selected in ("4", "OPENCL"):
            print("Cài đặt dành cho OPENCL...")
            install_opencl()
            break
        elif selected in ("5", "XPU"):
            print("Cài đặt dành cho XPU...")
            print("Phiên bản ONNXRUNTIME: 1. DIRECTML (Sử dụng trên mọi GPU) / 2. OPENVINO (Chỉ Intel GPU và tối ưu nhất)")

            while 1:
                provider_selected = str(input("Hãy chọn phiên bản ONNXRUNTIME (1-2): ")).strip().upper()

                if provider_selected in ("1", "DIRECTML"):
                    install_xpu("directml")
                    break
                elif provider_selected in ("2", "OPENVINO"):
                    install_xpu("openvino")
                    break
                else:
                    print("Vui lòng chọn giá trị hợp lệ!")

            break
        else:
            print("Vui lòng chọn phiên bản hợp lệ!")
    
    print("Hoàn tất quá trình cài đặt thư viện!")

if __name__ == "__main__":
    if not os.path.exists(python_dir):
        print("Không tìm thấy thời gian chạy python!")
        sys.exit()
    
    if sys.platform != "win32":
        print("Chỉ hỗ trợ trên Windows!")
        sys.exit()

    print("Lưu Ý: Hãy cài các phần phụ trợ bên ngoài được đề cập trong phần README.md trước khi tiếp tục!")

    install_requirements()
    install_ffmpeg()