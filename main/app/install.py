import os
import sys
import subprocess

sys.path.append(os.getcwd())

os.makedirs(os.path.join("assets", ".uv"), exist_ok=True)
os.environ["UV_CACHE_DIR"] = os.path.join("assets", ".uv")

# Determine the correct Python executable path based on the operating system
python_dir = os.path.join("runtime", "python.exe") if sys.platform == "win32" else os.path.join("venv", "bin", "python")

def run_command(command):
    """
    Executes a shell command and catches any execution errors.

    Args:
        command (str): The shell command string to be executed.
    """

    try:
        # Run command via shell and block until completion; raise exception on non-zero exit code
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nLỗi khi thực thi lệnh: {e}")

def install_dependencies():
    """
    Installs core prerequisite packages using 'uv' for ultra-fast performance.
    """

    print("Cài đặt các thư viện cơ bản...")
    # Bootstrap 'uv' into the current python environment
    run_command(f"{python_dir} -m pip install uv")
    # Install foundational utilities needed for setup and platform compatibility
    run_command(f"{python_dir} -m uv pip install six packaging python-dateutil platformdirs wget")

def install_ffmpeg():
    """
    Downloads static binaries of FFmpeg and FFprobe for Windows environment 
    from a remote repository if they do not exist locally.
    """

    # Download ffmpeg.exe if missing
    if not os.path.exists("ffmpeg.exe"):
        print("Cài đặt ffmpeg...")
        run_command('curl -L -o ffmpeg.exe https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/ffmpeg/ffmpeg.exe')

    # Download ffprobe.exe if missing
    if not os.path.exists("ffprobe.exe"):
        print("Cài đặt ffprobe...")
        run_command('curl -L -o ffprobe.exe https://huggingface.co/AnhP/Vietnamese-RVC-Project/resolve/main/ffmpeg/ffprobe.exe')

def install_ffmpeg_linux():
    """
    Installs FFmpeg on Linux using the system's package manager (APT, DNF, or Pacman).
    """

    # Try apt (Debian/Ubuntu), fallback to dnf (Fedora/RHEL), fallback to pacman (Arch)
    run_command("sudo apt update && sudo apt install ffmpeg -y || sudo dnf install ffmpeg -y || sudo pacman -S --noconfirm ffmpeg")

def remove_onnxruntime():
    """
    Removes general 'onnxruntime' references from requirements.txt to prevent 
    conflicts before installing specialized hardware-specific versions.
    """

    if not os.path.exists("requirements.txt"): return

    # Read all lines from the requirements file
    with open("requirements.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Rewrite the file filtering out the generic onnxruntime package
    with open("requirements.txt", "w", encoding="utf-8") as f:
        f.writelines(l for l in lines if "onnxruntime" not in l)

def install_base_system_deps():
    """
    Installs Linux system level libraries required for compiling C++ extensions 
    and handling audio/video components.
    """

    print("\nĐang cập nhật hệ thống và cài đặt thư viện lõi...")
    # Installs compilation tools, Python development headers, PortAudio, and system/rendering libraries
    run_command(f"apt update && apt install -y build-essential python3-dev portaudio19-dev libsndfile1 libgomp1 libglib2.0-0")

def install_cpu():
    """
    Installs CPU-only versions of PyTorch, standard ONNX Runtime, 
    and general project dependencies.
    """

    print("\nCài đặt dành cho CPU...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện cpu...")
    # Fetch CPU optimized wheels from PyTorch's official repository
    run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu")
    run_command(f"{python_dir} -m uv pip install onnxruntime")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_cuda(version = "cu118"):
    """
    Installs NVIDIA CUDA-accelerated libraries based on the specified toolkit version.

    Args:
        version (str): The CUDA version string identifier (e.g., 'cu118', 'cu121', 'cu128', 'cu130').
    """

    print("\nCài đặt dành cho CUDA...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện cuda...")
    # Lock numpy and numba versions for older stable CUDA configurations to prevent ABI breakage
    if version in ("cu118", "cu121"): run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")

    # Install specific PyTorch build matching the targeted CUDA API
    if version == "cu128":
        # This version is the most stable.
        run_command(f"{python_dir} -m uv pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128")
    else:
        run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/{version}")
    
    # Install matching CUDA-optimized ONNX Runtime build
    run_command(f"{python_dir} -m uv pip install onnxruntime-gpu" + ("==1.20.1" if version in ("cu118", "cu121") else "==1.22.0"))
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

    # Faiss-cpu is required as fallback/index handler for newer CUDA suites
    if version in ("cu128", "cu130"): run_command(f"{python_dir} -m uv pip install faiss-cpu==1.13.2")

def install_directml():
    """
    Installs DirectML-compatible backend packages for hardware-agnostic 
    GPU acceleration (ideal for older NVIDIA, AMD, or Intel cards on Windows).
    """

    print("\nCài đặt dành cho DIRECTML...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện directml...")

    run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")
    run_command(f"{python_dir} -m uv pip install torch==2.4.1 torchaudio==2.4.1 torchvision")
    # Install the DirectML device extension plug-in for PyTorch
    run_command(f"{python_dir} -m uv pip install torch-directml==0.2.5.dev240914")
    run_command(f"{python_dir} -m uv pip install onnxruntime-directml")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_opencl():
    """
    Installs OpenCL-backed extensions allowing GPU execution over cross-vendor hardware platforms.
    """

    print("\nCài đặt dành cho OPENCL...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện opencl...")

    run_command(f"{python_dir} -m uv pip install numpy==1.26.4 numba==0.61.0")
    run_command(f"{python_dir} -m uv pip install torch==2.6.0 torchaudio==2.6.0 torchvision")
    # Pull pre-built OpenCL DLPrim implementation bridge wheel directly from GitHub releases
    run_command(f"{python_dir} -m uv pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp312-none-win_amd64.whl")
    run_command(f"{python_dir} -m uv pip install onnxruntime-directml")
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")

def install_xpu(provider = "openvino"):
    """
    Installs Intel XPU acceleration packages targeting Intel Arc and integrated Iris Xe graphics.

    Args:
        provider (str): Execution provider for ONNX, either 'openvino' or 'directml'.
    """

    print("\nCài đặt dành cho XPU...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện xpu...")
    # Pull Intel XPU-enabled Torch wheel
    run_command(f"{python_dir} -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu")
    run_command(f"{python_dir} -m uv pip install openvino==2025.4.1")
    run_command(f"{python_dir} -m uv pip install triton" + "-windows" if sys.platform == "win32" else "")
    # Dynamically pick the optimized inferencing runtime provider based on user preference
    run_command(f"{python_dir} -m uv pip install " + ("onnxruntime-openvino==1.24.1" if provider == "openvino" else "onnxruntime-directml"))
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")
    run_command(f"{python_dir} -m uv pip install faiss-cpu==1.13.2")

def install_rocm(gfx_version, provider = "rocm"):
    """
    Installs AMD ROCm software stack dependencies mapped to specific GPU architectures.

    Args:
        gfx_version (str): The specific hardware ISA architecture code.
        provider (str): Execution target flavor, defaults to 'rocm'.
    """

    print(f"\nCài đặt dành cho ROCm ({gfx_version})...")

    install_dependencies()
    remove_onnxruntime()

    print("Cài đặt các gói thư viện rocm...")
    # Fetch nightly/staging builds targeting specific AMD ISA architectures
    run_command(f"{python_dir} -m uv pip install torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2-staging/{gfx_version}/")
    run_command(f"{python_dir} -m uv pip install onnxruntime" + "-rocm" if provider == "rocm" else "-directml") 
    run_command(f"{python_dir} -m uv pip install -r requirements.txt")
    run_command(f"{python_dir} -m uv pip install faiss-cpu==1.13.2")

def install_requirements_win():
    """
    Interactive terminal menu to select and deploy hardware backends specifically on Windows OS.
    """

    print("\nCác phiên bản Backend:")
    print("- 1. CPU (Đối với CPU).")
    print("- 2. CUDA (Đối với GPU NVIDIA).")
    print("- 3. DIRECTML (Đối với GPU NVIDIA (Dòng cũ như GTX 600 trở xuống) / AMD (Dòng cũ không hỗ trợ ROCM) / INTEL).")
    print("- 4. OPENCL (Đối với GPU NVIDIA (Dòng cũ như GTX 600 trở xuống) / AMD (Dòng cũ không hỗ trợ ROCM) / INTEL).")
    print("- 5. XPU (Đối với GPU INTEL Arc hoặc GPU hỗ trợ XPU).")
    # ROCm is still not very stable.
    # print("- 6. ROCM (Đối với GPU AMD hỗ trợ ROCM).")

    # Primary selection loop
    while 1:
        selected = str(input("Hãy chọn phiên bản Backend (Nhập 1-5): ")).strip().upper()

        if selected in ("1", "CPU"):
            install_cpu()
            break
        elif selected in ("2", "CUDA"):
            print("\nCác phiên bản CUDA (Lưu ý: Hãy chọn đúng phiên bản nếu như chọn sai có thể gây ra lỗi suy luận hoặc huấn luyện): \n 1. Phiên bản 118 dành cho các GPU GTX 10-Series trở lên. \n 2. Phiên bản 121 dành cho các GPU RTX 20-Series đến 30-Series trở lên. \n 3. Phiên bản 128 dành cho các GPU RTX 30-Series đến RTX 40-Series trở lên. \n 4. Phiên bản 130 dành cho các GPU RTX 50-Series trở lên.")

            cuda_selected = None
            # CUDA version sub-loop selection
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
                    print("\nVui lòng chọn phiên bản CUDA hợp lệ!")
            
            # Optional high-level acceleration frameworks selection for modern CUDA versions
            if cuda_selected in ("3", "4", "cu128", "cu130"):
                print("\nCài đặt gói phụ thuộc TensorRT (không khuyên dùng với người dùng phổ thông)")
                print("Lưu ý: Nếu cài đặt gói TensorRT, bạn sẽ cần thêm gói TensorRT từ Nvidia và thêm nó vào PATH hệ thống: https://developer.nvidia.com/tensorrt")

                while 1:
                    tensorrt_selected = str(input("Bạn có muốn sử dụng TensorRT không (Y/N): ")).strip().upper()

                    if tensorrt_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install tensorrt")
                        break
                    elif tensorrt_selected in ("N", "NO"):
                        break
                    else:
                        print("\nVui lòng chọn giá trị hợp lệ!")
                
                print("\nCài đặt gói phụ thuộc Triton (không khuyên dùng với người dùng phổ thông)")

                while 1:
                    triton_selected = str(input("Bạn có muốn sử dụng Triton không (Y/N): ")).strip().upper()

                    if triton_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install triton-windows")
                        break
                    elif triton_selected in ("N", "NO"):
                        break
                    else:
                        print("\nVui lòng chọn giá trị hợp lệ!")
            break
        elif selected in ("3", "DIRECTML"):
            install_directml()
            break
        elif selected in ("4", "OPENCL"):
            install_opencl()
            break
        elif selected in ("5", "XPU"):
            print("\nPhiên bản ONNXRUNTIME: \n 1. DIRECTML (Ổn định). \n 2. OPENVINO (Nhanh hơn).")

            while 1:
                provider_selected = str(input("Hãy chọn phiên bản ONNXRUNTIME (1-2): ")).strip().upper()

                if provider_selected in ("1", "DIRECTML"):
                    install_xpu("directml")
                    break
                elif provider_selected in ("2", "OPENVINO"):
                    install_xpu("openvino")
                    break
                else:
                    print("\nVui lòng chọn giá trị hợp lệ!")

            break
        elif selected in ("6", "ROCM"):
            print("\nLưu ý: ROCM có thể gặp vấn đề về hiệu suất hoặc lỗi trên các thiết bị chạy Windows và hãy chắc chắn GPU của bạn nằm trong danh sách hỗ trợ")

            print("\nPhân loại ROCm:")
            print(" 1. ROCM-DGPU (Dành cho GPU rời)")
            print(" 2. ROCM-APU (Dành cho GPU tích hợp)")

            while 1:
                rocm_type = str(input("Hãy chọn loại GPU của bạn (Nhập 1 hoặc 2): ")).strip().upper()

                if rocm_type in ("1", "ROCM-DGPU"):
                    print("\nCác phiên bản ROCM (Lưu ý: hãy chọn đúng phiên bản theo kiến trúc GPU của bạn, nếu không sẽ không thể nhận dạng được thiết bị)")
                    print(" 1. gfx900       -> Vega 56, Vega 64, Frontier Edition")
                    print(" 2. gfx906       -> Radeon VII, MI50, MI60")
                    print(" 3. gfx908       -> Instinct MI100")
                    print(" 4. gfx90a       -> Instinct MI210, MI250, MI250X")
                    print(" 5. gfx101X-dgpu -> Dòng RDNA1 (RX 5500, RX 5600, RX 5700 XT...)")
                    print(" 6. gfx103X-dgpu -> Dòng RDNA2 (RX 6600, RX 6700 XT, RX 6800, RX 6900 XT...)")
                    print(" 7. gfx110X-dgpu -> Dòng RDNA3 (RX 7600, RX 7700 XT, RX 7800 XT, RX 7900 XTX...)")
                    print(" 8. gfx120X-all -> Dòng RDNA4 (RX 9060, RX 9060 XT, RX 9070 XT...)")
                    
                    dgpu_mapping = {
                        "1": "gfx900", 
                        "2": "gfx906", 
                        "3": "gfx908", 
                        "4": "gfx90a", 
                        "5": "gfx101X-dgpu", 
                        "6": "gfx103X-dgpu", 
                        "7": "gfx110X-dgpu",
                        "8": "gfx120X-all"
                    }

                    while 1:
                        sub_select = str(input("Chọn số tương ứng với GPU rời của bạn (1-7): ")).strip()
                        if sub_select in dgpu_mapping:
                            install_rocm(dgpu_mapping[sub_select], "directml")
                            break
                        else:
                            print("\nVui lòng chọn đúng số từ 1 đến 7!")
                    break
                elif rocm_type in ("2", "ROCM-APU"):
                    print("\nCác phiên bản ROCM (Lưu ý: hãy chọn đúng phiên bản theo kiến trúc APU của bạn, nếu không sẽ không thể nhận dạng được thiết bị)")
                    print(" 1. gfx103X-all  -> APU tích hợp RDNA2 (Steam Deck, Ryzen 6000/7000 iGPU...)")
                    print(" 2. gfx110X-all  -> APU tích hợp RDNA3 (Ryzen 7040/8040 iGPU, ROG Ally...)")
                    print(" 3. gfx1150      -> APU Ryzen AI 9 dòng 300 / Strix Point (Radeon 880M/890M)")
                    print(" 4. gfx1151      -> APU Strix Halo hoặc các biến thể đặc biệt")
                    print(" 5. gfx1152      -> Biến thể APU RDNA3.5 di động tiết kiệm điện")
                    print(" 6. gfx1153      -> Biến thể APU RDNA3.5 phân khúc phổ thông")
                    print(" 7. gfx120X-all  -> Kiến trúc APU RDNA4 tích hợp mới nhất")
                    
                    apu_mapping = {
                        "1": "gfx103X-all", 
                        "2": "gfx110X-all", 
                        "3": "gfx1150", 
                        "4": "gfx1151", 
                        "5": "gfx1152", 
                        "6": "gfx1153", 
                        "7": "gfx120X-all"
                    }
                    
                    while 1:
                        sub_select = str(input("Chọn số tương ứng với APU của bạn (1-7): ")).strip()
                        if sub_select in apu_mapping:
                            install_rocm(apu_mapping[sub_select], "directml")
                            break
                        else:
                            print("\nVui lòng chọn đúng số từ 1 đến 7!")
                    break
                else:
                    print("\nVui lòng chọn 1 hoặc 2!")
            break
        else:
            print("\nVui lòng chọn phiên bản hợp lệ!")
    
    print("\nHoàn tất quá trình cài đặt thư viện!")

def install_requirements_linux():
    """
    Interactive terminal menu to select and deploy hardware backends specifically on Linux distributions.
    """

    print("\nCác phiên bản Backend:")
    print("- 1. CPU (Đối với CPU).")
    print("- 2. CUDA (Đối với GPU NVIDIA).")
    print("- 3. XPU (Đối với GPU INTEL Arc hoặc GPU hỗ trợ XPU).")
    print("- 4. ROCM (Đối với GPU AMD hỗ trợ ROCM).")

    while 1:
        selected = str(input("Hãy chọn phiên bản Backend (Nhập 1-5): ")).strip().upper()

        if selected in ("1", "CPU"):
            install_base_system_deps()
            install_cpu()
            break
        elif selected in ("2", "CUDA"):
            run_command(f"apt update && apt install -y nvidia-cuda-toolkit")
            print("\nCác phiên bản CUDA (Lưu ý: Hãy chọn đúng phiên bản nếu như chọn sai có thể gây ra lỗi suy luận hoặc huấn luyện): \n 1. Phiên bản 118 dành cho các GPU GTX 10-Series trở lên. \n 2. Phiên bản 121 dành cho các GPU RTX 20-Series đến 30-Series trở lên. \n 3. Phiên bản 128 dành cho các GPU RTX 30-Series đến RTX 40-Series trở lên. \n 4. Phiên bản 130 dành cho các GPU RTX 50-Series trở lên.")

            cuda_selected = None

            while 1:
                cuda_selected = str(input("Hãy chọn phiên bản CUDA (Nhập 1-4): ")).strip().upper()
                
                if cuda_selected in ("1", "118"):
                    install_base_system_deps()
                    install_cuda("cu118")
                    break
                elif cuda_selected in ("2", "121"):
                    install_base_system_deps()
                    install_cuda("cu121")
                    break
                elif cuda_selected in ("3", "128"):
                    install_base_system_deps()
                    install_cuda("cu128")
                    break
                elif cuda_selected in ("4", "130"):
                    install_base_system_deps()
                    install_cuda("cu130")
                    break
                else:
                    print("\nVui lòng chọn phiên bản CUDA hợp lệ!")
            
            if cuda_selected in ("3", "4", "cu128", "cu130"):
                print("\nCài đặt gói phụ thuộc TensorRT (không khuyên dùng với người dùng phổ thông)")
                print("Lưu ý: Nếu cài đặt gói TensorRT, bạn sẽ cần thêm gói TensorRT từ Nvidia và thêm nó vào PATH hệ thống: https://developer.nvidia.com/tensorrt")

                while 1:
                    tensorrt_selected = str(input("Bạn có muốn sử dụng TensorRT không (Y/N): ")).strip().upper()

                    if tensorrt_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install tensorrt")
                        break
                    elif tensorrt_selected in ("N", "NO"):
                        break
                    else:
                        print("\nVui lòng chọn giá trị hợp lệ!")
                
                print("\nCài đặt gói phụ thuộc Triton (không khuyên dùng với người dùng phổ thông)")

                while 1:
                    triton_selected = str(input("Bạn có muốn sử dụng Triton không (Y/N): ")).strip().upper()

                    if triton_selected in ("Y", "YES"):
                        run_command(f"{python_dir} -m uv pip install triton")
                        break
                    elif triton_selected in ("N", "NO"):
                        break
                    else:
                        print("\nVui lòng chọn giá trị hợp lệ!")
            break
        elif selected in ("3", "XPU"):
            install_base_system_deps()
            install_xpu("openvino")
            break
        elif selected in ("4", "ROCM"):
            print("\nLưu ý: ROCM có thể gặp vấn đề về hiệu suất hoặc lỗi trên các thiết bị chạy Windows và hãy chắc chắn GPU của bạn nằm trong danh sách hỗ trợ")

            print("\nPhân loại ROCm:")
            print(" 1. ROCM-DGPU (Dành cho GPU rời)")
            print(" 2. ROCM-APU (Dành cho GPU tích hợp)")

            while 1:
                rocm_type = str(input("Hãy chọn loại GPU của bạn (Nhập 1 hoặc 2): ")).strip().upper()

                if rocm_type in ("1", "ROCM-DGPU"):
                    print("\nCác phiên bản ROCM (Lưu ý: hãy chọn đúng phiên bản theo kiến trúc GPU của bạn, nếu không sẽ không thể nhận dạng được thiết bị)")
                    print(" 1. gfx900       -> Vega 56, Vega 64, Frontier Edition")
                    print(" 2. gfx906       -> Radeon VII, MI50, MI60")
                    print(" 3. gfx908       -> Instinct MI100")
                    print(" 4. gfx90a       -> Instinct MI210, MI250, MI250X")
                    print(" 5. gfx101X-dgpu -> Dòng RDNA1 (RX 5500, RX 5600, RX 5700 XT...)")
                    print(" 6. gfx103X-dgpu -> Dòng RDNA2 (RX 6600, RX 6700 XT, RX 6800, RX 6900 XT...)")
                    print(" 7. gfx110X-dgpu -> Dòng RDNA3 (RX 7600, RX 7700 XT, RX 7800 XT, RX 7900 XTX...)")
                    print(" 8. gfx120X-all -> Dòng RDNA4 (RX 9060, RX 9060 XT, RX 9070 XT...)")
                    
                    dgpu_mapping = {
                        "1": "gfx900", 
                        "2": "gfx906", 
                        "3": "gfx908", 
                        "4": "gfx90a", 
                        "5": "gfx101X-dgpu", 
                        "6": "gfx103X-dgpu", 
                        "7": "gfx110X-dgpu",
                        "8": "gfx120X-all"
                    }

                    while 1:
                        sub_select = str(input("Chọn số tương ứng với GPU rời của bạn (1-7): ")).strip()
                        if sub_select in dgpu_mapping:
                            install_base_system_deps()
                            install_rocm(dgpu_mapping[sub_select], "rocm")
                            break
                        else:
                            print("\nVui lòng chọn đúng số từ 1 đến 7!")
                    break
                elif rocm_type in ("2", "ROCM-APU"):
                    print("\nCác phiên bản ROCM (Lưu ý: hãy chọn đúng phiên bản theo kiến trúc APU của bạn, nếu không sẽ không thể nhận dạng được thiết bị)")
                    print(" 1. gfx103X-all  -> APU tích hợp RDNA2 (Steam Deck, Ryzen 6000/7000 iGPU...)")
                    print(" 2. gfx110X-all  -> APU tích hợp RDNA3 (Ryzen 7040/8040 iGPU, ROG Ally...)")
                    print(" 3. gfx1150      -> APU Ryzen AI 9 dòng 300 / Strix Point (Radeon 880M/890M)")
                    print(" 4. gfx1151      -> APU Strix Halo hoặc các biến thể đặc biệt")
                    print(" 5. gfx1152      -> Biến thể APU RDNA3.5 di động tiết kiệm điện")
                    print(" 6. gfx1153      -> Biến thể APU RDNA3.5 phân khúc phổ thông")
                    print(" 7. gfx120X-all  -> Kiến trúc APU RDNA4 tích hợp mới nhất")
                    
                    apu_mapping = {
                        "1": "gfx103X-all", 
                        "2": "gfx110X-all", 
                        "3": "gfx1150", 
                        "4": "gfx1151", 
                        "5": "gfx1152", 
                        "6": "gfx1153", 
                        "7": "gfx120X-all"
                    }
                    
                    while 1:
                        sub_select = str(input("Chọn số tương ứng với APU của bạn (1-7): ")).strip()
                        if sub_select in apu_mapping:
                            install_base_system_deps()
                            install_rocm(apu_mapping[sub_select], "rocm")
                            break
                        else:
                            print("\nVui lòng chọn đúng số từ 1 đến 7!")
                    break
                else:
                    print("\nVui lòng chọn 1 hoặc 2!")
            break
        else:
            print("\nVui lòng chọn phiên bản hợp lệ!")
    
    print("\nHoàn tất quá trình cài đặt thư viện!")


if __name__ == "__main__":
    # Validate the existence of the python runtime before starting any tasks
    if not os.path.exists(python_dir):
        print("\nKhông tìm thấy thời gian chạy python!")
        sys.exit(1)
    
    # Platform-specific routing logic
    if sys.platform == "win32":
        print("Lưu ý: Bạn cần chắc chắn đã cài đặt thư viện Visual C++ Redistributable Runtimes trước khi tiếp tục.")
        input("Nhấn Enter để tiếp tục.")

        install_requirements_win()
        install_ffmpeg()
    elif sys.platform == "linux":
        # Linux environments require root/administrator privileges to install apt packages
        if os.getuid() != 0:
            print("❌ Lỗi: Bạn phải chạy với quyền sudo!")
            sys.exit(1)

        install_requirements_linux()
        install_ffmpeg_linux()
    else:
        print("\nHiện chỉ hỗ trợ cài đặt trên Windows và Linux!")
        sys.exit(1)