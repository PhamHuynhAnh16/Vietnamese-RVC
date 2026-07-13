## Cài Đặt Dự Án

**Trước tiên bạn cần tải mã nguồn về máy, bạn có thể thực hiện nó thông qua hai cách.**

Cách 1. Sử dụng đối với Git:
- git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git
- cd Vietnamese-RVC

Cách 2. Tải trực tiếp trên github:
- Nhấn vào [Vietnamese RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/archive/refs/heads/main.zip) để tải trực tiếp về.
- Giải nén `Vietnamese-RVC-main.zip`.
- Vào thư mục Vietnamese-RVC-main, nhấp vào thanh đường dẫn tệp nhập `cmd` và nhấn Enter để mở Terminal.

Tiếp theo bạn sẽ thực hiến tiếp các bước cài đặt.

### Đối với Windows

Cần cài đặt bộ [Visual C++ Redistributable Runtimes](https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/) trước khi tiến hành tiếp.

Hãy sử dụng các phiên bản Python 3.10.x / 3.11.x / 3.12.x

**1. Tạo Môi Trường Ảo:**

```
python -m venv env
mkdir -p assets/.uv
set UV_CACHE_DIR="assets/.uv"
env\Scripts\python.exe -m pip install uv
env\Scripts\python.exe -m uv pip install six packaging python-dateutil platformdirs wget
env\Scripts\python.exe -c "from main.app.install import install_ffmpeg;install_ffmpeg()"
```

**2. Cài Đặt Theo Từng Phần Cứng:**

Tiếp tục chạy các lệnh này để tiếp tục cài đặt theo từng phần cứng.

<details>
<summary>Đối với CPU (Sử dụng CPU)</summary>

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
env\Scripts\python.exe -m uv pip install onnxruntime
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

</details>


<details>
<summary>Đối với CUDA (Sử dụng GPU NVIDIA)</summary>

- Đối với Cuda 11.8 (Dành cho GPU 10-Series trở lên)

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.20.1
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

- Đối với Cuda 12.1 (Dành cho GPU 20-Series đến 30-Series trở lên)

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.20.1
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

- Đối với Cuda 12.8 (Dành cho GPU 30-Series đến 40-Series trở lên)

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.22.0
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

- Đối với Cuda 13.0 (Dành cho GPU 50-Series trở lên)

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.22.0
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

- Nếu bạn muốn sử dụng TensorRT cho ONNXRUNTIME:

  - Cài đặt Runtime [TensorRT](https://developer.nvidia.com/tensorrt) từ Nvidia, giải nén và thêm đường dẫn thư mục bin của nó vào [PATH](https://www.google.com/search?q=path+environment+variable+windows) hệ thống.

  - Tiếp theo cài đặt:

  ```
  env\Scripts\python.exe -m uv pip install tensorrt
  ```

- Nếu bạn muốn sử dụng Compile để biên dịch mô hình:

  - Cài đặt thư viện:
  ```
  env\Scripts\python.exe -m uv pip install triton-windows
  ```

- Lưu Ý:
  - Hãy lựa chọn đúng phiên bản đối với GPU của bạn, nếu không nó có thể gây ra lỗi liên quan đến suy luận hoặc huấn luyện.
  - TensorRT khá là không ổn định nên là không khuyến khích cài đặt và sử dụng.
  - Compile có thể không thực sự cần thiết, nó chỉ thực sự hữu dụng khi bạn cần suy luận theo lô hoặc suy luận với các đầu vào lớn.

</details>


<details>
<summary>Đối với OPENCL (Sử dụng đối với GPU hỗ trợ OPENCL, có thể là IGPU, AMD, INTEL, NVIDIA)</summary>

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch==2.6.0 torchaudio==2.6.0 torchvision
env\Scripts\python.exe -m uv pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp311-none-win_amd64.whl
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

- Lưu ý: 
  - Thay cp311 thành cp312 nếu bạn sử dụng python 3.12.
  - Có vẻ như OPENCL đã không còn được hỗ trợ tiếp.
  - OPENCL không hỗ trợ quá tốt tác vụ đa luồng nên khi chạy trích xuất thường sẽ bị khóa ở 1 luồng.
  - Chỉ nên cài đặt trên python 3.11 do không có bản biên dịch cho python 3.10 với torch 2.6.0.
  - Demucs có thể gây quá tải và tràn bộ nhớ đối với GPU (nếu cần sử dụng demucs hãy mở tệp config.json trong main\configs sửa đối số demucs_cpu_mode thành true).
  - DDP không hỗ trợ huấn luyện đa GPU đối với OPENCL.
  - Một số thuật toán khác phải chạy trên cpu nên có thể hiệu suất của GPU có thể không sử dụng hết.

</details>


<details>
<summary>Đối với DIRECTML (Sử dụng đối với GPU hỗ trợ DIRECTML, có thể là IGPU, AMD, INTEL, NVIDIA)</summary>

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch==2.4.1 torchaudio==2.4.1 torchvision
env\Scripts\python.exe -m uv pip install torch-directml==0.2.5.dev240914
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

- Lưu ý: 
  - DIRECTML đã ngừng phát triển một khoảng thời gian dài.
  - DIRECTML không hỗ trợ quá tốt tác vụ đa luồng nên khi chạy trích xuất thường sẽ bị khóa ở 1 luồng.
  - DIRECTML có hỗ trợ 1 phần fp16 nhưng không được khuyến khích sử dụng vì có thể chỉ nhận được hiệu năng tương đương fp32.
  - DIRECTML được thiết kế để suy luận chứ không phải dùng để huấn luyện mặc dù có thể hoàn toàn chạy được huấn luyện nhưng sẽ không được khuyến khích.

</details>


<details>
<summary>Đối với XPU (Sử dụng đối với GPU INTEL)</summary>

- Trước tiên nếu như bạn muốn sử dụng ONNXRUNTIME OPENVINO hãy cài đặt [OpenVino Toolkit](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/windows/openvino_toolkit_windows_2025.4.1.20426.82bbf0292c5_x86_64.zip) giải nén và thêm Release từ \runtime\bin\intel64\Release vào [PATH](https://www.google.com/search?q=path+environment+variable+windows) hệ thống.

- Đối với sử dụng ONNXRUNTIME OPENVINO

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
env\Scripts\python.exe -m uv pip install openvino==2025.4.1
env\Scripts\python.exe -m uv pip install triton-windows
env\Scripts\python.exe -m uv pip install onnxruntime-openvino==1.24.1
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

- Đối với sử dụng ONNXRUNTIME DIRECTML

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
env\Scripts\python.exe -m uv pip install triton-windows
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

- Lưu ý: 
  - XPU không hỗ trợ DDP, nên là bạn sẽ không thể sử dụng huấn luyện đa GPU.
  - XPU không hỗ trợ kiểu dữ liệu FP64 nên lớp GradScaler không hoạt động, lớp này đã được điều chỉnh và ép kiểu về FP32 có thể mất một chút chính xác.

</details>


<details>
<summary>Đối với ZLUDA (Sử dụng đối với GPU AMD hỗ trợ ROCm)</summary>

- Kiểm tra GPU của bạn có được hỗ trợ hay không: [ROCM-Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html).
- Tải và cài đặt: [VC++ Runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) và [HIP-SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).
- Thêm thư mục bin từ HIP-SDK vào [PATH](https://www.google.com/search?q=path+environment+variable+windows) hệ thống.

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch==2.7.0 torchaudio==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu118
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.20.1
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

- Sao chép path-zluda-hipxx.bat (Thay xx theo phiên bản HIP SDK của bạn) và run_app.bat ra thư mục chính thay thế các tệp hiện tại.
- Chạy tệp path-zluda-hipxx.bat.

-Lưu ý:
  - Zluda hoạt động bằng cách biên dịch mã cuda sang hip sdk và quá trình này diễn ra cực kỳ chậm và trong lúc đó gpu của bạn sẽ không được sử dụng.
  - Nếu GPU của bạn không nằm trong danh sách hỗ trợ (ví dụ như: gfx803) bạn có thể thử dùng HIP SDK 5.7.0 và ghi đè thư mục library trong ROCm/5.7.0/bin/rocblas từ [ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases) (Không Khuyến Khích).
  - Zluda không được thiết kế cho các hệ thống thời gian thực nên chức năng thời gian thực của ứng dụng sẽ bị vô hiệu hóa.
  - Nếu Onnxruntime không hoạt động, bạn có thể thử đổi sang phiên bản onnxruntime-directml.
  ```
  env\Scripts\python.exe -m pip uninstall onnxruntime-gpu
  env\Scripts\python.exe -m uv pip install onnxruntime-directml
  ```

</details>


<details>
<summary>Đối với ROCm (Sử dụng đối với GPU AMD hỗ trợ ROCm)</summary>

- Kiểm tra GPU của bạn có được hỗ trợ hay không: [ROCM-Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html).
- Tải và cài đặt: [VC++ Runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) và [HIP-SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).
- Thay `{gfx_version}` thành mã GPU của bạn, các mã được hỗ trợ (`gfx101X-dgpu, gfx103X-all, gfx103X-dgpu, gfx110X-all, gfx110X-dgpu, gfx1150, gfx1151, gfx1152, gfx1153, gfx900, gfx906, gfx908, gfx90a, gfx120X-all, gfx90X-dcgpu, gfx94X-dcgpu, gfx950-dcgpu`)
- Đặt biến môi trường `set MIOPEN_FIND_MODE=2` và `set MIOPEN_FIND_ENFORCE=1` vào tệp run_app.bat.

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2-staging/{gfx_version}/
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

Lưu Ý:
  - Hiệu năng có thể không được đảm bảo.
  - Sử dụng ROCm trên dự án này không thực sự được khuyến nghị vì chưa thử nghiệm.
  - ROCm cũng khá là kén GPU AMD, nên là hãy chọn cho đúng phiên bản cho GPU của bạn.

</details>



### Đối với Linux

**1. Thiết Lập Môi Trường Ảo:**

```
sudo apt update -y
sudo apt install -y build-essential python3-dev portaudio19-dev libsndfile1 libgomp1 libglib2.0-0 ffmpeg
sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv venv
source "./venv/bin/activate"
python -m pip install uv
python -m uv pip install six packaging python-dateutil platformdirs wget
```

**2. Cài Đặt Theo Từng Phần Cứng:**

Tiếp tục chạy các lệnh này để tiếp tục cài đặt theo từng phần cứng.

<details>
<summary>Đối với CPU (Sử dụng CPU)</summary>

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu
python -m uv pip install onnxruntime
python -m uv pip install -r requirements.txt
```

</details>


<details>
<summary>Đối với CUDA (Sử dụng GPU NVIDIA)</summary>

- Cài đặt Cuda Toolkit.

```
sudo apt update
sudo apt install nvidia-cuda-toolkit
```

- Đối với Cuda 11.8 (Dành cho GPU 10-Series trở lên)

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
python -m uv pip install onnxruntime-gpu==1.20.1
python -m uv pip install -r requirements.txt
```

- Đối với Cuda 12.1 (Dành cho GPU 20-Series đến 30-Series trở lên)

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
python -m uv pip install onnxruntime-gpu==1.20.1
python -m uv pip install -r requirements.txt
```

- Đối với Cuda 12.8 (Dành cho GPU 30-Series đến 40-Series trở lên)

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch==2.7.1 torchaudio==2.7.1 torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install onnxruntime-gpu==1.22.0
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Đối với Cuda 13.0 (Dành cho GPU 50-Series trở lên)

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
python -m uv pip install onnxruntime-gpu==1.22.0
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Nếu bạn muốn sử dụng TensorRT cho ONNXRUNTIME:

  - Cài đặt Runtime [TensorRT](https://developer.nvidia.com/tensorrt) từ Nvidia. Giải nén và đặt biến môi trường:
  ```
  tar -xvf TensorRT-*.tar.gz
  cd TensorRT-*

  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib
  export PATH=$PATH:$(pwd)/bin
  ```

  - Tiếp theo cài đặt:
  ```
  python -m uv pip install tensorrt
  ```

- Nếu bạn muốn sử dụng Compile để biên dịch mô hình:

  - Cài đặt thư viện:
  ```
  python -m uv pip install triton
  ```

- Lưu Ý:
  - Hãy lựa chọn đúng phiên bản đối với GPU của bạn, nếu không nó có thể gây ra lỗi liên quan đến suy luận hoặc huấn luyện.
  - TensorRT khá là không ổn định nên là không khuyến khích cài đặt và sử dụng.
  - Compile có thể không thực sự cần thiết, nó chỉ thực sự hữu dụng khi bạn cần suy luận theo lô hoặc suy luận với các đầu vào lớn.

</details>


<details>
<summary>Đối với ROCm (Sử dụng đối với GPU AMD hỗ trợ ROCm)</summary>

- Kiểm tra GPU của bạn có được hỗ trợ hay không: [ROCM-Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html).
- Thay `{gfx_version}` thành mã GPU của bạn, các mã được hỗ trợ (`gfx101X-dgpu, gfx103X-all, gfx103X-dgpu, gfx110X-all, gfx110X-dgpu, gfx1150, gfx1151, gfx1152, gfx1153, gfx900, gfx906, gfx908, gfx90a, gfx120X-all, gfx90X-dcgpu, gfx94X-dcgpu, gfx950-dcgpu`)
- Đặt biến môi trường `export MIOPEN_FIND_MODE=2` và `export MIOPEN_FIND_ENFORCE=1` vào tệp run_app.sh.

```
wget https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt install ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt update
sudo apt install "linux-headers-$(uname -r)" "linux-modules-extra-$(uname -r)"
sudo apt install amdgpu-dkms
sudo apt install python3-setuptools python3-wheel
sudo usermod -a -G render,video $LOGNAME
sudo apt install rocm
```

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2-staging/{gfx_version}/
python -m uv pip install onnxruntime-directml
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

Lưu Ý:
  - Hiệu năng có thể không được đảm bảo.
  - Sử dụng ROCm trên dự án này không thực sự được khuyến nghị vì chưa thử nghiệm.
  - ROCm cũng khá là kén GPU AMD, nên là hãy chọn cho đúng phiên bản cho GPU của bạn.

</details>


<details>
<summary>Đối với XPU (Sử dụng đối với GPU INTEL)</summary>

- Trước tiên nếu như bạn muốn sử dụng ONNXRUNTIME OPENVINO hãy cài đặt [OpenVino Toolkit](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/linux/) giải nén và đặt biến môi trường hoặc chạy `setupvars.sh` trong cùng một phiên.

- Cài đặt môi trường cho GPU Intel:

```
sudo apt-get update -y
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
sudo apt-get install -y libze-dev intel-ocloc
```

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
python -m uv pip install openvino==2025.4.1
python -m uv pip install triton
python -m uv pip install onnxruntime-openvino==1.24.1
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Lưu ý: 
  - XPU không hỗ trợ DDP, nên là bạn sẽ không thể sử dụng huấn luyện đa GPU.
  - XPU không hỗ trợ kiểu dữ liệu FP64 nên lớp GradScaler không hoạt động, lớp này đã được điều chỉnh và ép kiểu về FP32 có thể mất một chút chính xác.

</details>