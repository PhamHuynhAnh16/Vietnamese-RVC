<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# Vietnamese RVC
Công cụ huấn luyện, chuyển đổi giọng nói chất lượng và hiệu suất cao đơn giản.

[![Vietnamese RVC](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/blob/main/LICENSE)

</div>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AnhP/RVC-GUI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/AnhP/Vietnamese-RVC-Project)

</div>

## Mô Tả Dự Án

Dự án này là một công cụ chuyển đổi giọng nói. Với mục tiêu tạo ra các sản phẩm chuyển đổi giọng nói chất lượng cao và hiệu suất tối ưu, dự án cho phép người dùng thay đổi giọng nói một cách mượt mà, tự nhiên.

Dự án này hướng tới sự thử nghiệm nghiên cứu của cá nhân hơn là về sự trải nghiệm, độ ổn định và có thể xảy ra lỗi trong quá trình sử dụng, nếu bạn muốn hướng đến một dự án có sự ổn định, mượt mà nhất hãy dùng thử [Applio](https://github.com/IAHispano/Applio), nếu bạn muốn hướng tới thử nghiệm đây sẽ là dự án dành cho bạn.

Dự án này có thể sẽ không cung cấp bản đóng gói sẳn, chỉ cung cấp mã nguồn và một số hướng dẫn. Để sử dụng được dự án này bạn sẽ phải tự cài đặt thủ công, quá trình cài đặt có thể sẽ rất phức tạp nên nếu bạn vẫn muốn sử dụng có thể liên hệ tôi thông qua discord.

## Các Tính Năng Chính Của Dự Án

**🔊 Nhóm Suy Luận (Xử Lý Âm Thanh)**

- Tách Nhạc: Sử dụng các thuật toán MDX-NET, VR, Demucs để tách lời hát và nhạc cụ một cách sạch sẽ.

- Chuyển Đổi Giọng Nói: Hỗ trợ chuyển đổi đơn lẻ, theo tệp đã tách hoặc xử lý hàng loạt cùng lúc.

- Chuyển Đổi Kết Hợp Nhận Diện Giọng Nói: Nhận dạng, chia tách và chuyển đổi với 2 mô hình giọng nói.

- Chuyển Văn Bản Thành Giọng Nói: Tạo giọng nói tự nhiên từ văn bản, tệp văn bản và tệp SRT.

- Chế Độ Thời Gian Thực: Chuyển đổi giọng nói trực tiếp với độ trễ thấp.

**⚙️ Nhóm Huấn Luyện (Phát Triển Mô Hình)**

- Huấn Luyện Mô Hình: Tùy chỉnh các thông số kỹ thuật để huấn luyện ra mô hình giọng nói chất lượng nhất.

- Tạo Dữ Liệu Huấn Luyện: Tự động cắt, lọc và tiền xử lý âm thanh thô từ đường dẫn Youtube.

- Tạo Bộ Tham Chiếu: Trích xuất các đặc trưng giọng nói làm chuẩn kiểm tra cho quá trình huấn luyện.

**🛠️ Nhóm Công Cụ Mở Rộng & Tùy Chỉnh**

- Điều Chỉnh Mô Hình: Dung hợp nhiều giọng nói, đọc thông tin tệp mô hình, chuyển đổi định dạng ONNX và SVC.

- Xử Lý Nâng Cao: Trích xuất cao độ (F0) và tự động tạo tệp phụ đề (SRT).

- Tùy Chỉnh Hệ Thống: Thay đổi chế độ Sáng/Tối, ngôn ngữ, chủ đề và phông chữ linh hoạt.

- Hiệu suất: Tùy chọn định dạng tính toán để tối ưu phần cứng: BF16, TF32, FP16, FP32.

**📥 Nhóm Tải Xuống**

- Mô Hình Giọng Nói: Tải trực tiếp qua liên kết, tìm kiếm trong kho dữ liệu và CSV hoặc tải lên từ máy tính.

- Mô Hình Huấn Luyện Sẵn: Tải trực tiếp qua liên kết, cung cấp danh sách mô hình hoặc tải lên từ máy tính.

## Công Nghệ Bên Trong Của Dự Án

**🎼 Phương Thức Trích Xuất Cao Độ (30+ Phương Pháp)**

- Các thuật toán: `pm, dio, crepe, fcpe, rmvpe, hpa-rmvpe, harvest, yin, pyin, swipe, piptrack, penn, djcm, swift, pesto...`

- Chế độ Trộn (Hybrid): Kết hợp nhiều phương thức (ví dụ: hybrid[rmvpe+harvest]) để tối ưu chất lượng.

💡 Lời khuyên: Khuyến nghị sử dụng RMVPE cho hầu hết các trường hợp để đảm bảo chất lượng ổn định nhất.

**🧠 Mô Hình Trích Xuất Nhúng (20+ Mô Hình)**

- Đa ngôn ngữ: `contentvec_base, hubert_base, vietnamese_hubert_base, japanese_hubert_base, korean_hubert_base, chinese_hubert_base, portuguese_hubert_base, spin, whisper`

- Định dạng mô hình được hỗ trợ: `fairseq (.pt)`, `onnx (.onnx)`, `transformers (.bin - .json)`, `spin (.bin - .json)`, `whisper (.pt)`.

⚠️ Lưu ý quan trọng:

- contentvec_base và hubert_base gần giống nhau và có thể dùng cho nhau, chỉ khác nhau về dung lượng và độ chính xác khi suy luận.

- Việc thay đổi mô hình nhúng yêu cầu huấn luyện lại mô hình RVC từ đầu. Các mô hình thông dụng hiện nay chủ yếu dùng contentvec_base hoặc hubert_base.

**🔊 Bộ Mã Hóa Giọng Nói (Vocoders)**

- Default (HiFi-GAN-NSF): Tùy chọn tiêu chuẩn, tương thích hoàn hảo với tất cả các phiên bản RVC.

- MRF-HiFi-GAN: Nâng cấp độ trung thực của âm thanh lên một tầm cao mới.

- RefineGAN: Mang lại chất lượng âm thanh vượt trội, trong trẻo và sắc nét.

- BigVGAN: Đỉnh cao của chất lượng âm thanh (siêu cao), nhưng hãy cẩn thận vì nó có thể biến GPU của bạn thành một chiếc "lò nướng" thực thụ do yêu cầu tài nguyên rất lớn.

## Yêu Cầu Hệ Thống

**1. Yêu cầu phần mềm**

- Hệ điều hành: Windows 10/11 hoặc Linux (Ubuntu).

- Python: Phiên bản 3.10, 3.11 hoặc 3.12.

- Thành phần bổ trợ bắt buộc:

  - Windows: Visual C++ Redistributable Runtimes.

  - Linux: Các thư viện bổ trợ (build-essential, libssl-dev, ffmpeg, v.v.).

**2. Yêu cầu phần cứng**

**Cấu Hình Tối Thiểu**

CPU: Hỗ trợ AVX và có ít nhất 2 nhân.

RAM: Tối thiểu 8GB (Khuyến nghị 16GB để xử lý các tệp âm thanh dài).

GPU: Không bắt buộc (Có thể chạy bằng CPU nhưng tốc độ siêu chậm).

Lưu trữ: 10GB trống (Nếu chỉ dùng cơ bản).

**Cấu Hình Khuyến Nghị**

CPU: Hỗ trợ AVX2, AVX512 và có ít nhất 4 nhân.

RAM: 16GB - 32GB trở lên.

GPU: NVIDIA RTX hoặc INTEL ARC (Tối thiểu 6GB hoặc 8GB trở lên).

Lưu trữ: 10GB trống (Nếu chỉ dùng cơ bản) hoặc hơn 120GB trống (Nếu dùng hết tất cả assets).

**Đây chỉ là cấu hình tham khảo vì không có một cấu hình xác định, nó sẽ dựa vào những thứ bạn dùng, thời gian và độ kiên nhẫn của bạn. Bạn có thể chỉ có CPU và quyết định huấn luyện mô hình trên nó thì cũng chả ai cấm được bạn.**

**GPU AMD không được đề cập do nếu sử dụng thông qua DIRECTML hoặc OPENCL sẽ cực kỳ chậm chỉ nhanh hơn sử dụng CPU, dùng thông qua Zluda thì thiếu ổn định, thời gian biên dịch lâu và yêu cầu nằm trong danh sách hỗ trợ của AMD. ROCM thì chưa được thử nghiệm và nó không có phiên bản dành cho Windows.**

## Cài Đặt Dự Án

**Trước tiên bạn cần tải mã nguồn về máy, bạn có thể thực hiện nó thông qua hai cách.**

Cách 1. Sử dụng đối với Git:
- git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git
- cd Vietnamese-RVC

Cách 2. Tải trực tiếp trên github:
- Nhấn vào [Vietnamese RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/archive/refs/heads/main.zip) để tải trực tiếp về.
- Giải nén `Vietnamese-RVC-main.zip`.
- Vào thư mục Vietnamese-RVC-main, nhấp vào thanh đường dẫn tệp nhập `cmd` và nhấn Enter để mở Terminal.

**Tiếp theo bạn sẽ thực hiến tiếp các bước cài đặt.**

<details>
<summary style="font-size: 20px;"> Đối Với Hệ Điều Hành Windows </summary>

- Cần cài đặt bộ [Visual C++ Redistributable Runtimes](https://www.techpowerup.com/download/visual-c-redistributable-runtime-package-all-in-one/) trước khi tiến hành tiếp.

- Bạn có thể chạy trực tiếp tệp `run_install.bat` để cài đặt hoặc làm theo các bước bên dưới.

- Khi chạy `run_install.bat` và bạn sử dụng CUDA thì sẽ được hỏi có cài TensorRT hay không, nếu chọn có bạn sẽ cần cài thêm [TensorRT](https://developer.nvidia.com/tensorrt) từ Nvidia, giải nén và thêm đường dẫn thư mục bin của nó vào [PATH](https://www.google.com/search?q=path+environment+variable+windows) hệ thống.

- Khi chạy `run_install.bat` và bạn sử dụng XPU thì sẽ được hỏi chọn phiên bản ONNXRUNTIME, tôi khuyên bạn chọn DIRECTML vì quá trình thử nghiệm của tôi OPENVINO có một số lỗi về kích thước động.

- Khi sử dụng ONNXRUNTIME OPENVINO, bạn cần cài đặt [OpenVino Toolkit](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/windows/openvino_toolkit_windows_2025.4.1.20426.82bbf0292c5_x86_64.zip) giải nén và thêm Release từ \runtime\bin\intel64\Release vào [PATH](https://www.google.com/search?q=path+environment+variable+windows) hệ thống.

- Hãy chuẩn bị và cài đặt PYTHON phiên bản 3.10.x, 3.11.x hoặc 3.12.x và tạo môi trường ảo.

- Sau quá trình cài đặt, bạn có thể xóa thư mục .uv bên trong thư mục assets để giảm nhẹ dung lượng.

**1. Tạo Môi Trường Ảo:**

```
python -m venv env
mkdir -p assets/.uv
set UV_CACHE_DIR="assets/.uv"
env\Scripts\python.exe -m pip install uv
env\Scripts\python.exe -m uv pip install six packaging python-dateutil platformdirs pywin32 wget
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
env\Scripts\python.exe -m uv pip install torch==2.7.0 torchaudio==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cu118
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
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
env\Scripts\python.exe -m uv pip install onnxruntime-gpu
env\Scripts\python.exe -m uv pip install -r requirements.txt
env\Scripts\python.exe -m uv pip install faiss-cpu==1.13.2
```

- Đối với Cuda 13.0 (Dành cho GPU 50-Series trở lên)

```
env\Scripts\python.exe -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
env\Scripts\python.exe -m uv pip install onnxruntime-gpu
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
  - DIRECTML không có hàm để dọn dẹp bộ nhớ, tôi đã tạo 1 hàm đơn giản để dọn dẹp bộ nhớ nhưng có thể sẽ không quá hiệu quả.
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
  - ONNXRUNTIME OPENVINO gặp lỗi với kích thước động nên là khuyên không sử dụng OPENVINO.
  - ONNXRUNTIME DIRECTML có thể sẽ chậm hơn cả sử dụng mô hình PYTORCH thuần nên là cũng không quá khuyên sử dụng các mô hình ONNX.
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
  - Nếu Onnxruntime không hoạt động, bạn có thể thử đổi sang phiên bản onnxruntime-directml.
  - Zluda không được thiết kế cho các hệ thống thời gian thực nên chức năng thời gian thực của ứng dụng sẽ bị vô hiệu hóa.
  - Bạn có thể thử Onnxruntime ROCm nhưng nó chỉ hoạt động trên python 3.10.x hoặc 3.12.x.
  ```
  env\Scripts\python.exe -m pip uninstall onnxruntime-gpu
  env\Scripts\python.exe -m uv pip install onnxruntime-rocm
  ```

</details>
</details>

<details>
<summary style="font-size: 20px;"> Đối Với Hệ Điều Hành Linux </summary>

**1. Thiết Lập Môi Trường Ảo:**

Hãy thay phiên bản PYTHON 3.11.9 thành 3.12.0 nếu như bạn dùng ROCm.

```
sudo apt update -y
sudo apt install -y curl git build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
curl https://pyenv.run | bash
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
source ~/.bashrc
pyenv local --unset
pyenv shell --unset
pyenv install 3.11.9
pyenv global 3.11.9
python -m pip install uv
python -m uv pip install six packaging python-dateutil platformdirs wget
```

**2. Cài Đặt Theo Từng Phần Cứng:**

Tiếp tục chạy các lệnh này để tiếp tục cài đặt theo từng phần cứng.

<details>
<summary>Đối với CPU (Sử dụng CPU)</summary>

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch==2.7.0 torchaudio==2.7.0 torchvision --index-url https://download.pytorch.org/whl/cpu
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
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu128
python -m uv pip install onnxruntime-gpu
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Đối với Cuda 13.0 (Dành cho GPU 50-Series trở lên)

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu130
python -m uv pip install onnxruntime-gpu
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
<summary>Đối với ROCm (Sử dụng đối với GPU AMD hỗ trợ RDNA)</summary>

- Đối với ROCm 5.7

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm5.7
python -m uv pip install onnxruntime-rocm
python -m uv pip install -r requirements.txt
```

- Đối với ROCm 6.4

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.4
python -m uv pip install onnxruntime-rocm
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Đối với ROCm 7.2

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm7.2
python -m uv pip install onnxruntime-rocm
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Lưu Ý:
  - Sử dụng ROCm trên dự án này không thực sự được khuyến nghị vì tôi chưa thử nghiệm được ROCm.
  - ROCm cũng khá là kén GPU AMD, nên là hãy chọn cho đúng phiên bản cho GPU của bạn.
  - Hãy thử nghiệm bằng `python -c "import torch; print(torch.cuda.is_available())"` nếu như trả về False nghĩa là cài đặt thất bại.

</details>


<details>
<summary>Đối với XPU (Sử dụng đối với GPU INTEL)</summary>

- Trước tiên nếu như bạn muốn sử dụng ONNXRUNTIME OPENVINO hãy cài đặt [OpenVino Toolkit](https://storage.openvinotoolkit.org/repositories/openvino/packages/2025.4.1/linux/) giải nén và đặt biến môi trường hoặc chạy `setupvars.sh` trong cùng một phiên.

- Cài đặt môi trường cho GPU Intel:

```
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt-get install -y libze-intel-gpu1 libze1 intel-metrics-discovery intel-opencl-icd clinfo intel-gsc
sudo apt-get install -y intel-media-va-driver-non-free libmfx-gen1 libvpl2 libvpl-tools libva-glx2 va-driver-all vainfo
sudo apt-get install -y libze-dev intel-ocloc
```

- Đối với sử dụng ONNXRUNTIME OPENVINO

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
python -m uv pip install openvino==2025.4.1
python -m uv pip install triton
python -m uv pip install onnxruntime-openvino==1.24.1
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Đối với sử dụng ONNXRUNTIME CPU

```
python -c "from main.app.install import remove_onnxruntime;remove_onnxruntime()"
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/xpu
python -m uv pip install triton
python -m uv pip install onnxruntime
python -m uv pip install -r requirements.txt
python -m uv pip install faiss-cpu==1.13.2
```

- Lưu ý: 
  - ONNXRUNTIME OPENVINO gặp lỗi với kích thước động nên là khuyên không sử dụng OPENVINO.
  - ONNXRUNTIME CPU khá chậm, cẩn thận với việc sử dụng mô hình ONNX.
  - XPU không hỗ trợ DDP, nên là bạn sẽ không thể sử dụng huấn luyện đa GPU.
  - XPU không hỗ trợ kiểu dữ liệu FP64 nên lớp GradScaler không hoạt động, lớp này đã được điều chỉnh và ép kiểu về FP32 có thể mất một chút chính xác.

</details>
</details>

## Colab Notebook

**Sử dụng với Google Colab**

- Mở Google Colab: [Vietnamese-RVC](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
- Bước 1: Chạy ô Cài đặt và đợi nó hoàn tất.
- Bước 2: Chạy ô Mở giao diện sử dụng (Khi này giao diện sẽ in ra 2 đường dẫn 1 là 0.0.0.0.7680 và 1 đường dẫn gradio có thể nhấp được, bạn chọn vào đường dẫn nhấp được và nó sẽ đưa bạn đến giao diện).

**Sử dụng với Kaggle**

- Mở Kaggle: [Vietnamese-RVC](https://www.kaggle.com/code/anhfxbphmhunh/vietnamese-rvc-kaggle/)
- Bước 1: Nhấn Copy - Edit.
- Bước 2: Chạy ô Cài đặt (Ô thứ nhất) và đợi nó hoàn tất.
- Bước 3: Chạy ô Mở giao diện sử dụng (Khi này giao diện sẽ in ra 3 đường dẫn 1 là đường dẫn đến localtunnel Tensorboard, 2 là 0.0.0.0.7680 và 3 đường dẫn gradio có thể nhấp được, bạn chọn vào đường dẫn nhấp được và nó sẽ đưa bạn đến giao diện).

## Tài Liệu Sử Dụng

**Tài liệu văn bản: [Words](/assets/Vietnamese-RVC-DOCS.pdf)**

## Sử Dụng Trên Máy Tính

**(Windows) Chạy tệp run_app.bat để mở giao diện sử dụng, chạy tệp run_tensorboard.bat để mở biểu đồ kiểm tra huấn luyện. (Lưu ý: không tắt Terminal)**
```
run_app.bat / run_tensorboard.bat
```

**Khởi động giao diện sử dụng. (Thêm `--allow_all_disk` vào lệnh để cho phép gradio truy cập tệp ngoài)**
```
env\Scripts\python.exe main\app\app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
env\Scripts\python.exe main\app\run_tensorboard.py --open
```

**Sử dụng bằng cú pháp**
```
env\Scripts\python.exe main\app\parser.py --help
```

**(Linux) Chạy tệp run_app.sh để mở giao diện sử dụng, chạy tệp run_tensorboard.sh để mở biểu đồ kiểm tra huấn luyện. (Lưu ý: không tắt Terminal)**
```
run_app.sh / run_tensorboard.sh
```

**Khởi động giao diện sử dụng.**
```
python main/app/app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
python main/app/run_tensorboard.py --open
```

**Sử dụng bằng cú pháp**
```
python main/app/parser.py --help
```

**Các khóa khi gọi giao diện dự án `main\app\app.py`**

`--client`: Khi được thêm vào nó sẽ kích hoạt chế độ Client của giao diện thời gian thực.

`--share`: Khi được thêm vào nó sẽ sử dụng đường dẫn chia sẽ của Gradio thay vì đường dẫn nội bộ.

`--open`: Khi được thêm vào nó sẽ trực tiếp mở giao diện khi khởi động thành công.

`--tensorboard`: Khi được thêm vào nó sẽ gọi trực tiếp Tensorboard trong cùng tiến trình giao diện.

`--allow_all_disk`: Khi được thêm vào, nó sẽ cho phép Gradio truy cập các tệp bên ngoài, giúp tránh lỗi không thể tải các tài nguyên ngoài trong giao diện Gradio.

`--debug`: Khi được thêm vào nó sẽ hiện thị tất cả gỡ lỗi của dự án.

## Giải Thích Một Số Thứ

**ONNX (Open Neural Network Exchange):**
Là một định dạng trung gian dùng để chuyển đổi mô hình (đặc biệt là từ PyTorch). ONNX giúp tối ưu hóa mô hình, tăng tốc thời gian suy luận và cho phép mô hình chạy trên nhiều runtime khác nhau Nhờ đó, việc triển khai trở nên linh hoạt và dễ dàng hơn trên nhiều nền tảng phần cứng.

**OCL (OpenCL - Open Computing Language):**
Là một tiêu chuẩn mở cho lập trình song song trên các hệ thống không đồng nhất. Trong dự án, OpenCL đóng vai trò là backend bổ trợ, giúp tận dụng các GPU không hỗ trợ tăng tốc xử lý máy học.

**DML (DirectML):**
Là API máy học của Microsoft, hoạt động trên nền tảng DirectX 12. DML cho phép chạy mô hình AI trên GPU (NVIDIA, AMD, Intel) trên Windows mà không cần CUDA, rất hữu ích cho việc mở rộng khả năng tương thích phần cứng.

**CUDA (Compute Unified Device Architecture):**
Là nền tảng tính toán song song của NVIDIA, cho phép khai thác sức mạnh GPU NVIDIA để tăng tốc các tác vụ AI và xử lý dữ liệu. Đây là backend phổ biến nhất do hiệu năng cao và hệ sinh thái cực kỳ mạnh mẽ.

**XPU:**
Là khái niệm (thường được Intel sử dụng) để chỉ các thiết bị tăng tốc tính toán tổng hợp (CPU + GPU). Trong PyTorch hoặc các framework khác, XPU thường ám chỉ GPU Intel (qua oneAPI hoặc IPEX). Việc hỗ trợ XPU giúp mở rộng khả năng chạy trên phần cứng của Intel.

**ROCm (Radeon Open Compute):**
Là nền tảng tính toán GPU của AMD, tương tự CUDA nhưng dành cho GPU AMD. ROCm cho phép chạy các mô hình AI trên GPU AMD với hiệu năng cao, đặc biệt trong môi trường Linux.

**ZLUDA:**
Là một lớp tương thích cho phép chạy các ứng dụng CUDA trên GPU không phải của NVIDIA (đặc biệt là dành cho GPU AMD) bằng cách ánh xạ các API CUDA sang nền tảng khác (như HIP - ROCm). ZLUDA giúp tận dụng các phần mềm chỉ hỗ trợ CUDA trên phần cứng không phải NVIDIA, tuy nhiên mức độ tương thích chưa hoàn toàn đầy đủ và hiệu năng có thể không ổn định tùy trường hợp.

**RVC (Retrieval-based Voice Conversion):**
Là một mô hình chuyển đổi giọng nói dựa trên truy hồi. RVC sử dụng embedding và chỉ mục để cải thiện chất lượng giọng nói đầu ra, giúp chuyển đổi giọng nhanh và tự nhiên hơn.

**SVC (Singing Voice Conversion):**
Là kỹ thuật chuyển đổi giọng hát, tập trung vào việc giữ nguyên giai điệu và nhịp điệu trong khi thay đổi giọng người hát. SVC thường phức tạp hơn RVC do phải xử lý thêm yếu tố âm nhạc như cao độ và biểu cảm.

## LƯU Ý

- **Các thanh trượt, ô chọn hay hộp thả có thể ẩn hiện theo từng tùy chọn và có thể nó nằm ẩn trong các menu xếp lớp nên hãy kiểm tra**
- **Hiện tại các bộ mã hóa mới như MRF HIFIGAN, REFINEGAN và BIGVGAN vẫn chưa đầy đủ các bộ huấn luyện trước**
- **Bộ mã hóa MRF HIFIGAN, REFINEGAN và BIGVGAN không hỗ trợ huấn luyện khi không không huấn luyện cao độ**
- **Huấn luyện năng lương chỉ thêm lớp học năng lượng nhưng gần như không tăng chất lượng**
- **Các mô hình trong kho lưu trữ Vietnamese-RVC được thu thập rải rác trên AI Hub, HuggingFace và các các kho lưu trữ khác. Có thể mang các giấy phép bản quyền khác nhau**

## Tuyên bố miễn trừ trách nhiệm

- **Dự án Vietnamese-RVC được phát triển với mục đích nghiên cứu, học tập và giải trí cá nhân. Tôi không khuyến khích cũng như không chịu trách nhiệm đối với bất kỳ hành vi lạm dụng công nghệ chuyển đổi giọng nói vì mục đích lừa đảo, giả mạo danh tính, hoặc vi phạm quyền riêng tư, bản quyền của bất kỳ cá nhân hay tổ chức nào.**

- **Người dùng cần tự chịu trách nhiệm với hành vi sử dụng phần mềm này và cam kết tuân thủ pháp luật hiện hành tại quốc gia nơi họ sinh sống hoặc hoạt động.**

- **Việc sử dụng giọng nói của người nổi tiếng, người thật hoặc nhân vật công chúng phải có sự cho phép hoặc đảm bảo không vi phạm pháp luật, đạo đức và quyền lợi của các bên liên quan.**

- **Tác giả của dự án không chịu trách nhiệm pháp lý đối với bất kỳ hậu quả nào phát sinh từ việc sử dụng phần mềm này.**

## Điều khoản sử dụng

- Bạn phải đảm bảo rằng các nội dung âm thanh bạn tải lên và chuyển đổi qua dự án này không vi phạm quyền sở hữu trí tuệ của bên thứ ba.

- Không được phép sử dụng dự án này cho bất kỳ hoạt động nào bất hợp pháp, bao gồm nhưng không giới hạn ở việc sử dụng để lừa đảo, quấy rối, hay gây tổn hại đến người khác.

- Bạn chịu trách nhiệm hoàn toàn đối với bất kỳ thiệt hại nào phát sinh từ việc sử dụng sản phẩm không đúng cách.

- Tôi sẽ không chịu trách nhiệm với bất kỳ thiệt hại trực tiếp hoặc gián tiếp nào phát sinh từ việc sử dụng dự án này.

## Dự án này được xây dựng dựa trên các dự án như sau

|                                                            Tác Phẩm                                                            |            Tác Giả            |      Giấy Phép      |
|--------------------------------------------------------------------------------------------------------------------------------|-------------------------------|---------------------|
| **[Applio](https://github.com/IAHispano/Applio/tree/main)**                                                                    | IAHispano                     | MIT License         |
| **[Python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main)**                                 | Nomad Karaoke                 | MIT License         |
| **[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main)**  | RVC Project                   | MIT License         |
| **[RVC-ONNX-INFER-BY-Anh](https://github.com/PhamHuynhAnh16/RVC_Onnx_Infer)**                                                  | Phạm Huỳnh Anh                | MIT License         |
| **[Torch-Onnx-Crepe-By-Anh](https://github.com/PhamHuynhAnh16/TORCH-ONNX-CREPE)**                                              | Phạm Huỳnh Anh                | MIT License         |
| **[Hubert-No-Fairseq](https://github.com/PhamHuynhAnh16/hubert-no-fairseq)**                                                   | Phạm Huỳnh Anh                | MIT License         |
| **[Local-attention](https://github.com/lucidrains/local-attention)**                                                           | Phil Wang                     | MIT License         |
| **[TorchFcpe](https://github.com/CNChTu/FCPE/tree/main)**                                                                      | CN_ChiTu                      | MIT License         |
| **[ContentVec](https://github.com/auspicious3000/contentvec)**                                                                 | Kaizhi Qian                   | MIT License         |
| **[Mediafiredl](https://github.com/Gann4Life/mediafiredl)**                                                                    | Santiago Ariel Mansilla       | MIT License         |
| **[Noisereduce](https://github.com/timsainb/noisereduce)**                                                                     | Tim Sainburg                  | MIT License         |
| **[World.py-By-Anh](https://github.com/PhamHuynhAnh16/world.py)**                                                              | Phạm Huỳnh Anh                | MIT License         |
| **[Mega.py](https://github.com/3v1n0/mega.py)**                                                                                | Marco Trevisan                | No License          |
| **[Gdown](https://github.com/wkentaro/gdown)**                                                                                 | Kentaro Wada                  | MIT License         |
| **[Whisper](https://github.com/openai/whisper)**                                                                               | OpenAI                        | MIT License         |
| **[PyannoteAudio](https://github.com/pyannote/pyannote-audio)**                                                                | pyannote                      | MIT License         |
| **[StftPitchShift](https://github.com/jurihock/stftPitchShift)**                                                               | Jürgen Hock                   | MIT License         |
| **[Penn](https://github.com/interactiveaudiolab/penn)**                                                                        | Interactive Audio Lab         | MIT License         |
| **[Voice Changer](https://github.com/deiteris/voice-changer)**                                                                 | Yury deiteris                 | MIT License         |
| **[Pesto](https://github.com/SonyCSLParis/pesto)**                                                                             | Sony CSL Paris                | LGPL 3.0            |
| **[PolTrain](https://github.com/Politrees/PolTrain)**                                                                          | Artyom Bebroy                 | MIT License         |
| **[Sovits-SVC-4.1](https://github.com/svc-develop-team/so-vits-svc/tree/4.1-Stable)**                                          | SVC Develop-Team              | AGPL 3.0            |
| **[RMVPE](https://github.com/yxlllc/RMVPE)**                                                                                   | yxlllc - Dream High           | Apache-2.0 License  |
| **[HPA-RMVPE](https://github.com/PhamHuynhAnh16/HPA-RMVPE)**                                                                   | Phạm Huỳnh Anh - Vidalnt      | No License          |
| **[DJCM](https://github.com/PhamHuynhAnh16/DJCM)**                                                                             | Phạm Huỳnh Anh - Dream High   | Apache-2.0 License  |

## Kho mô hình của công cụ tìm kiếm mô hình

- **[VOICE-MODELS.COM](https://voice-models.com/)**

## Báo cáo lỗi
- **Với trường hợp hệ thống báo cáo lỗi không hoạt động bạn có thể báo cáo lỗi cho tôi thông qua Discord `pham_huynh_anh` Hoặc [ISSUE](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues)**

## Lời Cảm Ơn

- Cảm ơn [Vidal](https://github.com/Vidalnt) đã cung cấp hỗ trợ phát triển HPA-RMVPE.
- Cảm ơn [Artyom Bebroy](https://github.com/Politrees) đã đề xuất CosineAnnealingLR cho huấn luyện.
- Cảm ơn [Dattobel](https://github.com/dattobel) đã cho mượn GPU Intel ARC để hỗ trợ phát triển dự án.
- Cảm ơn tất cả tác giả của các dự án được dựa vào đã cung cấp một nền tảng tuyệt vời để xây dựng dự án này.

## ☎️ Liên hệ tôi
- DISCORD: **pham_huynh_anh**