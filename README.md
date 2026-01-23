<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# Vietnamese RVC BY ANH
Công cụ chuyển đổi giọng nói chất lượng và hiệu suất cao đơn giản.

[![Vietnamese RVC](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
[![Licence](https://img.shields.io/badge/LICENSE-MIT-green?style=for-the-badge)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/blob/main/LICENSE)

</div>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AnhP/RVC-GUI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/AnhP/Vietnamese-RVC-Project)

</div>

## Mô tả

Dự án này là một công cụ chuyển đổi giọng nói. Với mục tiêu tạo ra các sản phẩm chuyển đổi giọng nói chất lượng cao và hiệu suất tối ưu, dự án cho phép người dùng thay đổi giọng nói một cách mượt mà, tự nhiên.

Dự án này hướng tới sự thử nghiệm nghiên cứu của cá nhân hơn là về sự trải nghiệm, độ ổn định và có thể xảy ra lỗi, nếu bạn muốn hướng đến một dự án có sự ổn định, mượt mà nhất hãy dùng thử [Applio](https://github.com/IAHispano/Applio), nếu bạn muốn hướng tới thử nghiệm đây sẽ là dự án dành cho bạn.

Dự án này có thể sẽ không cung cấp bản dựng sẳn, chỉ cung cấp mã nguồn để sử dụng được dự án bạn phải tự dựng thủ công trên máy của mình hoặc liên hệ hỗ trợ dựng thông qua discord của tôi.

## Các tính năng của dự án

- Tách nhạc (MDX-Net / Demucs / VR)

- Chuyển đổi giọng nói (Chuyển đổi phi thời gian thực / Chuyển đổi hàng loạt / Chuyển đổi với Whisper / Chuyển đổi văn bản / Chuyển đổi thời gian thực)

- Áp dụng hiệu ứng cho âm thanh

- Tạo dữ liệu huấn luyện (Từ đường dẫn liên kết)

- Huấn luyện mô hình (v1 / v2, bộ mã hóa chất lượng cao, huấn luyện năng lượng)

- Chuyển đổi mô hình RVC sang mô hình ONNX

- Tìm kiếm mô hình từ web

- Tạo bộ tham chiếu huấn luyện

**Phương thức trích xuất cao độ (38+): `pm, dio, crepe, fcpe, rmvpe, hpa-rmvpe, harvest, yin, pyin, swipe, piptrack, penn, djcm, swift, pesto`**

**Các mô hình trích xuất nhúng (21+): `contentvec_base, hubert_base, vietnamese_hubert_base, japanese_hubert_base, korean_hubert_base, chinese_hubert_base, portuguese_hubert_base, spin, whisper`**

- **Trích xuất nhúng hỗ trợ những mô hình từ: `fairseq (.pt)`, `onnx (.onnx)`, `transformers (.bin - .json)`, `spin (.bin - .json)`, `whisper (.pt)`.**
- **Trích xuất cao độ hỗ trợ việc trộn phương thức với nhau để cải thiện chất lượng, ví dụ: `hybrid[rmvpe+harvest]`.**
- **Bộ mã hóa giọng nói tùy chỉnh: `HiFiGAN`, `NSF-HiFiGAN`, `MRF-HiFiGAN`, `ReFineGAN`, `BigVGAN`.**

Lưu ý:
- Mô hình trích xuất cao độ có thể khiến chất lượng đầu ra kém đi khi sử dụng trong sai môi trường, khuyên dùng nhất là "RMVPE".
- Mô hình trích xuất nhúng contentvec_base và hubert_base là một chỉ khác nhau về mặt dung lượng và độ chính xác khi suy luận của nó.
- Việc thay đổi mô hình nhúng đòi hỏi việc huấn luyện lại từ đầu hoàn toàn mô hình RVC để sử dụng được. Các mô hình RVC thông dụng hiện tại sử dụng bộ nhúng contentvec_base hoặc hubert_base.
- Khi huấn luyện mô hình RVC, để đạt được chất lượng cao nhất khi huấn luyện cần phải cung cấp dữ liệu âm thanh sạch không tiếng ồn, không bị nhiễu hay dính tạp âm, giọng rõ ràng và càng nhiều âm thanh càng tốt yêu cầu từ 10 phút giọng trở lên.
- Khi huấn luyện nếu bạn không biết xem các giá trị thất thoát huấn luyện từ tensorboard, bạn có thể huấn luyện tầm 300-500 kỷ nguyên là tốt nhất, nhiều quá có thể khiến mô hình trở nên máy móc hơn.

## Cài đặt

Bước 1: Cài đặt các phần phụ trợ cần thiết

- Cài đặt Python từ trang chủ: **[PYTHON](https://www.python.org/ftp/python/3.11.8/python-3.11.8-amd64.exe)** (Dự án đã được kiểm tra trên Python 3.10.x và 3.11.x)
- Cài đặt FFmpeg từ nguồn và thêm vào PATH hệ thống: **[FFMPEG](https://github.com/BtbN/FFmpeg-Builds/releases)**

Bước 2: Cài đặt dự án (Dùng Git hoặc đơn giản là tải trên github)

Sử dụng đối với Git:
- git clone https://github.com/PhamHuynhAnh16/Vietnamese-RVC.git
- cd Vietnamese-RVC

Cài đặt bằng github:
- Vào https://github.com/PhamHuynhAnh16/Vietnamese-RVC
- Nhấn vào `<> Code` màu xanh lá chọn `Download ZIP`
- Giải nén `Vietnamese-RVC-main.zip`
- Vào thư mục Vietnamese-RVC-main chọn vào thanh đường dẫn nhập `cmd` và nhấn Enter

Bước 3: Cài đặt thư viện cần thiết:

<details>
<summary>Đối với Windows</summary>

Nhập lệnh:
```
python -m venv env
env\Scripts\python.exe -m pip install uv
env\Scripts\python.exe -m uv pip install six packaging python-dateutil platformdirs pywin32 onnxconverter_common wget
```

Tiếp tục chạy các lệnh dưới đây theo phiên bản phần cứng của bạn.

<details>
<summary>Đối với CPU (Sử dụng CPU cho việc tính toán)</summary>

```
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

</details>

<details>
<summary>Đối với CUDA (Sử dụng đối với GPU của Nvidia)</summary>

Có thể thay cu118 thành bản cu128 mới hơn nếu GPU hỗ trợ:
```
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

Lưu ý:
- Nếu gặp lỗi liên quan đến Pytorch hãy thử gỡ cài đặt Pytorch hiện tại và cài đặt phiên bản Pytorch cũ hơn ví dụ như cu121.
```
env\Scripts\python.exe -m pip uninstall -y torch torchaudio torchvision
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

- Nếu gặp lỗi liên quan đến Onnxruntime hãy thử gỡ cài đặt Onnxruntime hiện tại và cài đặt phiên bản Pytorch cũ hơn ví dụ 1.20.1. 
```
env\Scripts\python.exe -m pip uninstall -y onnxruntime-gpu
env\Scripts\python.exe -m uv pip install onnxruntime-gpu==1.20.1
```

- Nếu bạn không biết cách phân biệt giữa hai lỗi, hãy chạy cả hai lệnh trên:D

</details>

<details>
<summary>Đối với OPENCL (Sử dụng đối với GPU của AMD)</summary>

```
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch==2.6.0 torchaudio==2.6.0 torchvision
env\Scripts\python.exe -m uv pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp311-none-win_amd64.whl
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

Lưu ý: 
- Có vẻ như OPENCL đã không còn được hỗ trợ tiếp.
- Chỉ nên cài đặt trên python 3.11 do không có bản biên dịch cho python 3.10 với torch 2.6.0.
- Demucs có thể gây quá tải và tràn bộ nhớ đối với GPU (nếu cần sử dụng demucs hãy mở tệp config.json trong main\configs sửa đối số demucs_cpu_mode thành true).
- DDP không hỗ trợ huấn luyện đa GPU đối với OPENCL.
- Một số thuật toán khác phải chạy trên cpu nên có thể hiệu suất của GPU có thể không sử dụng hết.

</details>

<details>
<summary>Đối với DIRECTML (Sử dụng đối với GPU của AMD / Intel Graphics)</summary>

```
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch==2.4.1 torchaudio==2.4.1 torchvision
env\Scripts\python.exe -m uv pip install torch-directml==0.2.5.dev240914
env\Scripts\python.exe -m uv pip install onnxruntime-directml
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

Lưu ý: 
- Directml đã ngừng phát triển một khoảng thời gian dài.
- Directml không hỗ trợ quá tốt tác vụ đa luồng nên khi chạy trích xuất thường sẽ bị khóa ở 1 luồng.
- Directml có hỗ trợ 1 phần fp16 nhưng không được khuyến khích sử dụng vì có thể chỉ nhận được hiệu năng tương đương fp32.
- Directml không có hàm để dọn dẹp bộ nhớ, tôi đã tạo 1 hàm đơn giản để dọn dẹp bộ nhớ nhưng có thể sẽ không quá hiệu quả.
- Directml được thiết kế để suy luận chứ không phải dùng để huấn luyện mặc dù có thể hoàn toàn chạy được huấn luyện nhưng sẽ không được khuyến khích.

</details>

<details>
<summary>Đối với ZLUDA (Sử dụng đối với GPU của AMD hỗ trợ ROCm)</summary>

Kiểm tra GPU của bạn có được hỗ trợ hay không: [ROCM-Requirements](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/reference/system-requirements.html).
Tải và cài đặt: [VC++ Runtime](https://aka.ms/vs/17/release/vc_redist.x64.exe) và [HIP-SDK](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html).
Thêm thư mục bin từ HIP-SDK vào PATH hệ thống.

```
env\Scripts\python.exe -m uv pip install numpy==1.26.4 numba==0.61.0
env\Scripts\python.exe -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
env\Scripts\python.exe -m uv pip install -r requirements.txt
```

Sao chép path-zluda-hipxx.bat (Thay xx theo phiên bản HIP SDK của bạn) và run_app.bat ra thư mục chính thay thế các tệp hiện tại.
Chạy tệp path-zluda-hipxx.bat.

Lưu ý:
- Nếu GPU của bạn không nằm trong danh sách hỗ trợ (ví dụ như: gf803) bạn có thể thử dùng HIP SDK 5.7.0 và ghi đè thư mục library trong ROCm/5.7.0/bin/rocblas từ [ROCmLibs](https://github.com/likelovewant/ROCmLibs-for-gfx1103-AMD780M-APU/releases) (Không Khuyến Khích).
- Nếu Onnxruntime không hoạt động, bạn có thể thử đổi sang phiên bản onnxruntime-directml.
- Zluda không được thiết kế cho các hệ thống thời gian thực nên chức năng thời gian thực của ứng dụng sẽ bị vô hiệu hóa.

</details>
</details>

<details>
<summary>Đối với Linux</summary>

Nhập lệnh:
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
python -m uv pip install six packaging python-dateutil platformdirs onnxconverter_common wget
```

Tiếp tục chạy các lệnh dưới đây theo phiên bản phần cứng của bạn.

<details>
<summary>Đối với CPU (Sử dụng CPU cho việc tính toán)</summary>

```
python -m uv pip install -r requirements.txt
```

</details>

<details>
<summary>Đối với CUDA (Sử dụng đối với GPU của Nvidia)</summary>

Có thể thay cu118 thành bản cu128 mới hơn nếu GPU hỗ trợ:
```
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
python -m uv pip install -r requirements.txt
```

Lưu ý:
- Nếu gặp lỗi liên quan đến Pytorch hãy thử gỡ cài đặt Pytorch hiện tại và cài đặt phiên bản Pytorch cũ hơn ví dụ như cu121.
```
python -m pip uninstall -y torch torchaudio torchvision
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

- Nếu gặp lỗi liên quan đến Onnxruntime hãy thử gỡ cài đặt Onnxruntime hiện tại và cài đặt phiên bản Pytorch cũ hơn ví dụ 1.20.1. 
```
python -m pip uninstall -y onnxruntime-gpu
python -m uv pip install onnxruntime-gpu==1.20.1
```

- Nếu bạn không biết cách phân biệt giữa hai lỗi, hãy chạy cả hai lệnh trên:D

</details>

<details>
<summary>Đối với ROCm (Sử dụng đối với GPU của AMD)</summary>

Có thể thay rocm6.4 thành bản rocm7.1 mới hơn nếu GPU hỗ trợ:
```
python -m uv pip install numpy==1.26.4 numba==0.61.0
python -m uv pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/rocm6.4
python -m uv pip install -r requirements.txt
```

</details>
</details>

## Sử dụng

**Sử dụng với Google Colab**
- Mở Google Colab: [Vietnamese-RVC](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
- Bước 1: Chạy ô Cài đặt và đợi nó hoàn tất.
- Bước 2: Chạy ô Mở giao diện sử dụng (Khi này giao diện sẽ in ra 2 đường dẫn 1 là 0.0.0.0.7680 và 1 đường dẫn gradio có thể nhấp được, bạn chọn vào đường dẫn nhấp được và nó sẽ đưa bạn đến giao diện).

**(Windows) Chạy tệp run_app.bat để mở giao diện sử dụng, chạy tệp tensorboard.bat để mở biểu đồ kiểm tra huấn luyện. (Lưu ý: không tắt Command Prompt hoặc Terminal)**
```
run_app.bat / tensorboard.bat
```

**Khởi động giao diện sử dụng. (Thêm `--allow_all_disk` vào lệnh để cho phép gradio truy cập tệp ngoài)**
```
env\Scripts\python.exe main\app\app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
env\Scripts\python.exe main\app\run_tensorboard.py
```

**Sử dụng bằng cú pháp**
```
env\Scripts\python.exe main\app\parser.py --help
```

**(Linux) Chạy tệp run_app.sh để mở giao diện sử dụng, chạy tệp tensorboard.sh để mở biểu đồ kiểm tra huấn luyện. (Lưu ý: không tắt Terminal)**
```
run_app.sh / tensorboard.sh
```

**Khởi động giao diện sử dụng. **
```
python main/app/app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
python main/app/run_tensorboard.py
```

**Sử dụng bằng cú pháp**
```
python main/app/parser.py --help
```

## LƯU Ý

- **Hiện tại các bộ mã hóa mới như MRF HIFIGAN và REFINEGAN vẫn chưa đầy đủ các bộ huấn luyện trước**
- **Bộ mã hóa MRF HIFIGAN và REFINEGAN không hỗ trợ huấn luyện khi không không huấn luyện cao độ**
- **Huấn luyện năng lương có thể cải thiện chất lượng mô hình nhưng chưa có mô hình huấn luyện trước dành cho tính năng này**
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

|                                                            Tác Phẩm                                                            |         Tác Giả         |  Giấy Phép  |
|--------------------------------------------------------------------------------------------------------------------------------|-------------------------|-------------|
| **[Applio](https://github.com/IAHispano/Applio/tree/main)**                                                                    | IAHispano               | MIT License |
| **[Python-audio-separator](https://github.com/nomadkaraoke/python-audio-separator/tree/main)**                                 | Nomad Karaoke           | MIT License |
| **[Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/tree/main)**  | RVC Project             | MIT License |
| **[RVC-ONNX-INFER-BY-Anh](https://github.com/PhamHuynhAnh16/RVC_Onnx_Infer)**                                                  | Phạm Huỳnh Anh          | MIT License |
| **[Torch-Onnx-Crepe-By-Anh](https://github.com/PhamHuynhAnh16/TORCH-ONNX-CREPE)**                                              | Phạm Huỳnh Anh          | MIT License |
| **[Hubert-No-Fairseq](https://github.com/PhamHuynhAnh16/hubert-no-fairseq)**                                                   | Phạm Huỳnh Anh          | MIT License |
| **[Local-attention](https://github.com/lucidrains/local-attention)**                                                           | Phil Wang               | MIT License |
| **[TorchFcpe](https://github.com/CNChTu/FCPE/tree/main)**                                                                      | CN_ChiTu                | MIT License |
| **[ContentVec](https://github.com/auspicious3000/contentvec)**                                                                 | Kaizhi Qian             | MIT License |
| **[Mediafiredl](https://github.com/Gann4Life/mediafiredl)**                                                                    | Santiago Ariel Mansilla | MIT License |
| **[Noisereduce](https://github.com/timsainb/noisereduce)**                                                                     | Tim Sainburg            | MIT License |
| **[World.py-By-Anh](https://github.com/PhamHuynhAnh16/world.py)**                                                              | Phạm Huỳnh Anh          | MIT License |
| **[Mega.py](https://github.com/3v1n0/mega.py)**                                                                                | Marco Trevisan          | No License  |
| **[Gdown](https://github.com/wkentaro/gdown)**                                                                                 | Kentaro Wada            | MIT License |
| **[Whisper](https://github.com/openai/whisper)**                                                                               | OpenAI                  | MIT License |
| **[PyannoteAudio](https://github.com/pyannote/pyannote-audio)**                                                                | pyannote                | MIT License |
| **[StftPitchShift](https://github.com/jurihock/stftPitchShift)**                                                               | Jürgen Hock             | MIT License |
| **[Penn](https://github.com/interactiveaudiolab/penn)**                                                                        | Interactive Audio Lab   | MIT License |
| **[Voice Changer](https://github.com/deiteris/voice-changer)**                                                                 | Yury deiteris           | MIT License |
| **[Pesto](https://github.com/SonyCSLParis/pesto)**                                                                             | Sony CSL Paris          | LGPL 3.0    |

## Kho mô hình của công cụ tìm kiếm mô hình

- **[VOICE-MODELS.COM](https://voice-models.com/)**

## Báo cáo lỗi
- **Với trường hợp hệ thống báo cáo lỗi không hoạt động bạn có thể báo cáo lỗi cho tôi thông qua Discord `pham_huynh_anh` Hoặc [ISSUE](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues)**

## ☎️ Liên hệ tôi
- Discord: **pham_huynh_anh**