<div align="center">
<img alt="LOGO" src="assets/ico.png" width="300" height="300" />

# Vietnamese RVC BY ANH
Công cụ chuyển đổi giọng nói chất lượng và hiệu suất cao đơn giản.

[![Vietnamese RVC](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC)
[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
[![Licence](https://img.shields.io/github/license/saltstack/salt?style=for-the-badge)](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/blob/main/LICENSE)

</div>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/AnhP/RVC-GUI)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-blue)](https://huggingface.co/AnhP/Vietnamese-RVC-Project)

</div>

# Mô tả

Dự án này là một công cụ chuyển đổi giọng nói đơn giản, dễ sử dụng. Với mục tiêu tạo ra các sản phẩm chuyển đổi giọng nói chất lượng cao và hiệu suất tối ưu, dự án cho phép người dùng thay đổi giọng nói một cách mượt mà, tự nhiên.

# Các tính năng của dự án

- Tách nhạc (MDX-Net/Demucs)

- Chuyển đổi giọng nói (Chuyển đổi tệp / Chuyển đổi hàng loạt / Chuyển đổi với Whisper / Chuyển đổi văn bản)

- Chỉnh sửa nhạc nền

- Áp dụng hiệu ứng cho âm thanh

- Tạo dữ liệu huấn luyện (Từ đường dẫn liên kết)

- Huấn luyện mô hình (v1/v2, bộ mã hóa chất lượng cao)

- Dung hợp mô hình

- Đọc thông tin mô hình

- Xuất mô hình sang ONNX

- Tải xuống từ kho mô hình có sẳn

- Tìm kiếm mô hình từ web

- Trích xuất cao độ

- Hỗ trợ suy luận chuyển đổi âm thanh bằng mô hình ONNX

- Mô hình ONNX RVC cũng sẽ hỗ trợ chỉ mục để suy luận

**Phương thức trích xuất cao độ: `pm, dio, mangio-crepe-tiny, mangio-crepe-small, mangio-crepe-medium, mangio-crepe-large, mangio-crepe-full, crepe-tiny, crepe-small, crepe-medium, crepe-large, crepe-full, fcpe, fcpe-legacy, rmvpe, rmvpe-legacy, harvest, yin, pyin, swipe`**

**Các mô hình trích xuất nhúng: `contentvec_base, hubert_base, japanese_hubert_base, korean_hubert_base, chinese_hubert_base, portuguese_hubert_base, spin`**

- **Các mô hình trích xuất cao độ đều có phiên bản tăng tốc ONNX trừ các phương thức hoạt động bằng trình bao bọc.** 
- **Các mô hình trích xuất đều có thể kết hợp với nhau để tạo ra cảm giác mới mẻ, ví dụ: `hybrid[rmvpe+harvest]`.**
- **Các mô hình trích xuất nhúng có sẳn các chế độ nhúng như: fairseq, onnx, transformers, spin.**

# Hướng dẫn sử dụng

**Sẽ có nếu tôi thực sự rảnh...**

# Cài đặt

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
- Vào thư mục Vietnamese-RVC-main chọn vào thanh Path nhập `cmd` và nhấn Enter

Bước 3: Cài đặt thư viện cần thiết:

Nhập lệnh:
```
python -m venv env
env\\Scripts\\activate
```

Đối với CPU:
```
python -m pip install -r requirements.txt
```

Đối với CUDA (Có thể thay cu118 thành bản cu128 mới hơn nếu GPU hỗ trợ):
```
python -m pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements.txt
```

Đối với AMD:
```
python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision
python -m pip install https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.6-cp311-none-win_amd64.whl
python -m pip install onnxruntime-directml
python -m pip install -r requirements.txt
```

Lưu ý đối với AMD: 
- Chỉ cài đặt AMD trên python 3.11 vì DLPRIM không có bản cho python 3.10.
- RMVPE và Whisper phải chạy trên cpu vì có một số thuật toán không được hỗ trợ.
- Demucs có thể gây quá tải và tràn bộ nhớ đối với GPU (nếu cần sử dụng demucs hãy mở tệp config.json trong main\configs sửa đối số demucs_cpu_mode thành true).
- DDP không hỗ trợ huấn luyện đa GPU đối với OPENCL (AMD).
- Một số thuật toán khác phải chạy trên cpu nên có thể hiệu suất của GPU có thể không sử dụng hết.

# Sử dụng

**Sử dụng với Google Colab**
- Mở Google Colab: [Vietnamese-RVC](https://colab.research.google.com/github/PhamHuynhAnh16/Vietnamese-RVC-ipynb/blob/main/Vietnamese-RVC.ipynb)
- Bước 1: Chạy ô Cài đặt và đợi nó hoàn tất.
- Bước 2: Chạy ô Mở giao diện sử dụng (Khi này giao diện sẽ in ra 2 đường dẫn 1 là 0.0.0.0.7680 và 1 đường dẫn gradio có thể nhấp được, bạn chọn vào đường dẫn nhấp được và nó sẽ đưa bạn đến giao diện).

**Chạy tệp run_app để mở giao diện sử dụng, chạy tệp tensorboard để mở biểu đồ kiểm tra huấn luyện. (Lưu ý: không tắt Command Prompt hoặc Terminal)**
```
run_app.bat / tensorboard.bat
```

**Khởi động giao diện sử dụng. (Thêm `--allow_all_disk` vào lệnh để cho phép gradio truy cập tệp ngoài)**
```
env\\Scripts\\python.exe main\\app\\app.py --open
```

**Với trường hợp bạn sử dụng Tensorboard để kiểm tra huấn luyện**
```
env\\Scripts\\python.exe main/app/run_tensorboard.py
```

**Sử dụng bằng cú pháp**
```
python main\\app\\parser.py --help
```

# Cài đặt, sử dụng đơn giản

**Cài đặt phiên bản releases từ [Vietnamese_RVC](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/releases)**
- Chọn bản đúng với bạn và tải về máy.
- Giải nén dự án.
- Chạy tệp run_app.bat để mở giao diện hoạt động.

# Cấu trúc chính của mã nguồn:

<pre>
Vietnamese-RVC-main
├── assets
│   ├── binary
│   │   ├── decrypt.bin
│   │   └── world.bin
│   ├── f0
│   ├── languages
│   │   ├── en-US.json
│   │   └── vi-VN.json
│   ├── logs
│   │   └── mute
│   │       ├── f0
│   │       │   └── mute.wav.npy
│   │       ├── f0_voiced
│   │       │   └── mute.wav.npy
│   │       ├── sliced_audios
│   │       │   ├── mute32000.wav
│   │       │   ├── mute40000.wav
│   │       │   └── mute48000.wav
│   │       ├── sliced_audios_16k
│   │       │   └── mute.wav
│   │       ├── v1_extracted
│   │       │   ├── mute.npy
│   │       │   └── mute_spin.npy
│   │       └── v2_extracted
│   │           ├── mute.npy
│   │           └── mute_spin.npy
│   ├── models
│   │   ├── audioldm2
│   │   ├── embedders
│   │   ├── predictors
│   │   ├── pretrained_custom
│   │   ├── pretrained_v1
│   │   ├── pretrained_v2
│   │   ├── speaker_diarization
│   │   │   ├── assets
│   │   │   │   ├── gpt2.tiktoken
│   │   │   │   ├── mel_filters.npz
│   │   │   │   └── multilingual.tiktoken
│   │   │   └── models
│   │   └── uvr5
│   ├── presets
│   ├── weights
│   └── ico.png
├── audios
├── dataset
├── main
│   ├── app
│   │   ├── core
│   │   │   ├── downloads.py
│   │   │   ├── editing.py
│   │   │   ├── f0_extract.py
│   │   │   ├── inference.py
│   │   │   ├── model_utils.py
│   │   │   ├── presets.py
│   │   │   ├── process.py
│   │   │   ├── restart.py
│   │   │   ├── separate.py
│   │   │   ├── training.py
│   │   │   ├── tts.py
│   │   │   ├── ui.py
│   │   │   └── utils.py
│   │   ├── tabs
│   │   │   ├── downloads
│   │   │   │   └── downloads.py
│   │   │   ├── editing
│   │   │   │   ├── editing.py
│   │   │   │   └── child
│   │   │   │       ├── audio_editing.py
│   │   │   │       ├── audio_effects.py
│   │   │   │       └── quirk.py
│   │   │   ├── extra
│   │   │   │   ├── extra.py
│   │   │   │   └── child
│   │   │   │       ├── convert_model.py
│   │   │   │       ├── f0_extract.py
│   │   │   │       ├── fushion.py
│   │   │   │       ├── read_model.py
│   │   │   │       ├── report_bugs.py
│   │   │   │       └── settings.py
│   │   │   ├── inference
│   │   │   │   ├── inference.py
│   │   │   │   └── child
│   │   │   │       ├── convert.py
│   │   │   │       ├── convert_tts.py
│   │   │   │       ├── convert_with_whisper.py
│   │   │   │       └── separate.py
│   │   │   └── training
│   │   │       ├── training.py
│   │   │       └── child
│   │   │           ├── create_dataset.py
│   │   │           └── training.py
│   │   ├── app.py
│   │   ├── parser.py
│   │   ├── run_tensorboard.py
│   │   └── variables.py
│   ├── configs
│   │   ├── config.json
│   │   ├── config.py
│   │   ├── v1
│   │   │   ├── 32000.json
│   │   │   ├── 40000.json
│   │   │   └── 48000.json
│   │   └── v2
│   │       ├── 32000.json
│   │       ├── 40000.json
│   │       └── 48000.json
│   ├── inference
│   │   ├── audioldm2.py
│   │   ├── audio_effects.py
│   │   ├── create_dataset.py
│   │   ├── create_index.py
│   │   ├── extract.py
│   │   ├── separator_music.py
│   │   ├── training
│   │   │   ├── train.py
│   │   │   ├── data_utils.py
│   │   │   ├── losses.py
│   │   │   ├── mel_processing.py
│   │   │   └── utils.py
│   │   ├── conversion
│   │   │   ├── convert.py
│   │   │   ├── pipeline.py
│   │   │   └── utils.py
│   │   └── preprocess
│   │       ├── preprocess.py
│   │       └── slicer2.py
│   ├── library
│   │   ├── utils.py
│   │   ├── torch_amd.py
│   │   ├── algorithm
│   │   │   ├── attentions.py
│   │   │   ├── commons.py
│   │   │   ├── discriminators.py
│   │   │   ├── encoders.py
│   │   │   ├── modules.py
│   │   │   ├── normalization.py
│   │   │   ├── onnx_export.py
│   │   │   ├── residuals.py
│   │   │   ├── stftpitchshift.py
│   │   │   └── synthesizers.py
│   │   ├── architectures
│   │   │   ├── demucs_separator.py
│   │   │   ├── fairseq.py
│   │   │   └── mdx_separator.py
│   │   ├── audioldm2
│   │   │   ├── models.py
│   │   │   └── utils.py
│   │   ├── generators
│   │   │   ├── hifigan.py
│   │   │   ├── mrf_hifigan.py
│   │   │   ├── nsf_hifigan.py
│   │   │   └── refinegan.py
│   │   ├── predictors
│   │   │   ├── CREPE.py
│   │   │   ├── FCPE
│   │   │   │   ├── attentions.py
│   │   │   │   ├── encoder.py
│   │   │   │   ├── FCPE.py
│   │   │   │   ├── stft.py
│   │   │   │   ├── utils.py
│   │   │   │   └── wav2mel.py
│   │   │   ├── Generator.py
│   │   │   ├── RMVPE.py
│   │   │   ├── SWIPE.py
│   │   │   └── WORLD.py
│   │   ├── speaker_diarization
│   │   │   ├── audio.py
│   │   │   ├── ECAPA_TDNN.py
│   │   │   ├── embedding.py
│   │   │   ├── encoder.py
│   │   │   ├── features.py
│   │   │   ├── parameter_transfer.py
│   │   │   ├── segment.py
│   │   │   ├── speechbrain.py
│   │   │   └── whisper.py
│   │   └── uvr5_separator
│   │       ├── common_separator.py
│   │       ├── separator.py
│   │       ├── spec_utils.py
│   │       └── demucs
│   │           ├── apply.py
│   │           ├── demucs.py
│   │           ├── hdemucs.py
│   │           ├── htdemucs.py
│   │           ├── states.py
│   │           └── utils.py
│   └── tools
│       ├── gdown.py
│       ├── huggingface.py
│       ├── mediafire.py
│       ├── meganz.py
│       ├── noisereduce.py
│       └── pixeldrain.py
├── docker-compose-cpu.yaml
├── docker-compose-cuda118.yaml
├── docker-compose-cuda128.yaml
├── Dockerfile
├── Dockerfile.cuda118
├── Dockerfile.cuda128
├── LICENSE
├── README.md
├── requirements.txt
├── run_app.bat
└── tensorboard.bat
</pre>

# LƯU Ý

- **Hiện tại các bộ mã hóa mới như MRF HIFIGAN vẫn chưa đầy đủ các bộ huấn luyện trước**
- **Bộ mã hóa MRF HIFIGAN và REFINEGAN không hỗ trợ huấn luyện khi không không huấn luyện cao độ**
- **Các mô hình trong kho lưu trữ Vietnamese-RVC được thu thập rải rác trên AI Hub, HuggingFace và các các kho lưu trữ khác. Có thể mang các giấy phép bản quyền khác nhau (Ví dụ: Audioldm2 có các trọng số mô hình với điều khoản "Phi Thương Mại")**
- **Mã nguồn này có chứa thành phần phần mềm bên thứ ba được cấp phép với điều khoản "phi thương mại". Bất kỳ hành vi sử dụng thương mại nào, bao gồm kêu gọi tài trợ hoặc tài chính hóa phần mềm phái sinh, đều có thể vi phạm giấy phép và sẽ phải chịu trách nhiệm pháp lý tương ứng.**

# Tuyên bố miễn trừ trách nhiệm

- **Dự án Vietnamese-RVC được phát triển với mục đích nghiên cứu, học tập và giải trí cá nhân. Tôi không khuyến khích cũng như không chịu trách nhiệm đối với bất kỳ hành vi lạm dụng công nghệ chuyển đổi giọng nói vì mục đích lừa đảo, giả mạo danh tính, hoặc vi phạm quyền riêng tư, bản quyền của bất kỳ cá nhân hay tổ chức nào.**

- **Người dùng cần tự chịu trách nhiệm với hành vi sử dụng phần mềm này và cam kết tuân thủ pháp luật hiện hành tại quốc gia nơi họ sinh sống hoặc hoạt động.**

- **Việc sử dụng giọng nói của người nổi tiếng, người thật hoặc nhân vật công chúng phải có sự cho phép hoặc đảm bảo không vi phạm pháp luật, đạo đức và quyền lợi của các bên liên quan.**

- **Tác giả của dự án không chịu trách nhiệm pháp lý đối với bất kỳ hậu quả nào phát sinh từ việc sử dụng phần mềm này.**

# Điều khoản sử dụng

- Bạn phải đảm bảo rằng các nội dung âm thanh bạn tải lên và chuyển đổi qua dự án này không vi phạm quyền sở hữu trí tuệ của bên thứ ba.

- Không được phép sử dụng dự án này cho bất kỳ hoạt động nào bất hợp pháp, bao gồm nhưng không giới hạn ở việc sử dụng để lừa đảo, quấy rối, hay gây tổn hại đến người khác.

- Bạn chịu trách nhiệm hoàn toàn đối với bất kỳ thiệt hại nào phát sinh từ việc sử dụng sản phẩm không đúng cách.

- Tôi sẽ không chịu trách nhiệm với bất kỳ thiệt hại trực tiếp hoặc gián tiếp nào phát sinh từ việc sử dụng dự án này.

# Dự án này được xây dựng dựa trên các dự án như sau

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
| **[FcpeONNX](https://github.com/deiteris/voice-changer/blob/master-custom/server/utils/fcpe_onnx.py)**                         | Yury                    | MIT License |
| **[ContentVec](https://github.com/auspicious3000/contentvec)**                                                                 | Kaizhi Qian             | MIT License |
| **[Mediafiredl](https://github.com/Gann4Life/mediafiredl)**                                                                    | Santiago Ariel Mansilla | MIT License |
| **[Noisereduce](https://github.com/timsainb/noisereduce)**                                                                     | Tim Sainburg            | MIT License |
| **[World.py-By-Anh](https://github.com/PhamHuynhAnh16/world.py)**                                                              | Phạm Huỳnh Anh          | MIT License |
| **[Mega.py](https://github.com/3v1n0/mega.py)**                                                                                | Marco Trevisan          | No License  |
| **[Gdown](https://github.com/wkentaro/gdown)**                                                                                 | Kentaro Wada            | MIT License |
| **[Whisper](https://github.com/openai/whisper)**                                                                               | OpenAI                  | MIT License |
| **[PyannoteAudio](https://github.com/pyannote/pyannote-audio)**                                                                | pyannote                | MIT License |
| **[AudioEditingCode](https://github.com/HilaManor/AudioEditingCode)**                                                          | Hila Manor              | MIT License |
| **[StftPitchShift](https://github.com/jurihock/stftPitchShift)**                                                               | Jürgen Hock             | MIT License |
| **[Codename-RVC-Fork-3](https://github.com/codename0og/codename-rvc-fork-3)**                                                  | Codename;0              | MIT License |

# Kho mô hình của công cụ tìm kiếm mô hình

- **[VOICE-MODELS.COM](https://voice-models.com/)**

# Các phương pháp trích xuất Pitch trong RVC

Tài liệu này trình bày chi tiết các phương pháp trích xuất cao độ được sử dụng, thông tin về ưu, nhược điểm, sức mạnh và độ tin cậy của từng phương pháp theo trải nghiệm cá nhân.

| Phương pháp        |      Loại      |          Ưu điểm          |            Hạn chế           |      Sức mạnh      |     Độ tin cậy     |
|--------------------|----------------|---------------------------|------------------------------|--------------------|--------------------|
| pm                 | Praat          | Nhanh                     | Kém chính xác                | Thấp               | Thấp               |
| dio                | PYWORLD        | Thích hợp với Rap         | Kém chính xác với tần số cao | Trung bình         | Trung bình         |
| harvest            | PYWORLD        | Chính xác hơn DIO         | Xử lý chậm hơn               | Cao                | Rất cao            |
| crepe              | Deep Learning  | Chính xác cao             | Yêu cầu GPU                  | Rất cao            | Rất cao            |
| mangio-crepe       | crepe finetune | Tối ưu hóa cho RVC        | Đôi khi kém crepe gốc        | Trung bình đến cao | Trung bình đến cao |
| fcpe               | Deep Learning  | Chính xác, thời gian thực | Cần GPU mạnh                 | Khá                | Trung bình         |
| fcpe-legacy        | Old            | Chính xác, thời gian thực | Cũ hơn                       | Khá                | Trung bình         |
| rmvpe              | Deep Learning  | Hiệu quả với giọng hát    | Tốn tài nguyên               | Rất cao            | Xuất sắc           |
| rmvpe-legacy       | Old            | Hỗ trợ hệ thống cũ        | Cũ hơn                       | Cao                | Khá                |
| yin                | Librosa        | Đơn giản, hiệu quả        | Dễ lỗi bội                   | Trung bình         | Thấp               |
| pyin               | Librosa        | Ổn định hơn YIN           | Tính toán phức tạp hơn       | Khá                | Khá                |
| swipe              | WORLD          | Độ chính xác cao          | Nhạy cảm với nhiễu           | Cao                | Khá                |

# Báo cáo lỗi

- **Với trường hợp gặp lỗi khi sử dụng mã nguồn này tôi thực sự xin lỗi bạn vì trải nghiệm không tốt này, bạn có thể gửi báo cáo lỗi thông qua cách phía dưới**
- **Bạn có thể báo cáo lỗi cho tôi thông qua hệ thống báo cáo lỗi webhook trong giao diện sử dụng**
- **Với trường hợp hệ thống báo cáo lỗi không hoạt động bạn có thể báo cáo lỗi cho tôi thông qua Discord `pham_huynh_anh` Hoặc [ISSUE](https://github.com/PhamHuynhAnh16/Vietnamese-RVC/issues)**

# ☎️ Liên hệ tôi
- Discord: **pham_huynh_anh**