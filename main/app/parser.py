import os
import sys

sys.path.append(os.getcwd())

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"

try:
    argv = sys.argv[1]
except IndexError:
    argv = None

argv_is_allows = ["--audio_effects", "--audioldm2", "--convert", "--create_dataset", "--create_index", "--extract", "--preprocess", "--separator_music", "--train", "--help_audio_effects", "--help_audioldm2", "--help_convert", "--help_create_dataset", "--help_create_index", "--help_extract", "--help_preprocess", "--help_separator_music",  "--help_train", "--help"]

if argv not in argv_is_allows:
    print("Cú pháp không hợp lệ! Sử dụng --help để biết thêm")
    quit()

if argv_is_allows[0] in argv: from main.inference.audio_effects import main
elif argv_is_allows[1] in argv: from main.inference.audioldm2 import main
elif argv_is_allows[2] in argv: from main.inference.conversion.convert import main
elif argv_is_allows[3] in argv: from main.inference.create_dataset import main
elif argv_is_allows[4] in argv: from main.inference.create_index import main
elif argv_is_allows[5] in argv: from main.inference.extract import main
elif argv_is_allows[6] in argv: from main.inference.preprocess.preprocess import main
elif argv_is_allows[7] in argv: from main.inference.separator_music import main
elif argv_is_allows[8] in argv: from main.inference.training.train import main
elif argv_is_allows[9] in argv:
    print("""Các tham số của `--audio_effects`:
        1. Đường dẫn tệp:
            - `--input_path` (bắt buộc): Đường dẫn đến tệp âm thanh đầu vào.
            - `--output_path` (mặc định: `./audios/apply_effects.wav`): Đường dẫn lưu tệp đầu ra.
            - `--export_format` (mặc định: `wav`): Định dạng xuất tệp (`wav`, `mp3`, ...).

        2. Lấy mẫu lại:
            - `--resample` (mặc định: `False`): Có lấy mẫu lại hay không.
            - `--resample_sr` (mặc định: `0`): Tần số lấy mẫu mới (Hz).

        3. Hiệu ứng chorus:
            - `--chorus`: Bật/tắt chorus.
            - `--chorus_depth`, `--chorus_rate`, `--chorus_mix`, `--chorus_delay`, `--chorus_feedback`: Các thông số điều chỉnh chorus.

        4. Hiệu ứng distortion:
            - `--distortion`: Bật/tắt distortion.
            - `--drive_db`: Mức độ méo âm thanh.

        5. Hiệu ứng reverb:
            - `--reverb`: Bật/tắt hồi âm.
            - `--reverb_room_size`, `--reverb_damping`, `--reverb_wet_level`, `--reverb_dry_level`, `--reverb_width`, `--reverb_freeze_mode`: Điều chỉnh hồi âm.

        6. Hiệu ứng pitch shift:
            - `--pitchshift`: Bật/tắt thay đổi cao độ.
            - `--pitch_shift`: Giá trị dịch cao độ.

        7. Hiệu ứng delay:
            - `--delay`: Bật/tắt delay.
            - `--delay_seconds`, `--delay_feedback`, `--delay_mix`: Điều chỉnh thời gian trễ, phản hồi và hòa trộn.

        8. Compressor:
            - `--compressor`: Bật/tắt compressor.
            - `--compressor_threshold`, `--compressor_ratio`, `--compressor_attack_ms`, `--compressor_release_ms`: Các thông số nén.

        9. Limiter:
            - `--limiter`: Bật/tắt giới hạn mức âm thanh.
            - `--limiter_threshold`, `--limiter_release`: Ngưỡng giới hạn và thời gian nhả.

        10. Gain (Khuếch đại):
            - `--gain`: Bật/tắt gain.
            - `--gain_db`: Mức gain (dB).

        11. Bitcrush:
            - `--bitcrush`: Bật/tắt hiệu ứng giảm độ phân giải.
            - `--bitcrush_bit_depth`: Số bit của bitcrush.

        12. Clipping:
            - `--clipping`: Bật/tắt cắt âm thanh.
            - `--clipping_threshold`: Ngưỡng clipping.

        13. Phaser:
            - `--phaser`: Bật/tắt hiệu ứng phaser.
            - `--phaser_rate_hz`, `--phaser_depth`, `--phaser_centre_frequency_hz`, `--phaser_feedback`, `--phaser_mix`: Điều chỉnh hiệu ứng phaser.

        14. Boost bass & treble:
            - `--treble_bass_boost`: Bật/tắt tăng cường âm bass và treble.
            - `--bass_boost_db`, `--bass_boost_frequency`, `--treble_boost_db`, `--treble_boost_frequency`: Các thông số tăng bass và treble.

        15. Fade in & fade out:
            - `--fade_in_out`: Bật/tắt hiệu ứng fade.
            - `--fade_in_duration`, `--fade_out_duration`: Thời gian fade vào/ra.

        16. Kết hợp âm thanh:
            - `--audio_combination`: Bật/tắt ghép nhiều tệp âm thanh.
            - `--audio_combination_input`: Đường dẫn tệp âm thanh bổ sung.
            - `--main_volume`: Âm lượng của âm thanh chính.
            - `--combination_volume`:: Âm lượng của âm thanh cần kết hợp.
    """)
    quit()
elif argv_is_allows[10] in argv:
    print("""Các tham số của --audioldm2:
        1. Đường dẫn tệp:
            - `--input_path` (bắt buộc): Đường dẫn đến tệp âm thanh đầu vào.
            - `--output_path` (mặc định: `./output.wav`): Đường dẫn lưu tệp đầu ra.
            - `--export_format` (mặc định: `wav`): Định dạng xuất tệp.

        2. Cấu hình âm thanh:
            - `--sample_rate` (mặc định: `44100`): Tần số lấy mẫu (Hz).

        3. Cấu hình mô hình AudioLDM:
            - `--audioldm_model` (mặc định: `audioldm2-music`): Chọn mô hình AudioLDM để xử lý.

        4. Prompt hướng dẫn mô hình:
            - `--source_prompt` (mặc định: ``): Mô tả âm thanh nguồn.
            - `--target_prompt` (mặc định: ``): Mô tả âm thanh đích.

        5. Cấu hình thuật toán xử lý:
            - `--steps` (mặc định: `200`): Số bước xử lý trong quá trình tổng hợp âm thanh.
            - `--cfg_scale_src` (mặc định: `3.5`): Hệ số điều chỉnh hướng dẫn cho âm thanh nguồn.
            - `--cfg_scale_tar` (mặc định: `12`): Hệ số điều chỉnh hướng dẫn cho âm thanh đích.
            - `--t_start` (mặc định: `45`): Mức độ chỉnh sửa.

        6. Tối ưu hóa tính toán:
            - `--save_compute` (mặc định: `False`): Có bật chế độ tối ưu tính toán hay không.
    """)
    quit()
elif argv_is_allows[11] in argv:
    print("""Các tham số của --convert:
        1. Cấu hình xử lý giọng nói:
            - `--pitch` (mặc định: `0`): Điều chỉnh cao độ.
            - `--filter_radius` (mặc định: `3`): Độ mượt của đường F0.
            - `--index_rate` (mặc định: `0.5`): Tỷ lệ sử dụng chỉ mục giọng nói.
            - `--volume_envelope` (mặc định: `1`): Hệ số điều chỉnh biên độ âm lượng.
            - `--protect` (mặc định: `0.33`): Bảo vệ phụ âm.

        2. Cấu hình mẫu (frame hop):
            - `--hop_length` (mặc định: `64`): Bước nhảy khi xử lý âm thanh.

        3. Cấu hình F0:
            - `--f0_method` (mặc định: `rmvpe`): Phương pháp dự đoán F0 (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--f0_autotune` (mặc định: `False`): Có tự động điều chỉnh F0 hay không.
            - `--f0_autotune_strength` (mặc định: `1`): Cường độ hiệu chỉnh tự động F0.
            - `--f0_file` (mặc định: ``): Đường dẫn tệp F0 có sẵn.
            - `--f0_onnx` (mặc định: `False`): Có sử dụng phiên bản ONNX của F0 hay không.
            - `--proposal_pitch` (mặc định: `False`): Đề xuất cao độ thay vì điều chỉnh thủ công.

        4. Mô hình nhúng:
            - `--embedder_model` (mặc định: `contentvec_base`): Mô hình nhúng sử dụng.
            - `--embedders_mode` (mặc định: `fairseq`): Chế độ nhúng (`fairseq`, `transformers`, `onnx`).

        5. Đường dẫn tệp:
            - `--input_path` (bắt buộc): Đường dẫn tệp âm thanh đầu vào.
            - `--output_path` (mặc định: `./audios/output.wav`): Đường dẫn lưu tệp đầu ra.
            - `--export_format` (mặc định: `wav`): Định dạng xuất tệp.
            - `--pth_path` (bắt buộc): Đường dẫn đến tệp mô hình `.pth`.
            - `--index_path` (mặc định: `None`): Đường dẫn tệp chỉ mục (nếu có).

        6. Làm sạch âm thanh:
            - `--clean_audio` (mặc định: `False`): Có áp dụng làm sạch âm thanh không.
            - `--clean_strength` (mặc định: `0.7`): Mức độ làm sạch.

        7. Resampling & chia nhỏ âm thanh:
            - `--resample_sr` (mặc định: `0`): Tần số lấy mẫu mới (0 nghĩa là giữ nguyên).
            - `--split_audio` (mặc định: `False`): Có chia nhỏ audio trước khi xử lý không.

        8. Kiểm tra & tối ưu hóa:
            - `--checkpointing` (mặc định: `False`): Bật/tắt checkpointing để tiết kiệm RAM.

        9. Dịch formant:
            - `--formant_shifting` (mặc định: `False`): Có bật hiệu ứng dịch formant không.
            - `--formant_qfrency` (mặc định: `0.8`): Hệ số dịch formant theo tần số.
            - `--formant_timbre` (mặc định: `0.8`): Hệ số thay đổi màu sắc giọng.
    """)
    quit()
elif argv_is_allows[12] in argv:
    print("""Các tham số của --create_dataset:
        1. Đường dẫn & cấu hình dataset:
            - `--input_audio` (bắt buộc): Đường dẫn liên kết đến âm thanh (Liên kết Youtube, có thể dùng dấu `,` để dùng nhiều liên kết).
            - `--output_dataset` (mặc định: `./dataset`): Thư mục xuất dữ liệu đầu ra.
            - `--sample_rate` (mặc định: `44100`): Tần số lấy mẫu cho âm thanh.

        2. Làm sạch dữ liệu:
            - `--clean_dataset` (mặc định: `False`): Có áp dụng làm sạch dữ liệu hay không.
            - `--clean_strength` (mặc định: `0.7`): Mức độ làm sạch dữ liệu.

        3. Tách giọng & hiệu ứng:
            - `--separator_reverb` (mặc định: `False`): Có tách vang giọng không.
            - `--kim_vocal_version` (mặc định: `2`): Phiên bản mô hình Kim Vocal để tách (`1`, `2`).

        4. Cấu hình phân đoạn âm thanh:
            - `--overlap` (mặc định: `0.25`): Mức độ chồng lấn giữa các đoạn khi tách.
            - `--segments_size` (mặc định: `256`): Kích thước của từng phân đoạn.

        5. Cấu hình MDX (Music Demixing):
            - `--mdx_hop_length` (mặc định: `1024`): Bước nhảy MDX khi xử lý.
            - `--mdx_batch_size` (mặc định: `1`): Kích thước batch khi xử lý MDX.
            - `--denoise_mdx` (mặc định: `False`): Có áp dụng khử nhiễu khi tách bằng MDX không.

        6. Bỏ qua phần âm thanh:
            - `--skip` (mặc định: `False`): Có bỏ qua giây âm thanh nào không.
            - `--skip_start_audios` (mặc định: `0`): Thời gian (giây) cần bỏ qua ở đầu audio.
            - `--skip_end_audios` (mặc định: `0`): Thời gian (giây) cần bỏ qua ở cuối audio.
    """)
    quit()
elif argv_is_allows[13] in argv:
    print("""Các tham số của --create_index:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản (`v1`, `v2`).
            - `--index_algorithm` (mặc định: `Auto`): Thuật toán index sử dụng (`Auto`, `Faiss`, `KMeans`).
    """)
    quit()
elif argv_is_allows[14] in argv:
    print("""Các tham số của --extract:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản RVC (`v1`, `v2`).

        2. Cấu hình F0:
            - `--f0_method` (mặc định: `rmvpe`): Phương pháp dự đoán F0 (`pm`, `dio`, `mangio-crepe-tiny`, `mangio-crepe-small`, `mangio-crepe-medium`, `mangio-crepe-large`, `mangio-crepe-full`, `crepe-tiny`, `crepe-small`, `crepe-medium`, `crepe-large`, `crepe-full`, `fcpe`, `fcpe-legacy`, `rmvpe`, `rmvpe-legacy`, `harvest`, `yin`, `pyin`, `swipe`).
            - `--pitch_guidance` (mặc định: `True`): Có sử dụng hướng dẫn cao độ hay không.
            - `--f0_autotune` (mặc định: `False`): Có tự động điều chỉnh F0 hay không.
            - `--f0_autotune_strength` (mặc định: `1`): Cường độ hiệu chỉnh tự động F0.

        3. Cấu hình xử lý:
            - `--hop_length` (mặc định: `128`): Độ dài bước nhảy trong quá trình xử lý.
            - `--cpu_cores` (mặc định: `2`): Số lượng luồng CPU sử dụng.
            - `--gpu` (mặc định: `-`): Chỉ định GPU sử dụng (ví dụ: `0` cho GPU đầu tiên, `-` để tắt GPU).
            - `--sample_rate` (bắt buộc): Tần số lấy mẫu của âm thanh đầu vào.

        4. Cấu hình nhúng:
            - `--embedder_model` (mặc định: `contentvec_base`): Tên mô hình nhúng.
            - `--f0_onnx` (mặc định: `False`): Có sử dụng phiên bản ONNX của F0 hay không.
            - `--embedders_mode` (mặc định: `fairseq`): Chế độ nhúng (`fairseq`, `transformers`, `onnx`).
    """)
    quit()
elif argv_is_allows[15] in argv:
    print("""Các tham số của --preprocess:
        1. Thông tin mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.

        2. Cấu hình dữ liệu:
            - `--dataset_path` (mặc định: `./dataset`): Đường dẫn thư mục chứa tệp dữ liệu.
            - `--sample_rate` (bắt buộc): Tần số lấy mẫu của dữ liệu âm thanh.

        3. Cấu hình xử lý:
            - `--cpu_cores` (mặc định: `2`): Số lượng luồng CPU sử dụng.
            - `--cut_preprocess` (mặc định: `True`): Có cắt tệp dữ liệu hay không.
            - `--process_effects` (mặc định: `False`): Có áp dụng tiền xử lý hay không.
            - `--clean_dataset` (mặc định: `False`): Có làm sạch tệp dữ liệu hay không.
            - `--clean_strength` (mặc định: `0.7`): Độ mạnh của quá trình làm sạch dữ liệu.
    """)
    quit()
elif argv_is_allows[16] in argv:
    print("""Các tham số của --separator_music:
        1. Đường dẫn dữ liệu:
            - `--input_path` (bắt buộc): Đường dẫn tệp âm thanh đầu vào.
            - `--output_path` (mặc định: `./audios`): Thư mục lưu tệp đầu ra.
            - `--format` (mặc định: `wav`): Định dạng xuất tệp (`wav`, `mp3`,...).

        2. Cấu hình xử lý âm thanh:
            - `--shifts` (mặc định: `2`): Số lượng dự đoán.
            - `--segments_size` (mặc định: `256`): Kích thước phân đoạn âm thanh.
            - `--overlap` (mặc định: `0.25`): Mức độ chồng lấn giữa các đoạn.
            - `--mdx_hop_length` (mặc định: `1024`): Bước nhảy MDX khi xử lý.
            - `--mdx_batch_size` (mặc định: `1`): Kích thước lô.

        3. Xử lý làm sạch:
            - `--clean_audio` (mặc định: `False`): Có làm sạch âm thanh hay không.
            - `--clean_strength` (mặc định: `0.7`): Độ mạnh của bộ lọc làm sạch.

        4. Cấu hình mô hình:
            - `--model_name` (mặc định: `HT-Normal`): Mô hình tách nhạc (`Main_340`, `Main_390`, `Main_406`, `Main_427`, `Main_438`, `Inst_full_292`, `Inst_HQ_1`, `Inst_HQ_2`, `Inst_HQ_3`, `Inst_HQ_4`, `Inst_HQ_5`, `Kim_Vocal_1`, `Kim_Vocal_2`, `Kim_Inst`, `Inst_187_beta`, `Inst_82_beta`, `Inst_90_beta`, `Voc_FT`, `Crowd_HQ`, `Inst_1`, `Inst_2`, `Inst_3`, `MDXNET_1_9703`, `MDXNET_2_9682`, `MDXNET_3_9662`, `Inst_Main`, `MDXNET_Main`, `MDXNET_9482`, `HT-Normal`, `HT-Tuned`, `HD_MMI`,  `HT_6S`).
            - `--kara_model` (mặc định: `Version-1`): Phiên bản mô hình tách bè (`Version-1`, `Version-2`).

        5. Hiệu ứng và xử lý hậu kỳ:
            - `--backing` (mặc định: `False`): Có tách bè hay không.
            - `--mdx_denoise` (mặc định: `False`): Có sử dụng khử nhiễu MDX hay không.
            - `--reverb` (mặc định: `False`): Có tách vang hay không.
            - `--backing_reverb` (mặc định: `False`): có tách vang cho giọng bè không.

        6. Tần số lấy mẫu:
            - `--sample_rate` (mặc định: `44100`): Tần số lấy mẫu của âm thanh đầu ra.
    """)
    quit()
elif argv_is_allows[17] in argv:
    print("""Các tham số của --train:
        1. Cấu hình mô hình:
            - `--model_name` (bắt buộc): Tên mô hình.
            - `--rvc_version` (mặc định: `v2`): Phiên bản RVC (`v1`, `v2`).
            - `--model_author` (tùy chọn): Tác giả của mô hình.

        2. Cấu hình lưu:
            - `--save_every_epoch` (bắt buộc): Số kỷ nguyên giữa mỗi lần lưu.
            - `--save_only_latest` (mặc định: `True`): Chỉ lưu điểm mới nhất.
            - `--save_every_weights` (mặc định: `True`): Lưu tất cả trọng số của mô hình.

        3. Cấu hình huấn luyện:
            - `--total_epoch` (mặc định: `300`): Tổng số kỷ nguyên huấn luyện.
            - `--batch_size` (mặc định: `8`): Kích thước lô trong quá trình huấn luyện.
            - `--sample_rate` (bắt buộc): Tần số lấy mẫu của âm thanh.

        4. Cấu hình thiết bị:
            - `--gpu` (mặc định: `0`): Chỉ định GPU để sử dụng (số hoặc `-` nếu không dùng GPU).
            - `--cache_data_in_gpu` (mặc định: `False`): Lưu dữ liệu vào GPU để tăng tốc.

        5. Cấu hình huấn luyện nâng cao:
            - `--pitch_guidance` (mặc định: `True`): Sử dụng hướng dẫn cao độ.
            - `--g_pretrained_path` (mặc định: ``): Đường dẫn đến trọng số G đã huấn luyện trước.
            - `--d_pretrained_path` (mặc định: ``): Đường dẫn đến trọng số D đã huấn luyện trước.
            - `--vocoder` (mặc định: `Default`): Bộ mã hóa được sử dụng (`Default`, `MRF-HiFi-GAN`, `RefineGAN`).

        6. Phát hiện huấn luyện quá mức:
            - `--overtraining_detector` (mặc định: `False`): Bật/tắt chế độ phát hiện huấn luyện quá mức.
            - `--overtraining_threshold` (mặc định: `50`): Ngưỡng để xác định huấn luyện quá mức.

        7. Xử lý dữ liệu:
            - `--cleanup` (mặc định: `False`): Dọn dẹp tệp huấn luyện cũ để tiến hành huấn luyện lại từ đầu.

        8. Tối ưu:
            - `--checkpointing` (mặc định: `False`): Bật/tắt checkpointing để tiết kiệm RAM.
            - `--deterministic` (mặc định: `False`): Khi bật sẽ sử dụng các thuật toán có tính xác định cao, đảm bảo rằng mỗi lần chạy cùng một dữ liệu đầu vào sẽ cho kết quả giống nhau.
            - `--benchmark` (mặc định: `False`): Khi bật sẽ thử nghiệm và chọn thuật toán tối ưu nhất cho phần cứng và kích thước cụ thể.
            - `--optimizer` (mặc định: `AdamW`): Trình tối ưu hóa được sử dụng (`AdamW`, `RAdam`).
    """)
    quit()
elif argv_is_allows[18] in argv:
    print("""Sử dụng:
        1. `--help_audio_effects`: Trợ giúp về phần thêm hiệu ứng âm thanh.
        2. `--help_audioldm2`: Trợ giúp về phần chỉnh sửa nhạc.
        3. `--help_convert`: Trợ giúp về chuyển đổi âm thanh.
        4. `--help_create_dataset`: Trợ giúp về tạo dữ liệu huấn luyện.
        5. `--help_create_index`: Trợ giúp về tạo chỉ mục.
        6. `--help_extract`: Trợ giúp về trích xuất dữ liệu huấn luyện.
        7. `--help_preprocess`: Trợ giúp về xử lý trước dữ liệu.
        8. `--help_separator_music`: Trợ giúp về tách nhạc.
        9. `--help_train`: Trợ giúp về huấn luyện mô hình.
    """)
    quit()

if __name__ == "__main__":
    import torch.multiprocessing as mp

    if "--train" in argv: mp.set_start_method("spawn")
    if "--preprocess" in argv or "--extract" in argv: mp.set_start_method("spawn", force=True)

    main()