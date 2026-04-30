import os
import sys

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.app.variables import config, translations, configs

def f0_extract(
    audio, 
    f0_method, 
    predictor_onnx,
    pitch,
    filter_radius,
    f0_autotune,
    f0_autotune_strength,
    proposal_pitch,
    proposal_pitch_threshold
):
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2

    import numpy as np
    import matplotlib.pyplot as plt

    from main.library.utils import check_assets, load_audio
    from main.library.predictors.Generator import Generator

    check_assets(f0_method, "", predictor_onnx, "")

    f0_path = os.path.join(configs["f0_path"], os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])
    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    y = load_audio(audio, sample_rate=16000)

    f0_generator = Generator(
        sample_rate=16000, 
        hop_length=160, 
        f0_min=configs.get("f0_min", 50), 
        f0_max=configs.get("f0_max", 1100), 
        alpha=0.5, 
        is_half=config.is_half, 
        device=config.device, 
        predictor_onnx=predictor_onnx, 
        delete_predictor_onnx=predictor_onnx
    )

    _, pitchf = f0_generator.calculator(
        x_pad=config.x_pad, 
        f0_method=f0_method, 
        x=y, 
        f0_up_key=pitch, 
        p_len=None, 
        filter_radius=filter_radius, 
        f0_autotune=f0_autotune, 
        f0_autotune_strength=f0_autotune_strength, 
        manual_f0=None, 
        proposal_pitch=proposal_pitch,
        proposal_pitch_threshold=proposal_pitch_threshold
    )

    F_temp = np.array(pitchf, dtype=np.float32)
    F_temp[F_temp == 0] = np.nan

    f0 = 1200 * np.log2(F_temp / 8.175798915643707)

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(pitchf):
            f.write(f"{i},{f0_value}\n")

    gr_info(translations["extract_done"])

    return [txt_path, image_path]