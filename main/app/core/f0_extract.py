import os
import sys
import librosa

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from main.app.core.ui import gr_info, gr_warning
from main.library.predictors.Generator import Generator
from main.app.variables import config, translations, configs
from main.library.utils import check_predictors, get_providers

def f0_extract(audio, f0_method, f0_onnx):
    if not audio or not os.path.exists(audio) or os.path.isdir(audio): 
        gr_warning(translations["input_not_valid"])
        return [None]*2

    check_predictors(f0_method, f0_onnx)

    f0_path = os.path.join(configs["f0_path"], os.path.splitext(os.path.basename(audio))[0])
    image_path = os.path.join(f0_path, "f0.png")
    txt_path = os.path.join(f0_path, "f0.txt")

    gr_info(translations["start_extract"])

    if not os.path.exists(f0_path): os.makedirs(f0_path, exist_ok=True)

    y, sr = librosa.load(audio, sr=None)

    f0_generator = Generator(sr, 160, 50, 1600, is_half=config.is_half, device=config.device, providers=get_providers(), f0_onnx_mode=f0_onnx)
    _, pitchf = f0_generator.calculator(config.x_pad, f0_method, y, 0, None, 3, False, 0, None, False)

    F_temp = np.array(pitchf, dtype=np.float32)
    F_temp[F_temp == 0] = np.nan

    f0 = 1200 * np.log2(F_temp / librosa.midi_to_hz(0))

    plt.figure(figsize=(10, 4))
    plt.plot(f0)
    plt.title(f0_method)
    plt.xlabel(translations["time_frames"])
    plt.ylabel(translations["Frequency"])
    plt.savefig(image_path)
    plt.close()

    with open(txt_path, "w") as f:
        for i, f0_value in enumerate(f0):
            f.write(f"{i * sr / 160},{f0_value}\n")

    gr_info(translations["extract_done"])

    return [txt_path, image_path]