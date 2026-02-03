import os
import sys
import json
import shutil

from random import shuffle

sys.path.append(os.getcwd())

from main.app.core.ui import configs, config
from main.inference.extracting.embedding import create_mute_file

def mute_file(
    embedders_mode, 
    embedders_model, 
    mute_base_path, 
    rvc_version
):
    if embedders_mode.startswith(("spin", "whisper")):
        mute_file = f"mute_{embedders_model}.npy"
    else:
        mute_file = {
            "contentvec_base": "mute.npy",
            "hubert_base": "mute.npy",
            "vietnamese_hubert_base": "mute_vietnamese.npy",
            "japanese_hubert_base": "mute_japanese.npy",
            "korean_hubert_base": "mute_korean.npy",
            "chinese_hubert_base": "mute_chinese.npy",
            "portuguese_hubert_base": "mute_portuguese.npy"
        }.get(embedders_model, None)

    if mute_file is None or not os.path.exists(mute_file):
        create_mute_file(
            rvc_version, 
            embedders_model, 
            embedders_mode, 
            config.is_half
        )
        mute_file = f"mute_{embedders_model}.npy"

    return os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file)

def generate_config(rvc_version, sample_rate, model_path):
    config_save_path = os.path.join(model_path, "config.json")

    if not os.path.exists(config_save_path): 
        shutil.copy(
            os.path.join("main", "configs", rvc_version, f"{sample_rate}.json"), 
            config_save_path
        )

def generate_filelist(
    pitch_guidance, 
    model_path, 
    rvc_version, 
    sample_rate, 
    embedders_mode = "fairseq", 
    embedder_model = "hubert_base", 
    rms_extract = False, 
    include_mutes = 2
):
    f0_dir, f0nsf_dir, energy_dir = None, None, None

    gt_wavs_dir, feature_dir = (
        os.path.join(model_path, "sliced_audios"), 
        os.path.join(model_path, f"{rvc_version}_extracted")
    )

    if pitch_guidance: 
        f0_dir, f0nsf_dir = (
            os.path.join(model_path, "f0"), 
            os.path.join(model_path, "f0_voiced")
        )

    if rms_extract: 
        energy_dir = os.path.join(model_path, "energy")

    gt_wavs_files, feature_files = (
        set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), 
        set(name.split(".")[0] for name in os.listdir(feature_dir))
    )

    names = gt_wavs_files & feature_files

    if pitch_guidance: 
        names = (
            names & 
            set(name.split(".")[0] for name in os.listdir(f0_dir)) & 
            set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
        )

    if rms_extract: 
        names = (
            names & 
            set(name.split(".")[0] for name in os.listdir(energy_dir))
        )
    
    options, sids = [], []
    mute_base_path = os.path.join(configs["logs_path"], "mute")

    for name in names:
        sid = name.split("_")[0]
        if sid not in sids: sids.append(sid)

        option = {
            True: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{energy_dir}/{name}.wav.npy|{sid}",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{sid}"
            },
            False: {
                True: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{energy_dir}/{name}.wav.npy|{sid}",
                False: f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{sid}"
            }
        }[pitch_guidance][rms_extract]

        options.append(option)

    if include_mutes > 0:
        mute_audio_path, mute_feature_path = (
            os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), 
            mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version)
        )

        mute_f0_path, mute_f0nsf_path = (
            os.path.join(mute_base_path, 'f0', 'mute.wav.npy'), 
            os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')
        )

        mute_energy_path = os.path.join(mute_base_path, 'energy', 'mute.wav.npy')
        
        for sid in sids * include_mutes:
            option = {
                True: {
                    True: f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{mute_energy_path}|{sid}",
                    False: f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}"
                },
                False: {
                    True: f"{mute_audio_path}|{mute_feature_path}|{mute_energy_path}|{sid}",
                    False: f"{mute_audio_path}|{mute_feature_path}|{sid}"
                }
            }[pitch_guidance][rms_extract]

            options.append(option)

    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

    configs_path = os.path.join(model_path, "config.json")
    with open(configs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["sid"] = len(sids)

    with open(configs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)