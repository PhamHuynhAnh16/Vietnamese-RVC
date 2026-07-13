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
    """
    Locates or generates the correct silent (.npy) reference embedding file.

    Selects pre-computed embedding assets based on the specific language model variation, 
    or dynamically creates one using the corresponding extraction backend if missing.

    Args:
        embedders_mode (str): Backend framework configuration type (e.g., 'fairseq', 'onnx').
        embedders_model (str): Name or path identifier of the embedder model.
        mute_base_path (str): Root workspace folder containing the reference mute assets.
        rvc_version (str): Target framework execution variant ('v1' or 'v2').

    Returns:
        str: Absolute system filepath pointing to the verified mute array file.
    """

    # Map explicit pre-built file names based on language-specific embedder variants
    mute_file = {
        "contentvec_base": "mute.npy",
        "hubert_base": "mute.npy",
        "vietnamese_hubert_base": "mute_vietnamese.npy",
        "japanese_hubert_base": "mute_japanese.npy",
        "korean_hubert_base": "mute_korean.npy",
        "chinese_hubert_base": "mute_chinese.npy",
        "portuguese_hubert_base": "mute_portuguese.npy",
        "spin-v1": "mute_spin-v1.npy",
        "spin-v2": "mute_spin-v2.npy"
    }.get(embedders_model, None)

    # Dynamic fallback: trigger extraction if file mapping is unregistered or missing from disk
    if mute_file is None or not os.path.exists(os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file)):
        create_mute_file(
            rvc_version, 
            embedders_model, 
            embedders_mode, 
            config.is_half
        )
        # Assign newly extracted file naming syntax convention
        mute_file = f"mute_{embedders_model}.npy"

    return os.path.join(mute_base_path, f"{rvc_version}_extracted", mute_file)

def generate_config(rvc_version, sample_rate, model_path):
    """
    Clones the global architecture JSON template into the local model experiment folder.

    Args:
        rvc_version (str): RVC version variant used to select the config subfolder.
        sample_rate (int): Target audio sample rate used to match the JSON template.
        model_path (str): Target directory where config.json will be saved.
    """

    config_save_path = os.path.join(model_path, "config.json")
    # Only clone template if no localized instance currently populates the experiment workspace
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
    include_mutes = 2
):
    """
    Validates completed extraction artifacts and builds the dataset filelist index.

    Cross-references file outputs, appends silent placeholder entries allocated per 
    identified speaker ID (SID), updates training state configuration details, and writes 
    the completed `filelist.txt`.

    Args:
        pitch_guidance (bool): If True, requires and writes paths for coarse and fine F0 components.
        model_path (str): Local path pointing to active training workspace directories.
        rvc_version (str): Active pipeline variation ('v1' or 'v2').
        sample_rate (int): Sample rate used to locate the correct silent placeholder file.
        embedders_mode (str, optional): Target extraction library format. Defaults to "fairseq".
        embedder_model (str, optional): Identity key name of the embedder. Defaults to "hubert_base".
        include_mutes (int, optional): Number of silent references to generate per Speaker ID. Defaults to 2.
    """

    f0_dir, f0nsf_dir = None, None
    # Map core workspace paths for chopped audio slices and content embeddings
    gt_wavs_dir, feature_dir = (
        os.path.join(model_path, "sliced_audios"), 
        os.path.join(model_path, f"{rvc_version}_extracted")
    )

    # Conditionally include pitch-tracking folders based on model needs
    if pitch_guidance: 
        f0_dir, f0nsf_dir = (
            os.path.join(model_path, "f0"), 
            os.path.join(model_path, "f0_voiced")
        )

    # Extract base filenames (excluding extensions) to align dataset features
    gt_wavs_files, feature_files = (
        set(name.split(".")[0] for name in os.listdir(gt_wavs_dir)), 
        set(name.split(".")[0] for name in os.listdir(feature_dir))
    )

    # Use set intersections to find files that successfully completed all extraction steps
    names = gt_wavs_files & feature_files

    if pitch_guidance: 
        names = (
            names & 
            set(name.split(".")[0] for name in os.listdir(f0_dir)) & 
            set(name.split(".")[0] for name in os.listdir(f0nsf_dir))
        )
    
    options, sids = [], []
    mute_base_path = os.path.join(configs["logs_path"], "mute")
    # Step 1: Parse verified names, extract speaker IDs, and format dataset rows
    for name in names:
        sid = name.split("_")[0]
        if sid not in sids: sids.append(sid)

        # Build paths separated by pipe characters matching structural requirements expected by RVC datasets
        option = f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{f0_dir}/{name}.wav.npy|{f0nsf_dir}/{name}.wav.npy|{sid}" if pitch_guidance else f"{gt_wavs_dir}/{name}.wav|{feature_dir}/{name}.npy|{sid}"
        options.append(option)

    # Step 2: Format and inject silent placeholder rows to prevent voice artifacting
    if include_mutes > 0:
        mute_audio_path, mute_feature_path = (
            os.path.join(mute_base_path, "sliced_audios", f"mute{sample_rate}.wav"), 
            mute_file(embedders_mode, embedder_model, mute_base_path, rvc_version)
        )

        mute_f0_path, mute_f0nsf_path = (
            os.path.join(mute_base_path, 'f0', 'mute.wav.npy'), 
            os.path.join(mute_base_path, 'f0_voiced', 'mute.wav.npy')
        )
        
        # Distribute silent entries proportionally across all discovered unique speaker categories
        for sid in sids * include_mutes:
            option = f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}" if pitch_guidance else f"{mute_audio_path}|{mute_feature_path}|{sid}"
            options.append(option)

    # Step 3: Shuffle execution lines randomly to minimize batch bias during training loops
    shuffle(options)
    with open(os.path.join(model_path, "filelist.txt"), "w") as f:
        f.write("\n".join(options))

    # Step 4: Dynamically update global dictionary dimensions ('sid' count parameter) within config JSON file
    configs_path = os.path.join(model_path, "config.json")
    with open(configs_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["sid"] = len(sids)

    with open(configs_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)