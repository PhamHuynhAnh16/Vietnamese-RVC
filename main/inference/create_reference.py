import os
import sys
import time
import torch
import shutil
import warnings
import argparse

import numpy as np

from tqdm import tqdm

sys.path.append(os.getcwd())

from main.library.audio.audio import load_audio
from main.app.variables import config, logger, translations, configs
from main.library.utils import load_embedders_model, extract_features, check_assets, strtobool

warnings.filterwarnings("ignore")

# Signal processing global parameters initialized from config specifications
F0_MIN, F0_MAX, HOP_SIZE, SAMPLE_RATE, FRAME_LENGTH = configs.get("f0_min", 50), configs.get("f0_max", 1100), 160, 16000, 2048

def parse_arguments():
    """
    Parses command-line arguments configuring the audio reference creation parameters.

    Returns:
        argparse.Namespace: Object containing validated operational configurations.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--create_reference", action='store_true')
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--reference_name", type=str, default="reference")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--predictor_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_up_key", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--proposal_pitch", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--proposal_pitch_threshold", type=float, default=255.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--embedders_mix", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mix_layers", type=int, default=9, required=False)
    parser.add_argument("--embedders_mix_ratio", type=float, default=0.5)

    return parser.parse_args()

def main():
    """Main execution orchestrator routing runtime command-line arguments."""

    args = parse_arguments()

    (
        audio_path, 
        reference_name, 
        pitch_guidance, 
        version, 
        embedder_model, 
        embedders_mode, 
        f0_method, 
        predictor_onnx, 
        f0_up_key, 
        filter_radius, 
        f0_autotune, 
        f0_autotune_strength, 
        proposal_pitch, 
        proposal_pitch_threshold, 
        alpha,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio
    ) = (
        args.audio_path, 
        args.reference_name, 
        args.pitch_guidance, 
        args.version, 
        args.embedder_model, 
        args.embedders_mode, 
        args.f0_method, 
        args.predictor_onnx, 
        args.f0_up_key, 
        args.filter_radius, 
        args.f0_autotune, 
        args.f0_autotune_strength, 
        args.proposal_pitch, 
        args.proposal_pitch_threshold, 
        args.alpha,
        args.embedders_mix,
        args.embedders_mix_layers,
        args.embedders_mix_ratio
    )

    # Configure structural log data map mapping parameter values to localization files
    log_data = {
        translations["audio_path"]: audio_path, 
        translations["reference_name"]: reference_name,
        translations['extract_f0']: pitch_guidance, 
        translations['training_version']: version, 
        translations['hubert_model']: embedder_model, 
        translations["embed_mode"]: embedders_mode, 
        translations['f0_method']: f0_method, 
        translations["predictor_onnx"]: predictor_onnx, 
        translations["pitch"]: f0_up_key, 
        translations["filter_radius"]: filter_radius, 
        translations["autotune"]: f0_autotune, 
        translations["autotune_rate_info"]: f0_autotune_strength,
        translations["proposal_pitch"]: proposal_pitch, 
        translations["proposal_pitch_threshold"]: proposal_pitch_threshold,
        translations["alpha_label"]: alpha,
        translations["embedders_mix"]: embedders_mix,
        translations["embedders_mix_layers"]: embedders_mix_layers,
        translations["embedders_mix_ratio"]: embedders_mix_ratio,
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    # Validate the availability of target network files before entering computing phases
    check_assets(
        f0_method, 
        embedder_model, 
        predictor_onnx, 
        embedders_mode
    )

    create_reference(
        audio_path, 
        reference_name, 
        pitch_guidance, 
        version, 
        embedder_model, 
        embedders_mode, 
        f0_method, 
        predictor_onnx, 
        f0_up_key, 
        filter_radius, 
        f0_autotune, 
        f0_autotune_strength, 
        proposal_pitch, 
        proposal_pitch_threshold,
        alpha,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio
    )

def create_reference(
    audio_path, 
    reference_name,
    pitch_guidance = True,
    version = "v2",
    embedder_model = "hubert_base", 
    embedders_mode = "fairseq", 
    f0_method = "rmvpe",
    predictor_onnx = False,
    f0_up_key = 0,
    filter_radius = 3,
    f0_autotune = False,
    f0_autotune_strength = 1,
    proposal_pitch = False,
    proposal_pitch_threshold = 255.0,
    alpha = 0.5,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5
):
    """
    Loads target waveforms, extracts voice encoder hidden units, 
    tracks explicit frequency lines, and saves representations to a structured cache directory.
    """

    device = config.device
    is_half = config.is_half

    # Enforce standard early escape boundary if the requested source media path remains null
    if not audio_path:
        logger.warning(translations["not_found_audio"])
        sys.exit(1)

    # Establish complete workspace string path mapping target variables
    output_reference = os.path.join(
        configs["reference_path"], 
        f"{reference_name}_{version}_{embedder_model}_{pitch_guidance}"
    )

    # Purge legacy target assets directory locations matching current profiles
    if os.path.exists(output_reference): shutil.rmtree(reference_name, ignore_errors=True)

    os.makedirs(output_reference, exist_ok=True)
    logger.info(translations["start_create_reference"])
    start_time = time.time()

    with tqdm(total=4, desc=translations["create_reference"], ncols=100, unit="a") as pbar:
        # Step 1: Decode source waveform tracks mapping uniform resampling constraints
        audio = load_audio(audio_path, sample_rate=SAMPLE_RATE)
        pbar.update(1)

        # Apply strict peak normalization bounding constraints to prevent digital clipping
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1: audio /= audio_max

        # Trim residual tail frames out to match uniform encoder segment structures (320 hop sizing ratios)
        trimmed_len = (len(audio) // 320) * 320
        audio = audio[:trimmed_len]

        # Pad boundaries sequentially to compensate for receptive field edge alignment compression losses
        audio_pad = torch.nn.functional.pad(
            torch.from_numpy(audio).to(
                torch.float16 if is_half else torch.float32
            ).to(device).unsqueeze(0), 
            (40, 40), 
            mode="reflect"
        )
        pbar.update(1)

        # Step 2: Initialize embedding extractor model assets via designated system devices
        embedder = load_embedders_model(embedder_model, embedders_mode)
        if isinstance(embedder, torch.nn.Module): 
            embedder = embedder.to(torch.float16 if is_half else torch.float32).eval().to(device)
            if config.compile_all: embedder = torch.compile(embedder, mode=config.compile_mode)

        # Step 3: Extract feature tokens without tracking gradient graphs
        with torch.no_grad():
            feats = extract_features(
                embedder, 
                audio_pad.view(1, -1), 
                version, 
                mix=embedders_mix, 
                mix_layers=embedders_mix_layers, 
                mix_ratio=embedders_mix_ratio
            )

        # Save voice linguistic identity vectors to output folder
        np.save(
            os.path.join(output_reference, "feats.npy"), 
            feats.squeeze(0).float().cpu().numpy(), 
            allow_pickle=False
        )

        pbar.update(1)
        # Step 4: Extract and compute Pitch (F0) profiles if requested
        if pitch_guidance:
            from main.library.predictors.Generator import Generator

            generator = Generator(
                sample_rate=SAMPLE_RATE, 
                hop_length=HOP_SIZE, 
                f0_min=F0_MIN, 
                f0_max=F0_MAX, 
                alpha=alpha, 
                is_half=is_half, 
                device=device, 
                predictor_onnx=predictor_onnx
            )

            # Compute fine-grained pitch curves alongside coarsened quantization assignments
            pitch, pitchf = generator.calculator(
                x_pad=config.x_pad, 
                f0_method=f0_method, 
                x=audio, 
                f0_up_key=f0_up_key, 
                p_len=audio.shape[0] // 160 + 1, 
                filter_radius=filter_radius, 
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength, 
                manual_f0=None, 
                proposal_pitch=proposal_pitch, 
                proposal_pitch_threshold=proposal_pitch_threshold
            )

            # Write individual binary arrays out to workspace destination points
            np.save(
                os.path.join(output_reference, "pitch_coarse.npy"), 
                pitch, 
                allow_pickle=False
            )

            np.save(
                os.path.join(output_reference, "pitch_fine.npy"), 
                pitchf, 
                allow_pickle=False
            )

        pbar.update(1)

    logger.info(translations["create_reference_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

if __name__ == "__main__": main()