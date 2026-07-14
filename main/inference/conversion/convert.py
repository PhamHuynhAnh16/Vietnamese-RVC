import os
import sys
import time
import torch
import librosa
import logging
import argparse
import warnings

import numpy as np
import soundfile as sf

from tqdm import tqdm
from scipy import signal

sys.path.append(os.getcwd())

from main.library.audio.upscaler import FlashSR
from main.app.core.ui import replace_export_format
from main.library.audio.noisereduce import TorchGate
from main.inference.conversion.pipeline import Pipeline
from main.library.audio.audio import load_audio, cut, restore
from main.app.variables import config, logger, translations, file_types
from main.library.audio.audio_processing import preprocess, postprocess
from main.library.utils import check_assets, check_upscaler, load_embedders_model, load_model, strtobool, load_faiss_index

if not config.debug_mode:
    warnings.filterwarnings("ignore")

    for l in [
        "torch", 
        "faiss", 
        "omegaconf", 
        "httpx", 
        "httpcore", 
        "faiss.loader", 
        "numba.core", 
        "urllib3", 
        "transformers", 
        "matplotlib"
    ]:
        logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    """
    Parses command-line arguments for the voice conversion pipeline.

    Returns:
        argparse.Namespace: The parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--convert", action='store_true')
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--filter_radius", type=int, default=3)
    parser.add_argument("--index_rate", type=float, default=0.5)
    parser.add_argument("--rms_mix_rate", type=float, default=1)
    parser.add_argument("--protect", type=float, default=0.33)
    parser.add_argument("--hop_length", type=int, default=64)
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./audios/output.wav")
    parser.add_argument("--export_format", type=str, default="wav")
    parser.add_argument("--pth_path",  type=str,  required=True)
    parser.add_argument("--index_path", type=str, default="")
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--clean_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--resample_sr", type=int, default=0)
    parser.add_argument("--split_audio", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--checkpointing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_file", type=str, default="")
    parser.add_argument("--predictor_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--formant_shifting", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--formant_qfrency", type=float, default=0.8)
    parser.add_argument("--formant_timbre", type=float, default=0.8)
    parser.add_argument("--proposal_pitch", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--proposal_pitch_threshold", type=float, default=255.0)
    parser.add_argument("--audio_processing", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--sid", type=int, default=0, required=False)
    parser.add_argument("--embedders_mix", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mix_layers", type=int, default=9, required=False)
    parser.add_argument("--embedders_mix_ratio", type=float, default=0.5)
    parser.add_argument("--noise_scale", type=float, default=0.35)
    parser.add_argument("--nprobe", type=int, default=1)
    parser.add_argument("--audio_upscaler", type=lambda x: bool(strtobool(x)), default=False)

    return parser.parse_args()

def main():
    """
    Main entry point function. Unpacks command line arguments and hands over execution 
    to the conversion controller.
    """

    args = parse_arguments()

    # Unpack parsed arguments safely into a local tuple structure
    (
        pitch, 
        filter_radius, 
        index_rate, 
        rms_mix_rate, 
        protect, 
        hop_length, 
        f0_method, 
        input_path, 
        output_path, 
        pth_path, 
        index_path, 
        f0_autotune, 
        f0_autotune_strength, 
        clean_audio, 
        clean_strength, 
        export_format, 
        embedder_model, 
        resample_sr, 
        split_audio, 
        checkpointing, 
        f0_file, 
        predictor_onnx, 
        embedders_mode, 
        formant_shifting, 
        formant_qfrency, 
        formant_timbre, 
        proposal_pitch, 
        proposal_pitch_threshold, 
        audio_processing, 
        alpha,
        sid,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio,
        noise_scale,
        nprobe,
        audio_upscaler
    ) = (
        args.pitch, 
        args.filter_radius, 
        args.index_rate, 
        args.rms_mix_rate,
        args.protect, 
        args.hop_length, 
        args.f0_method, 
        args.input_path, 
        args.output_path, 
        args.pth_path, 
        args.index_path, 
        args.f0_autotune, 
        args.f0_autotune_strength, 
        args.clean_audio, 
        args.clean_strength, 
        args.export_format, 
        args.embedder_model, 
        args.resample_sr, 
        args.split_audio, 
        args.checkpointing, 
        args.f0_file, 
        args.predictor_onnx, 
        args.embedders_mode, 
        args.formant_shifting, 
        args.formant_qfrency, 
        args.formant_timbre, 
        args.proposal_pitch, 
        args.proposal_pitch_threshold, 
        args.audio_processing, 
        args.alpha,
        args.sid,
        args.embedders_mix,
        args.embedders_mix_layers,
        args.embedders_mix_ratio,
        args.noise_scale,
        args.nprobe,
        args.audio_upscaler
    )
    
    # Run the compiled configurations in the pipeline controller
    run_convert_script(
        pitch=pitch, 
        filter_radius=filter_radius, 
        index_rate=index_rate, 
        rms_mix_rate=rms_mix_rate, 
        protect=protect, 
        hop_length=hop_length, 
        f0_method=f0_method, 
        input_path=input_path, 
        output_path=output_path, 
        pth_path=pth_path, 
        index_path=index_path, 
        f0_autotune=f0_autotune, 
        f0_autotune_strength=f0_autotune_strength, 
        clean_audio=clean_audio, 
        clean_strength=clean_strength, 
        export_format=export_format, 
        embedder_model=embedder_model, 
        resample_sr=resample_sr, 
        split_audio=split_audio, 
        checkpointing=checkpointing, 
        f0_file=f0_file, 
        predictor_onnx=predictor_onnx, 
        embedders_mode=embedders_mode, 
        formant_shifting=formant_shifting, 
        formant_qfrency=formant_qfrency, 
        formant_timbre=formant_timbre, 
        proposal_pitch=proposal_pitch, 
        proposal_pitch_threshold=proposal_pitch_threshold, 
        audio_processing=audio_processing, 
        alpha=alpha,
        sid=sid,
        embedders_mix=embedders_mix, 
        embedders_mix_layers=embedders_mix_layers, 
        embedders_mix_ratio=embedders_mix_ratio,
        noise_scale=noise_scale,
        nprobe=nprobe,
        audio_upscaler=audio_upscaler
    )

def run_convert_script(
    pitch=0, 
    filter_radius=3, 
    index_rate=0.5, 
    rms_mix_rate=1, 
    protect=0.5, 
    hop_length=64, 
    f0_method="rmvpe", 
    input_path=None, 
    output_path="./output.wav", 
    pth_path=None, 
    index_path=None, 
    f0_autotune=False, 
    f0_autotune_strength=1, 
    clean_audio=False, 
    clean_strength=0.7, 
    export_format="wav", 
    embedder_model="hubert_base", 
    resample_sr=0, 
    split_audio=False, 
    checkpointing=False, 
    f0_file=None, 
    predictor_onnx=False, 
    embedders_mode="fairseq", 
    formant_shifting=False, 
    formant_qfrency=0.8, 
    formant_timbre=0.8, 
    proposal_pitch=False, 
    proposal_pitch_threshold=255.0, 
    audio_processing=False,
    alpha=0.5,
    sid=0,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5,
    noise_scale = 0.35,
    nprobe = 1,
    audio_upscaler = False
):
    """
    Validates components, coordinates IO batching/single files, and maps runtime configurations 
    to execute the conversion pipeline.

    Args:
        pitch (int): Semitone pitch shift value. Defaults to 0.
        filter_radius (int): Median filter radius for pitch smoothing. Defaults to 3.
        index_rate (float): FAISS index feature blend ratio. Defaults to 0.5.
        rms_mix_rate (float): Volume envelope blend ratio. Defaults to 1.
        protect (float): Consonant protection factor threshold. Defaults to 0.5.
        hop_length (int): Frame step size for pitch tracking components. Defaults to 64.
        f0_method (str): Core pitch extraction methodology name. Defaults to "rmvpe".
        input_path (str): Target file system directory or single audio file location. Defaults to None.
        output_path (str): Path where processed audio will be saved. Defaults to "./output.wav".
        pth_path (str): Path to weights file (.pth or .onnx formats). Defaults to None.
        index_path (str): Path to matching .index file tracker. Defaults to None.
        f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
        f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
        clean_audio (bool): Applies background noise gating blocks. Defaults to False.
        clean_strength (float): Suppression ratio for noise reduction gate. Defaults to 0.7.
        export_format (str): Outbound audio encoding identifier (e.g., 'wav', 'mp3'). Defaults to "wav".
        embedder_model (str): Acoustic feature model identification. Defaults to "hubert_base".
        resample_sr (int): Forced target resample sampling rate. Defaults to 0.
        split_audio (bool): Segments long files by silence thresholds. Defaults to False.
        checkpointing (bool): Toggles PyTorch gradient checkpointing blocks. Defaults to False.
        f0_file (str): Input external manual pitch data matrix track. Defaults to None.
        predictor_onnx (bool): Switches F0 tracking to accelerated ONNX backend. Defaults to False.
        embedders_mode (str): Framework source indicator for speech embedding models. Defaults to "fairseq".
        formant_shifting (bool): Modulates vocal format frequencies early on. Defaults to False.
        formant_qfrency (float): Formant queuing factor filter width. Defaults to 0.8.
        formant_timbre (float): Formant frequency transposition scalar ratio. Defaults to 0.8.
        proposal_pitch (bool): Enable automatic pitch key shifting calculation based on median F0 alignment.
        proposal_pitch_threshold (float): The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.
        audio_processing (bool): Evaluates early/late acoustic stage equalization treatments. Defaults to False.
        alpha (float): Feature blending weighting for underlying components. Defaults to 0.5.
        sid (int): Target speaker index for multi-speaker networks. Defaults to 0.
        embedders_mix (bool): Combines multiple internal layer hidden states. Defaults to False.
        embedders_mix_layers (int): Number of embedding layers to slice. Defaults to 9.
        embedders_mix_ratio (float): Proportional weighting for embedding layers blending. Defaults to 0.5.
        noise_scale (float): Control scalar scaling flow variance. Defaults to 0.35.
        nprobe (int): IVF FAISS space search accuracy parameter. Defaults to 1.
        audio_upscaler (bool): Enforces super-resolution upscaling pipelines. Defaults to False.
    """

    # Verify presence of dependent external assets and upscaler weights
    if audio_upscaler: check_upscaler()
    check_assets(f0_method, embedder_model, predictor_onnx=predictor_onnx, embedders_mode=embedders_mode)

    # Dictionary construction for debugging logs containing human-readable localizations
    log_data = {
        translations["pitch"]: pitch, 
        translations["filter_radius"]: filter_radius, 
        translations["index_strength"]: index_rate, 
        translations["rms_mix_rate"]: rms_mix_rate, 
        translations["protect"]: protect, 
        translations["hop_length"]: hop_length, 
        translations["f0_method"]: f0_method, 
        translations["audio_path"]: input_path, 
        translations["output_path"]: replace_export_format(output_path, export_format), 
        translations["model_path"]: pth_path, 
        translations["indexpath"]: index_path, 
        translations["autotune"]: f0_autotune, 
        translations["autotune_rate_info"]: f0_autotune_strength,
        translations["clear_audio"]: clean_audio, 
        translations["clean_strength"]: clean_strength,
        translations["sample_rate"]: resample_sr,
        translations["export_format"]: export_format, 
        translations["hubert_model"]: embedder_model, 
        translations["split_audio"]: split_audio, 
        translations["memory_efficient_training"]: checkpointing, 
        translations["predictor_onnx"]: predictor_onnx, 
        translations["f0_file"]: f0_file,
        translations["embed_mode"]: embedders_mode, 
        translations["proposal_pitch"]: proposal_pitch, 
        translations["proposal_pitch_threshold"]: proposal_pitch_threshold,
        translations["audio_processing"]: audio_processing,
        translations["alpha_label"]: alpha,
        translations["embedders_mix"]: embedders_mix,
        translations["embedders_mix_layers"]: embedders_mix_layers,
        translations["embedders_mix_ratio"]: embedders_mix_ratio,
        translations["formant_qfrency"]: formant_qfrency,
        translations["formant_timbre"]: formant_timbre,
        translations["noise_scale"]: noise_scale,
        translations["nprobe"]: nprobe,
        translations["audio_upscaler"]: audio_upscaler
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")
    
    # Model sanity validation check
    if not pth_path or not os.path.exists(pth_path) or os.path.isdir(pth_path) or not pth_path.endswith((".pth", ".onnx")):
        logger.warning(translations["provide_model"])
        sys.exit(1)

    start_time = time.time()
    # Instantiate the wrapper converter architecture
    cvt = VoiceConverter(pth_path, embedder_model, embedders_mode, sid, noise_scale, checkpointing, hop_length, alpha, predictor_onnx, clean_audio, clean_strength, audio_upscaler)

    # Write current Process ID (PID) to disk so other sub-processes or GUIs can track/cancel this task
    pid_path = os.path.join("assets", "convert_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))

    def convert_audio(audio_path, output_audio):
        """Helper inner function wrapping the conversion pipeline parameters invocation."""

        cvt.convert_audio(
            pitch=pitch, 
            filter_radius=filter_radius, 
            index_rate=index_rate, 
            rms_mix_rate=rms_mix_rate, 
            protect=protect, 
            f0_method=f0_method, 
            audio_input_path=audio_path, 
            audio_output_path=output_audio, 
            index_path=index_path, 
            f0_autotune=f0_autotune, 
            f0_autotune_strength=f0_autotune_strength, 
            export_format=export_format, 
            resample_sr=resample_sr, 
            f0_file=f0_file, 
            formant_shifting=formant_shifting, 
            formant_qfrency=formant_qfrency, 
            formant_timbre=formant_timbre, 
            split_audio=split_audio, 
            proposal_pitch=proposal_pitch, 
            proposal_pitch_threshold=proposal_pitch_threshold,
            audio_processing=audio_processing,
            embedders_mix=embedders_mix, 
            embedders_mix_layers=embedders_mix_layers, 
            embedders_mix_ratio=embedders_mix_ratio,
            nprobe=nprobe
        )

    if os.path.isdir(input_path):
        # Case A: Batch Processing if the input path points to a Directory

        logger.info(translations["convert_batch"])
        # Filter and extract compatible audio items from target path
        audio_files = [
            f 
            for f in os.listdir(input_path) 
            if f.lower().endswith(tuple(file_types))
        ]

        if not audio_files: 
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(translations["found_audio"].format(audio_files=len(audio_files)))
        # Loop through valid audio filenames and initiate conversions
        for audio in audio_files:
            audio_path = os.path.join(input_path, audio)
            output_audio = os.path.join(input_path, os.path.splitext(audio)[0] + f"_output.{export_format}")

            logger.info(f"{translations['convert_audio']} '{audio_path}'...")
            if os.path.exists(output_audio): os.remove(output_audio)

            convert_audio(audio_path, output_audio)

        logger.info(
            translations["convert_batch_success"].format(
                elapsed_time=f"{(time.time() - start_time):.2f}", 
                output_path=replace_export_format(output_path, export_format)
            )
        )
    else:
        # Case B: Standard single audio file processing

        if not os.path.exists(input_path):
            logger.warning(translations["not_found_audio"])
            sys.exit(1)

        logger.info(f"{translations['convert_audio']} '{input_path}'...")
        if os.path.exists(output_path): os.remove(output_path)

        convert_audio(input_path, output_path)

        logger.info(
            translations["convert_audio_success"].format(
                input_path=input_path, 
                elapsed_time=f"{(time.time() - start_time):.2f}", 
                output_path=replace_export_format(output_path, export_format)
            )
        )

    # Cleanup the PID file after execution finishes successfully
    if os.path.exists(pid_path): os.remove(pid_path)

class VoiceConverter:
    """
    Handles initialization of neural networks (Vocoders, Embedders, Synthesisers) 
    and drives the overall inference sequence.
    """

    def __init__(
        self, 
        model_path, 
        embedder_model, 
        embedders_mode,
        sid = 0,
        noise_scale = 0.35,
        checkpointing = False,
        hop_length = 160,
        alpha = 0.5,
        predictor_onnx = False,
        clean_audio = False,
        clean_strength = 0.5,
        audio_upscaler = False
    ):
        """
        Initializes sub-components, handles tensor dimensions, and configures hardware mapping.

        Args:
            model_path (str): File system path to the generator target weights file (.pth or .onnx).
            embedder_model (str): Name string identifier for the target speech representation embedder model.
            embedders_mode (str): Backend selection framework flag (e.g., 'fairseq').
            sid (int): Source/Target speaker unique identifier integer index. Defaults to 0.
            noise_scale (float): Ground truth flow-matching distribution scale variance. Defaults to 0.35.
            checkpointing (bool): Reduces peak memory consumption footprints at the cost of processing speed. Defaults to False.
            hop_length (int): Resolution frame configuration window stride length. Defaults to 160.
            alpha (float): Balancing parameter ratio across underlying modules. Defaults to 0.5.
            predictor_onnx (bool): Routes frequency extraction modules into ONNX runtime engine. Defaults to False.
            clean_audio (bool): Activates noise reduction threshold filter profiles. Defaults to False.
            clean_strength (float): Suppression gain level factor for the acoustic noise gate. Defaults to 0.5.
            audio_upscaler (bool): Flags ultra-high super-resolution networks execution. Defaults to False.
        """

        self.vc = None
        self.index = None
        self.net_g = None 
        self.tgt_sr = None 
        self.big_tsr = None
        self.hubert_model = None
        self.f0_generator = None

        self.alpha = alpha
        self.sample_rate = 16000
        self.hop_length = hop_length
        self.predictor_onnx = predictor_onnx
        # Decide floating point mode based on configuration settings (FP16 or FP32)
        self.dtype = torch.float16 if config.is_half else torch.float32

        # Step 1: Initialize the acoustic embedder model (HuBERT/ContentVec)
        self.setup_hubert(embedder_model, embedders_mode)
        # Step 2: Initialize generator weights (RVC/SVC Synthesizer backbone)
        self.setup_vc(model_path, sid, checkpointing, noise_scale)
        # Step 3: Conditional setup for noise gating modules
        self.tg = TorchGate(self.tgt_sr, prop_decrease=clean_strength).to(config.device) if clean_audio else None
        # Step 4: Conditional setup for audio super-resolution models
        self.flash_sr = FlashSR(os.path.join("assets", "models", "upscalers", "upscalers.pth"), device=config.device, is_half=config.is_half) if audio_upscaler else None

    def convert_audio(
        self, 
        audio_input_path, 
        audio_output_path, 
        index_path, 
        pitch, 
        f0_method, 
        index_rate, 
        rms_mix_rate, 
        protect,  
        f0_autotune, 
        f0_autotune_strength, 
        filter_radius, 
        export_format, 
        resample_sr = 0, 
        f0_file = None, 
        formant_shifting = False, 
        formant_qfrency = 0.8, 
        formant_timbre = 0.8, 
        split_audio = False, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 0, 
        audio_processing = False, 
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        nprobe = 1
    ):
        """
        Executes internal pipeline logic on loaded numpy waveforms to yield the 
        timbre-converted audio output.

        Args:
            audio_input_path (str): File path pointing to the original source audio sample.
            audio_output_path (str): Path indicating where the final synthetic file will sit.
            index_path (str): Matching feature indexing database vector path file (.index).
            pitch (int): Transposition factor interval shifts in semitone quantities.
            f0_method (str): Algorithm method name chosen to construct pitch tracks.
            index_rate (float): Distance blending contribution ratio scale from FAISS query vectors.
            rms_mix_rate (float): Volume dynamic envelope mixing factor scaling coefficient.
            protect (float): Structural unvoiced phonemes and breath protect boundary threshold.
            f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
            f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
            filter_radius (int): Window width used to compute median filter smoothing on tracking logs.
            export_format (str): Audio output wrapper container metadata encoding format type.
            resample_sr (int): Secondary forced outbound sample rate adjustment stage. Defaults to 0.
            f0_file (str): Overriding custom timeline array values text file source. Defaults to None.
            formant_shifting (bool): Pre-adjusts speech format structure early in the loading block. Defaults to False.
            formant_qfrency (float): Shape resonance quality bandwidth adjustments mapping. Defaults to 0.8.
            formant_timbre (float): Vowel structure timbre placement shift multiplier values. Defaults to 0.8.
            split_audio (bool): Cuts dense sound files at low silence nodes into smaller batches. Defaults to False.
            proposal_pitch (bool): Enable automatic pitch key shifting calculation based on median F0 alignment.
            proposal_pitch_threshold (float): The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.
            audio_processing (bool): Activates signal equalization enhancement steps. Defaults to False.
            embedders_mix (bool): Accumulates discrete deep layered representation layers together. Defaults to False.
            embedders_mix_layers (int): Upper layer range constraint target slicing amount. Defaults to 9.
            embedders_mix_ratio (float): Blending step coefficient ratio for the mixed representation values. Defaults to 0.5.
            nprobe (int): Search depth parameter applied inside indexed space buckets. Defaults to 1.
        """

        def inference(audio, index, big_tsr, f0_file, pbar):
            """Executes inner pipeline process flow call."""

            return self.vc.pipeline(
                audio=audio, 
                f0_up_key=pitch, 
                f0_method=f0_method, 
                index=index,
                big_tsr=big_tsr, 
                index_rate=index_rate, 
                filter_radius=filter_radius, 
                rms_mix_rate=rms_mix_rate, 
                protect=protect, 
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength, 
                f0_file=f0_file, 
                pbar=pbar, 
                proposal_pitch=proposal_pitch,
                proposal_pitch_threshold=proposal_pitch_threshold,
                embedders_mix=embedders_mix, 
                embedders_mix_layers=embedders_mix_layers, 
                embedders_mix_ratio=embedders_mix_ratio,
            )

        try:
            with tqdm(total=10, desc=translations["convert_audio"], ncols=100, unit="a", leave=not split_audio) as pbar:
                # Load input audio with optional early formant-shifting augmentations
                audio = load_audio(audio_input_path, sample_rate=self.sample_rate, formant_shifting=formant_shifting, formant_qfrency=formant_qfrency, formant_timbre=formant_timbre)
                if audio_processing: audio = preprocess(audio, self.sample_rate)

                # Peak volume normalization safeguard block
                try:
                    audio_max = np.abs(audio).max() / 0.95
                    if audio_max > 1: audio /= audio_max
                except:
                    # In case of numeric failures, copy source file directly as a fallback measure
                    import shutil

                    shutil.copy(audio_input_path, audio_output_path)
                    return
                
                # Dynamically load FAISS indices if vectors are required but absent
                if index_rate != 0 and (self.index is None or self.big_tsr is None):
                    self.index, self.big_tsr = load_faiss_index(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"), nprobe, cpu_mode=True)

                pbar.update(1)
                # Chunk splitting strategy for long or intensive files
                if split_audio:
                    pbar.close()
                    # Segment audio files based on -60dB silence thresholds
                    chunks = cut(audio, self.sample_rate, db_thresh=-60, min_interval=500)

                    logger.info(f"{translations['split_total']}: {len(chunks)}")
                    # Instantiate recalculated chunk-length progress tracker bars
                    pbar = tqdm(total=len(chunks) * 5 + 4, desc=translations["convert_audio"], ncols=100, unit="a", leave=True)

                pbar.update(1)

                # Perform the underlying voice inference operations
                if split_audio:
                    converted_chunks = [(start, end, inference(waveform, index=self.index, big_tsr=self.big_tsr, f0_file=None, pbar=pbar)) for waveform, start, end in chunks]
                    # Stitch converted audio pieces back together seamlessly
                    audio_output = restore(converted_chunks, total_len=len(audio), dtype=converted_chunks[0][2].dtype)
                else: audio_output = inference(audio, index=self.index, big_tsr=self.big_tsr, f0_file=f0_file, pbar=pbar)

                pbar.update(1)

                # Acoustic post-processing filtering & Noise reduction gate evaluation
                if audio_processing: audio_output = postprocess(audio_output, self.tgt_sr)
                if self.tg is not None: audio_output = self.tg(torch.from_numpy(audio_output).unsqueeze(0).to(config.device).float()).squeeze(0).cpu().detach().numpy()

                audio_output_resample = None
                target_len = int(np.round(len(audio) / self.sample_rate * self.tgt_sr))
                # Polyphase resampling filter alignment if length mismatches are observed
                if len(audio_output) != target_len: audio_output = signal.resample_poly(audio_output, target_len, len(audio_output))

                # Handle upscaling super-resolution routines or standard resampling
                if self.flash_sr is not None:
                    audio_output_resample = self.flash_sr.upscaler(audio_output, sample_rate=self.tgt_sr, pbar=pbar)
                    self.tgt_sr = 192000 # FlashSR outputs ultra-high fidelity audio
                elif self.tgt_sr != resample_sr and resample_sr > 0: 
                    audio_output_resample = librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=resample_sr, res_type="soxr_vhq")
                    self.tgt_sr = resample_sr

                pbar.update(1)

                # Export audio array block back into a physical file on disk
                try:
                    sf.write(
                        audio_output_path, 
                        audio_output if audio_output_resample is None else audio_output_resample, 
                        self.tgt_sr, 
                        format=export_format
                    )
                except:
                    # Secondary fallback to highly compatible standard 48kHz sampling rate if hardware/drivers object
                    logger.info(translations["sr_not_support"])

                    sf.write(
                        audio_output_path, 
                        librosa.resample(audio_output, orig_sr=self.tgt_sr, target_sr=48000, res_type="soxr_vhq"), 
                        48000, 
                        format=export_format
                    )

                pbar.update(1)
        except Exception as e:
            import traceback

            logger.debug(traceback.format_exc())
            logger.error(translations["error_convert"].format(e=e))
    
    def setup_hubert(self, embedder_model, embedders_mode):
        """Loads and optimizes the discrete feature representation model."""

        models = load_embedders_model(embedder_model, embedders_mode)

        if isinstance(models, torch.nn.Module): 
            models = models.to(config.device).to(self.dtype).eval()
            # Compile model via TorchInductor if requested for accelerated graph execution
            if config.compile_all: models = torch.compile(models, mode=config.compile_mode)

        self.hubert_model = models
    
    def setup_predictor(self):
        """Prepares the fundamental frequency (F0) tracking generator module."""

        from main.library.predictors.Generator import Generator

        self.f0_generator = Generator(
            self.sample_rate, 
            self.hop_length, 
            config.configs.get("f0_min", 50), 
            config.configs.get("f0_max", 1100), 
            self.alpha, 
            config.is_half, 
            config.device, 
            self.predictor_onnx
        )

    def setup_vc(self, weight_root, sid, checkpointing, noise_scale):
        """Loads and configures the target voice synthesizer architectures."""
    
        model = load_model(weight_root)

        if weight_root.endswith(".pth"):
            # Standard .pth PyTorch model pipeline
            from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

            self.tgt_sr = model["config"][-1]
            model["config"][-3] = model["weight"]["emb_g.weight"].shape[0]

            use_f0 = model.get("f0", 1)
            version = model.get("version", "v1")
            vocoder = model.get("vocoder", "Default")
            hidden_dim = 768 if version == "v2" else 256

            # Differentiate RVC vs SVC model architectures
            if model.get("architecture", "RVC"):
                self.net_g = Synthesizer(
                    *model["config"], 
                    use_f0=use_f0, 
                    text_enc_hidden_dim=hidden_dim, 
                    vocoder=vocoder, 
                    checkpointing=checkpointing
                )
            else:
                self.net_g = SynthesizerSVC(
                    *model["config"], 
                    text_enc_hidden_dim=hidden_dim, 
                    vocoder=vocoder, 
                    checkpointing=checkpointing, 
                    noise_scale=noise_scale
                )

            # Inject weights into neural network state dictionary architectures
            self.net_g.load_state_dict(model["weight"], strict=False)
            self.net_g.eval().to(config.device).to(self.dtype)
            self.net_g.remove_weight_norm()
            del self.net_g.enc_q

            if config.compile_all: self.net_g = torch.compile(self.net_g, mode=config.compile_mode)
        else: # ONNX model fallback graph parsing routes
            self.net_g = model.to(config.device)
            self.tgt_sr = model.cpt.get("tgt_sr", 32000)
            use_f0 = model.cpt.get("f0", 1)
            version = model.cpt.get("version", "v1")

        # Initialize tracking predictors if pitch computation is flagged
        if use_f0: self.setup_predictor()
        # Wrap configurations directly into runtime Pipeline components
        sid = torch.tensor(sid, device=config.device).unsqueeze(0).long()
        self.vc = Pipeline(self.tgt_sr, config, self.net_g, self.hubert_model, self.f0_generator, version, sid, self.dtype)

if __name__ == "__main__": main()