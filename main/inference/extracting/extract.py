import os
import sys
import logging
import argparse
import warnings

import torch.multiprocessing as mp

sys.path.append(os.getcwd())

from main.library.utils import check_assets, strtobool
from main.inference.extracting.feature import run_pitch_extraction
from main.app.variables import config, logger, translations, configs
from main.inference.extracting.embedding import run_embedding_extraction
from main.inference.extracting.preparing_files import generate_config, generate_filelist

if not config.debug_mode:
    warnings.filterwarnings("ignore")
    for l in ["torch", "faiss", "httpx", "httpcore", "faiss.loader", "numba.core", "urllib3", "matplotlib"]:
        logging.getLogger(l).setLevel(logging.ERROR)

def parse_arguments():
    """
    Parses command-line arguments for the extraction process.

    Returns:
        argparse.Namespace: Object containing all parsed command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--rvc_version", type=str, default="v2")
    parser.add_argument("--f0_method", type=str, default="rmvpe")
    parser.add_argument("--pitch_guidance", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--hop_length", type=int, default=128)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--gpu", type=str, default="-")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--embedder_model", type=str, default="hubert_base")
    parser.add_argument("--predictor_onnx", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mode", type=str, default="fairseq")
    parser.add_argument("--f0_autotune", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--f0_autotune_strength", type=float, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--include_mutes", type=int, default=2)
    parser.add_argument("--embedders_mix", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--embedders_mix_layers", type=int, default=9, required=False)
    parser.add_argument("--embedders_mix_ratio", type=float, default=0.5)
    parser.add_argument("--architecture", type=str, default="RVC")

    return parser.parse_args()

def main():
    """
    Main orchestrator function for the feature extraction pipeline.
    Validates assets, maps execution devices, handles logging, and runs 
    the pitch/embedding extractions sequentially.
    """

    args = parse_arguments()
    # Unpack and map command-line arguments to descriptive local variables
    (
        f0_method, 
        hop_length, 
        num_processes, 
        gpus, version, 
        pitch_guidance, 
        sample_rate, 
        embedder_model, 
        predictor_onnx, 
        embedders_mode, 
        f0_autotune, 
        f0_autotune_strength, 
        alpha,
        include_mutes,
        embedders_mix,
        embedders_mix_layers,
        embedders_mix_ratio,
        architecture
    ) = (
        args.f0_method, 
        args.hop_length, 
        args.cpu_cores, 
        args.gpu, 
        args.rvc_version, 
        args.pitch_guidance, 
        args.sample_rate, 
        args.embedder_model, 
        args.predictor_onnx, 
        args.embedders_mode, 
        args.f0_autotune, 
        args.f0_autotune_strength, 
        args.alpha,
        args.include_mutes,
        args.embedders_mix,
        args.embedders_mix_layers,
        args.embedders_mix_ratio,
        args.architecture
    )

    # Step 1: Pre-execution validations (Ensure dependency models/assets exist)
    check_assets(f0_method, embedder_model, predictor_onnx=predictor_onnx, embedders_mode=embedders_mode)
    # Step 2: Establish paths and resource allocation limits
    exp_dir = os.path.join(configs["logs_path"], args.model_name)
    num_processes = max(1, num_processes) # Ensure at least 1 process runs

    # Step 3: Hardware device mapping based on the system backends (CUDA, Intel XPU, OpenCL, DirectML)
    devices = ["cpu"] if gpus == "-" else [
        (
            f"cuda:{idx}"
        ) if config.device.startswith("cuda") else (
            f"xpu:{idx}" if config.device.startswith("xpu") else f"{'ocl' if config.device.startswith('ocl') else 'privateuseone'}:{idx}"
        ) 
        for idx in gpus.split("-")
    ]

    # Step 4: Compile configuration parameters for debug logging
    log_data = {
        translations['modelname']: args.model_name, 
        translations['export_process']: exp_dir, 
        translations['f0_method']: f0_method, 
        translations['pretrain_sr']: sample_rate, 
        translations['cpu_core']: num_processes, 
        "Gpu": gpus, 
        translations['hop_length']: hop_length, 
        translations['training_version']: version, 
        translations['extract_f0']: pitch_guidance, 
        translations['hubert_model']: embedder_model, 
        translations["predictor_onnx"]: predictor_onnx, 
        translations["embed_mode"]: embedders_mode, 
        translations["alpha_label"]: alpha,
        translations["include_mutes"]: include_mutes,
        translations["embedders_mix"]: embedders_mix,
        translations["embedders_mix_layers"]: embedders_mix_layers,
        translations["embedders_mix_ratio"]: embedders_mix_ratio,
        translations["architecture"]: architecture
    }

    # Print the operational parameters to the log file/terminal for tracking
    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    # Step 5: Save current Process ID (PID) to track/kill execution if needed
    pid_path = os.path.join(exp_dir, "extract_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    # Step 6: Sequential Execution of Core Feature Extractions
    try:
        # Extract fundamental frequencies (Pitch tracking)
        run_pitch_extraction(
            exp_dir, 
            f0_method, 
            hop_length, 
            num_processes, 
            devices, 
            predictor_onnx, 
            config.is_half, 
            f0_autotune, 
            f0_autotune_strength, 
            alpha
        )
        # Extract content embeddings
        run_embedding_extraction(
            exp_dir, 
            version, 
            num_processes, 
            devices, 
            embedder_model, 
            embedders_mode, 
            config.is_half,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio
        )
        # Create a training configuration file
        generate_config(
            version, 
            sample_rate, 
            exp_dir
        )
        # Generate the formatted dataset path filelist required by the trainer
        generate_filelist(
            pitch_guidance, 
            exp_dir, 
            version, 
            sample_rate, 
            embedders_mode, 
            embedder_model, 
            include_mutes
        )
    except Exception as e:
        # Log localized error if any processing step crashes
        logger.error(f"{translations['extract_error']}: {e}")

    # Step 7: Post-execution cleanup (Remove PID tracking file)
    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['extract_success']} {args.model_name}.")

if __name__ == "__main__": 
    # Force 'spawn' multiprocessing method to guarantee cross-platform stability
    mp.set_start_method("spawn", force=True)
    main()