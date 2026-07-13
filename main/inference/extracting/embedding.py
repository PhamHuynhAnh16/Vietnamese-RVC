import os
import sys
import time
import tqdm
import torch
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from main.library.audio.audio import load_audio
from main.app.variables import logger, translations, config
from main.inference.extracting.setup_path import setup_paths
from main.library.utils import load_embedders_model, extract_features

def process_file_embedding(
    files, 
    embedder_model, 
    embedders_mode, 
    device, 
    version, 
    is_half, 
    threads,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5
):
    """
    Processes a slice of audio files on a designated device to extract content embeddings.

    Uses a thread pool executor internally to parallelize audio reading, model 
    inference, and `.npy` matrix writes to disk.

    Args:
        files (list): A list of tuples containing (source_wav_path, target_output_directory).
        embedder_model (str): Name or path identifier of the embedder model.
        embedders_mode (str): Backend framework format (e.g., 'fairseq', 'onnx').
        device (str/torch.device): Target device used for model evaluations (e.g., 'cuda:0', 'cpu').
        version (str): RVC/SVC framework version ('v1' or 'v2').
        is_half (bool): If True, loads and processes via half-precision float16 tensor weights.
        threads (int): Maximum number of concurrent worker threads inside this worker process.
        embedders_mix (bool, optional): Enables multi-layer blending features. Defaults to False.
        embedders_mix_layers (int, optional): The target layer boundary index to mix. Defaults to 9.
        embedders_mix_ratio (float, optional): Alpha/mixing factor for layers. Defaults to 0.5.
    """

    # Select working data precision type
    dtype = torch.float16 if is_half else torch.float32
    # Load and initialize the selected acoustic embedding model architecture
    model = load_embedders_model(
        embedder_model, 
        embedders_mode
    )
    # If the loaded instance is a PyTorch native module, move it to target device and evaluate mode
    if isinstance(model, torch.nn.Module): 
        model = model.to(device).to(dtype).eval()
        if config.compile_all: model = torch.compile(model, mode=config.compile_mode)

    def worker(file_info):
        """Inner worker thread executing individual audio file processing safely."""

        try:
            file, out_path = file_info
            # Build full destination path; handles both unified dir structures or direct targets
            out_file_path = os.path.join(
                out_path, 
                os.path.basename(file.replace("wav", "npy"))
            ) if os.path.isdir(out_path) else out_path

            # Skip execution if features were computed and saved previously
            if os.path.exists(out_file_path): return

            feats = torch.from_numpy(
                load_audio(file, sample_rate=16000)
            ).to(device).to(dtype)

            # Pass features to model without gradient historical graphs tracking
            with torch.no_grad():
                feats = extract_features(
                    model, 
                    feats.view(1, -1), 
                    version, 
                    mix=embedders_mix, 
                    mix_layers=embedders_mix_layers, 
                    mix_ratio=embedders_mix_ratio
                )

            # Squeeze batch dimensions, revert back to system CPU memory space as floating type numpy matrix
            feats = feats.squeeze(0).float().cpu().numpy()
            # Confirm integrity of computed array matrix (Check for corruption/NaNs)
            if not np.isnan(feats).any(): 
                np.save(
                    out_file_path, 
                    feats, 
                    allow_pickle=False
                )
            else: logger.warning(f"{file} {translations['NaN']}")
        except:
            # Safely capture operational anomalies without blocking sister parallel execution threads
            logger.debug(traceback.format_exc())

    # Dispatch tasks to thread-pool and update terminal interactive progress bar status metrics
    with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                pbar.update(1)

def run_embedding_extraction(
    exp_dir, 
    version, 
    num_processes, 
    devices, 
    embedder_model, 
    embedders_mode, 
    is_half,
    embedders_mix = False,
    embedders_mix_layers = 9,
    embedders_mix_ratio = 0.5
):
    """
    Orchestrates embedding extraction across the entire dataset using multi-processing.

    Splits the workload seamlessly across available compute instances (GPUs/CPUs/XPUs).

    Args:
        exp_dir (str): Main logging/experiment home directory path.
        version (str): Target framework execution variant.
        num_processes (int): Upper bound limits on allocated thread distributions.
        devices (list): String IDs of target logical device pools to schedule jobs on.
        embedder_model (str): Name or path identifier of the embedder model.
        embedders_mode (str): Model backend platform (e.g., 'fairseq', 'onnx').
        is_half (bool): Flag indicating half-precision application.
        embedders_mix (bool, optional): Layer mixing feature status flag. Defaults to False.
        embedders_mix_layers (int, optional): Boundary index for blending. Defaults to 9.
        embedders_mix_ratio (float, optional): Alpha ratio blend configurations. Defaults to 0.5.
    """

    # Setup working folders according to the standard project tree configuration structure
    wav_path, out_path = setup_paths(exp_dir, version)
    logger.info(translations["start_extract_hubert"])

    if ( # Fallback to single process execution if specific hardware/ONNX constraints exist
        (config.device.startswith("ocl") and embedders_mode == "onnx") or 
        config.device.startswith("privateuseone")
    ): 
        num_processes = 1 

    # Collect and sort all valid source wave file paths within dataset
    paths = sorted([
        (os.path.join(wav_path, file), out_path) 
        for file in os.listdir(wav_path) if file.endswith(".wav")
    ])

    start_time = time.time()
    # Spawn a ProcessPoolExecutor containing one dedicated process per computing device
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([
            executor.submit(
                process_file_embedding, 
                paths[i::len(devices)], # Slice paths to distribute subset evenly among devices
                embedder_model, embedders_mode, 
                devices[i], 
                version, 
                is_half, 
                num_processes // len(devices), # Split thread counts proportionally per device process
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio
            ) 
            for i in range(len(devices))
        ])

    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))

def create_mute_file(version, embedder_model, embedders_mode, is_half):
    """
    Generates a reference embedding matrix for a standard silent/mute audio file.

    This silent file is essential for padding or stabilizing RVC/SVC context sequences.

    Args:
        version (str): RVC/SVC version ('v1' or 'v2').
        embedder_model (str): Name or path identifier of the embedder model.
        embedders_mode (str): Model format (e.g., 'fairseq', 'onnx').
        is_half (bool): Use half precision if supported.
    """

    start_time = time.time()
    logger.info(translations["start_extract_hubert"])

    # Extract embedding for the static dummy mute file path mapping
    process_file_embedding(
        [(
            os.path.join(config.configs["logs_path"], "mute", "sliced_audios_16k", "mute.wav"), 
            os.path.join(config.configs["logs_path"], "mute", f"{version}_extracted", f"mute_{embedder_model}.npy")
        )], 
        embedder_model, 
        embedders_mode, 
        config.device, 
        version, 
        is_half, 
        1 # Single thread limit is sufficient for a single small item
    )

    # Force cleanup after background operations wrap up
    logger.info(translations["extract_hubert_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))