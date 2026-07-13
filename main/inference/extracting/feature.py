import os
import sys
import tqdm
import time
import traceback
import concurrent.futures

import numpy as np

sys.path.append(os.getcwd())

from main.library.audio.audio import load_audio
from main.inference.extracting.setup_path import setup_paths
from main.app.variables import config, logger, translations, configs

class FeatureInput:
    """
    Handles initialization and state management of pitch calculation pipelines.

    Encapsulates target pitch constraints, hardware state allocations, and lazy 
    initialization of the fundamental frequency generator instance.
    """

    def __init__(
        self, 
        is_half=False, 
        device="cuda"
    ):
        """
        Initializes the FeatureInput processor with default or specified runtime variables.

        Args:
            is_half (bool, optional): Runs calculation under half-precision FP16 if True.
            device (str, optional): Target compute context instance (e.g., 'cuda:0').
        """

        self.sample_rate = 16000
        # Retrieve target frequency boundaries with standard defaults if unassigned
        self.f0_min = configs.get("f0_min", 50)
        self.f0_max = configs.get("f0_max", 1100)
        self.is_half = is_half
        self.device = device
        self.f0_gen = None # Lazy initializer placeholder for the Generator core module

    def process_file(
        self, 
        file_info, 
        f0_method, 
        hop_length, 
        predictor_onnx, 
        f0_autotune, 
        f0_autotune_strength, 
        alpha
    ):
        """
        Calculates and exports coarse and fine fundamental frequency arrays for a single file.

        Args:
            file_info (tuple): Structure containing (input_path, output_coarse, output_fine, file_input_source).
            f0_method (str): Key of the selected algorithm (e.g., 'rmvpe', 'harvest').
            hop_length (int): Hop step frame sizes utilized during extraction.
            predictor_onnx (bool): Toggles ONNX runtime environment execution.
            f0_autotune (bool): Flag indicating autotuning feature toggles.
            f0_autotune_strength (float): Scalar strength multiplier for autotuning modifications.
            alpha (float): Blending factor configuration scalar.
        """

        # Lazy load the Generator engine within the worker process boundaries to avoid serialization traps
        if self.f0_gen is None: 
            from main.library.predictors.Generator import Generator

            self.f0_gen = Generator(
                self.sample_rate, 
                hop_length, 
                self.f0_min, 
                self.f0_max, 
                alpha, 
                self.is_half, 
                self.device, 
                predictor_onnx
            )

        inp_path, opt_path1, opt_path2, file_inp = file_info
        # Avoid redundant operations if the output coarse/fine files already exist on disk
        if os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy"): return

        try:
            # Perform numerical pitch tracking calculation
            pitch, pitchf = self.f0_gen.calculator(
                x_pad=config.x_pad, 
                f0_method=f0_method, 
                x=load_audio(file_inp, sample_rate=self.sample_rate), # Force downsample to 16kHz
                f0_up_key=0, 
                p_len=None, 
                filter_radius=3, 
                f0_autotune=f0_autotune, 
                f0_autotune_strength=f0_autotune_strength, 
                manual_f0=None, 
                proposal_pitch=False, 
                proposal_pitch_threshold=0.0
            )

            # Export extracted matrices safely without saving pickle metadata headers
            np.save(
                opt_path2, 
                pitchf, 
                allow_pickle=False
            )

            np.save(
                opt_path1, 
                pitch, 
                allow_pickle=False
            )
        except Exception as e:
            # Log clear contextual localized target failures without crashing global loops
            logger.info(f"{translations['extract_file_error']} {inp_path}: {e}")
            logger.debug(traceback.format_exc())

    def process_files(
        self, 
        files, 
        f0_method, 
        hop_length, 
        predictor_onnx, 
        device, 
        is_half, 
        threads, 
        f0_autotune, 
        f0_autotune_strength, 
        alpha
    ):
        """
        Processes a block slice of files allocated to a process via internal I/O threadpools.

        Args:
            files (list): Aggregation of file information tuples.
            f0_method (str): Name of target computation algorithm.
            hop_length (int): Frame stride constraints.
            predictor_onnx (bool): ONNX application setting toggles.
            device (str): Process-specific target hardware architecture.
            is_half (bool): Precision mapping controller.
            threads (int): Maximum thread configurations allowed within this process block.
            f0_autotune (bool): Autotune toggle activation state.
            f0_autotune_strength (float): Power modifier for internal autotune operations.
            alpha (float): Scaling modifier configuration.
        """

        # Assign current worker process scope states
        self.device = device
        self.is_half = is_half

        def worker(file_info):
            """Inner scope runner executing thread calculations securely."""

            self.process_file(
                file_info, 
                f0_method, 
                hop_length, 
                predictor_onnx, 
                f0_autotune, 
                f0_autotune_strength, 
                alpha
            )

        # Spin up a ThreadPoolExecutor to maximize CPU utilization during file loading and saving
        with tqdm.tqdm(total=len(files), ncols=100, unit="p", leave=True) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                for _ in concurrent.futures.as_completed([executor.submit(worker, f) for f in files]):
                    pbar.update(1)

def run_pitch_extraction(
    exp_dir, 
    f0_method, 
    hop_length, 
    num_processes, 
    devices, 
    predictor_onnx, 
    is_half, 
    f0_autotune, 
    f0_autotune_strength, 
    alpha
):
    """
    Orchestrates fundamental frequency (F0) extraction across the dataset using multi-processing.
    Splits file batches evenly based on the total number of hardware execution contexts.

    Args:
        exp_dir (str): Main experiment folder location on disk.
        f0_method (str): Core target tracking algorithm moniker.
        hop_length (int): Time-domain step windows.
        num_processes (int): Max CPU thread allocation scale.
        devices (list): Sequence list of operational compute strings.
        predictor_onnx (bool): Flag indicating ONNX model state operations.
        is_half (bool): Half precision enablement controller.
        f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
        f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
        alpha (float): Blending factor configuration variable.
    """

    # Unpack output directory pathways mapped out by structural workspace parameters
    input_root, *output_roots = setup_paths(exp_dir)
    output_root1, output_root2 = output_roots if len(output_roots) == 2 else (output_roots[0], None)

    logger.info(translations["extract_f0_method"].format(num_processes=num_processes, f0_method=f0_method))
    # Throttle process limits to 1 to guarantee stability on non-native platform backends when using complex neural models
    if config.device.startswith(("ocl", "privateuseone")) and (
        "crepe" in f0_method or 
        "fcpe" in f0_method or 
        "rmvpe" in f0_method or 
        "penn" in f0_method or 
        "swift" in f0_method or
        "pesto" in f0_method
    ):
        num_processes = 1

    # Map complete source-to-destination path tuples while filtering out spectator file structures
    paths = [
        (
            os.path.join(input_root, name), 
            os.path.join(output_root1, name) if output_root1 else None, 
            os.path.join(output_root2, name) if output_root2 else None, 
            os.path.join(input_root, name)
        ) 
        for name in sorted(os.listdir(input_root)) 
        if "spec" not in name
    ]

    start_time = time.time()
    feature_input = FeatureInput(is_half=config.is_half, device=config.device)

    # Distribute the dataset slices via a ProcessPoolExecutor (one process mapped per logical computing device)
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(devices)) as executor:
        concurrent.futures.wait([
            executor.submit(
                feature_input.process_files, 
                paths[i::len(devices)], # Stride slicing maps data to devices evenly
                f0_method, 
                hop_length, 
                predictor_onnx, 
                devices[i], 
                is_half, 
                num_processes // len(devices), # Scale down thread distribution per process proportionally
                f0_autotune, 
                f0_autotune_strength, 
                alpha
            ) 
            for i in range(len(devices))
        ])
    
    logger.info(translations["extract_f0_success"].format(elapsed_time=f"{(time.time() - start_time):.2f}"))