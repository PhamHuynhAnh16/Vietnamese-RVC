import os
import sys
import time
import torch
import logging
import librosa
import argparse
import warnings

import numpy as np
import torch.multiprocessing as mp

from tqdm import tqdm
from scipy import signal
from scipy.io import wavfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.getcwd())

from main.library.utils import strtobool
from main.library.audio.audio import load_audio
from main.inference.preprocess.slicer2 import Slicer
from main.app.variables import config, logger, translations, configs, file_types

if not config.debug_mode:
    warnings.filterwarnings("ignore")
    for l in ["numba.core.byteflow", "numba.core.ssa", "numba.core.interpreter"]:
        logging.getLogger(l).setLevel(logging.ERROR)

OVERLAP, MAX_AMPLITUDE, ALPHA, HIGH_PASS_CUTOFF, SAMPLE_RATE_16K = 0.3, 0.9, 0.75, 48, 16000

def parse_arguments():
    """
    Parses command-line arguments for the dataset preprocessing script.

    Returns:
        argparse.Namespace: A namespace object populated with parsed arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--preprocess", action='store_true')
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./dataset")
    parser.add_argument("--sample_rate", type=int, required=True)
    parser.add_argument("--cpu_cores", type=int, default=2)
    parser.add_argument("--cut_preprocess", type=str, default="Automatic")
    parser.add_argument("--process_effects", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_dataset", type=lambda x: bool(strtobool(x)), default=False)
    parser.add_argument("--clean_strength", type=float, default=0.7)
    parser.add_argument("--chunk_len", type=float, default=3.0, required=False)
    parser.add_argument("--overlap_len", type=float, default=0.3, required=False)
    parser.add_argument("--normalization_mode", type=str, default="none", required=False)
    parser.add_argument("--architecture", type=str, default="RVC")

    return parser.parse_args()

class PreProcess:
    """
    Handles audio file parsing, normalization, slicing, and resampling 
    to prepare clean datasets for RVC model training.
    """

    def __init__(
        self, 
        sr, 
        exp_dir, 
        per,
        architecture = "RVC"
    ):
        """
        Initializes the PreProcess class instance with pipeline hyper-parameters.

        Args:
            sr (int): Target sample rate for the processing output.
            exp_dir (str): Root directory where sliced output files will be written.
            per (float): Duration per window block used during automatic splitting.
            architecture (str, optional): Target architecture style. Defaults to "RVC".
        """

        # Configure the voice activity detection slicer depending on the model architecture type
        if architecture == "RVC": slicer_params = {"sr": sr, "threshold": -42, "min_length": 1500, "min_interval": 400, "hop_size": 15, "max_sil_kept": 500}
        else: slicer_params = {"sr": sr, "threshold": -40, "min_length": 7500, "min_interval": 100, "hop_size": 10, "max_sil_kept": 800}

        self.slicer = Slicer(
            **slicer_params
        )
        # Design a 5th-order digital High-Pass Butterworth filter to strip low-end rumbling
        self.b_high, self.a_high = signal.butter(
            N=5, 
            Wn=HIGH_PASS_CUTOFF, 
            btype="high", 
            fs=sr
        )
        self.sr = sr
        self.tg = None
        self.per = per
        self.exp_dir = exp_dir
        # Fall back to CPU processing if non-standard OpenCL/private hardware backends are specified
        self.device = config.device if not config.device.startswith(("ocl", "privateuseone")) else "cpu"
        # Set up processing output locations for source audio rate and downsampled 16kHz tracks
        self.gt_wavs_dir = os.path.join(exp_dir, "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "sliced_audios_16k")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def _normalize_audio(self, audio):
        """
        Normalizes the amplitude profile of a given numpy array track block.

        Args:
            audio (np.ndarray): Single-channel floating-point audio time series.

        Returns:
            np.ndarray: Rescaled audio data, or None if signal clip threshold is breached.
        """

        tmp_max = np.abs(audio).max()
        # Discard corrupted segments or samples with massive distortion artifacts
        if tmp_max > 2.5: return None
        # Dynamic range compression using blending alpha coefficients
        return (audio / tmp_max * (MAX_AMPLITUDE * ALPHA)) + (1 - ALPHA) * audio

    def process_audio_segment(self, normalized_audio, sid, idx0, idx1, normalization_mode):
        """
        Saves a chunked waveform segment into target folders at both full scale and 16kHz sample rates.

        Args:
            normalized_audio (Optional[np.ndarray]): Fragment wave data to preserve.
            sid (int): Speaker identification token label.
            idx0 (int): Base audio index identifier.
            idx1 (int): Sub-segment counter within the current stream partition.
            normalization_mode (str): String flag indicating whether to apply late normalization.
        """

        if normalized_audio is None:
            logger.debug(f"{sid}-{idx0}-{idx1}-filtered")
            return
        
        if normalization_mode == "post": normalized_audio = self._normalize_audio(normalized_audio)
        
        # Output standard target sample rate audio snippet
        wavfile.write(
            os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{idx1}.wav"), 
            self.sr, 
            normalized_audio.astype(np.float32)
        )
        # Resample utilizing SoX high-quality algorithm and save down to a standard 16kHz structure
        wavfile.write(
            os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{idx1}.wav"), 
            SAMPLE_RATE_16K, 
            librosa.resample(
                normalized_audio, 
                orig_sr=self.sr, 
                target_sr=SAMPLE_RATE_16K, 
                res_type="soxr_vhq"
            ).astype(np.float32)
        )

    def simple_cut(
        self, 
        audio, 
        sid, 
        idx0, 
        chunk_len, 
        overlap_len, 
        normalization_mode
    ):
        """
        Splits a single continuous audio vector via uniform chunks and fixed overlaps.

        Args:
            audio (np.ndarray): The raw sound array dataset.
            sid (int): Speaker tracking identification index.
            idx0 (int): Master catalog index referencing global tracks.
            chunk_len (float): Desired block parsing length in seconds.
            overlap_len (float): Overlapping offset margin boundary calculated in seconds.
            normalization_mode (str): Parameter selecting late audio normalizer tracking.
        """

        chunk_length = int(self.sr * chunk_len)
        overlap_length = int(self.sr * overlap_len)
        i = 0

        while i < len(audio):
            chunk = audio[i : i + chunk_length]
            if normalization_mode == "post": chunk = self._normalize_audio(chunk)
            # Ensure that partial trailing tracks matching less than window boundaries are omitted
            if len(chunk) == chunk_length:
                wavfile.write(
                    os.path.join(self.gt_wavs_dir, f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}.wav"), 
                    self.sr, 
                    chunk.astype(np.float32)
                )

                wavfile.write(
                    os.path.join(self.wavs16k_dir, f"{sid}_{idx0}_{i // (chunk_length - overlap_length)}.wav"), 
                    SAMPLE_RATE_16K, 
                    librosa.resample(
                        chunk, 
                        orig_sr=self.sr, 
                        target_sr=SAMPLE_RATE_16K, 
                        res_type="soxr_vhq"
                    ).astype(np.float32)
                )

            i += chunk_length - overlap_length

    def process_audio(
        self, 
        path, 
        idx0, 
        sid, 
        cut_preprocess, 
        process_effects, 
        clean_dataset, 
        clean_strength, 
        chunk_len, 
        overlap_len, 
        normalization_mode
    ):
        """
        Primary execution node for a file unit processing tracking sequence.

        Args:
            path (str): Relative or absolute target path pointing to disk source files.
            idx0 (int): Base reference catalog identifier.
            sid (int): Associated speaker identification marker tag.
            cut_preprocess (str): Target slice configuration directive ("Skip", "Simple", "Automatic").
            process_effects (bool): Enables or disables high pass frequency scrubbing routines.
            clean_dataset (bool): Toggles deep gating modules on input sound tracks.
            clean_strength (float): Dynamic dampening strength targeting signal noises.
            chunk_len (float): Chunk sizing metrics in seconds.
            overlap_len (float): Tracking overlap intervals in seconds.
            normalization_mode (str): Normalization window context flag selection.

        Returns:
            float: Evaluated length of processed signal duration parameter block in seconds.
        """

        dataset_length = 0

        try:
            # Load input sample tracking metrics from memory maps
            audio = load_audio(path, sample_rate=self.sr)
            dataset_length = librosa.get_duration(y=audio, sr=self.sr)

            # Apply high-pass linear filtering routines if requested
            if process_effects: audio = signal.lfilter(self.b_high, self.a_high, audio)
            if normalization_mode == "pre": audio = self._normalize_audio(audio)

            # Implement deep learning noise-gate architectures via PyTorch operations
            if clean_dataset: 
                if self.tg is None: 
                    from main.library.audio.noisereduce import TorchGate

                    self.tg = TorchGate(
                        self.sr, 
                        prop_decrease=clean_strength
                    ).to(self.device)

                # Format data vectors to match expected Torch tensor shapes before gating
                audio = self.tg(
                    torch.from_numpy(audio).unsqueeze(0).to(self.device).float()
                ).squeeze(0).cpu().detach().numpy()

            # Route execution logic branches based on slicing type configuration
            if cut_preprocess == "Skip":
                self.process_audio_segment(
                    audio,
                    sid,
                    idx0,
                    0,
                    normalization_mode,
                )
            elif cut_preprocess == "Simple":
                self.simple_cut(
                    audio,
                    sid,
                    idx0,
                    chunk_len,
                    overlap_len,
                    normalization_mode,
                )
            elif cut_preprocess == "Automatic":
                idx1 = 0
                # Use voice detection boundaries to extract audio components cleanly
                for audio_segment in self.slicer.slice(audio):
                    i = 0

                    while 1:
                        start = int(self.sr * (self.per - OVERLAP) * i)
                        i += 1

                        if len(audio_segment[start:]) > (self.per + OVERLAP) * self.sr:
                            self.process_audio_segment(
                                audio_segment[start : start + int(self.per * self.sr)], 
                                sid, 
                                idx0, 
                                idx1, 
                                normalization_mode
                            )

                            idx1 += 1
                        else:
                            self.process_audio_segment(
                                audio_segment[start:], 
                                sid, 
                                idx0, 
                                idx1, 
                                normalization_mode
                            )

                            idx1 += 1
                            break
        except Exception as e:
            raise RuntimeError(f"{translations['process_audio_error']}: {e}")
        return dataset_length

def format_duration(seconds):
    """
    Converts raw floating-point second arrays into readable timestamps.

    Args:
        seconds (float): Raw metric count denoting total tracked timeline seconds.

    Returns:
        str: Timestamp string formatted cleanly as HH:MM:SS.
    """

    return f"{int(seconds // 3600):02}:{int((seconds % 3600) // 60):02}:{int(seconds % 60):02}"

def process_file(args):
    """
    Unpacks multi-processing pipeline arguments to invoke the core audio preprocessing method.

    Args:
        args (Tuple): Complex zipped container holding configuration metrics, references, and paths.

    Returns:
        float: Calculated temporal duration length values back to tracking monitors.
    """

    (
        pp, 
        file, 
        cut_preprocess, 
        process_effects, 
        clean_dataset, 
        clean_strength, 
        chunk_len, 
        overlap_len, 
        normalization_mode
    ) = args

    file_path, idx0, sid = file

    return pp.process_audio(
        file_path, 
        idx0, 
        sid, 
        cut_preprocess, 
        process_effects, 
        clean_dataset, 
        clean_strength, 
        chunk_len, 
        overlap_len, 
        normalization_mode
    )

def preprocess_training_set(
    input_root, 
    sr, 
    num_processes, 
    exp_dir, 
    per, 
    cut_preprocess, 
    process_effects, 
    clean_dataset, 
    clean_strength, 
    chunk_len, 
    overlap_len, 
    normalization_mode,
    architecture = "RVC"
):
    """
    Crawls raw audio file locations and parallelizes data-preprocessing tasks using concurrent workers.

    Args:
        input_root (str): Source dataset tree traversal path.
        sr (int): Target frequency sample tracking size.
        num_processes (int): Worker pool scale dimensions.
        exp_dir (str): Logging target output path.
        per (float): Sliding partition tracker segment index metrics.
        cut_preprocess (str): Target pipeline operational slicing mode parameter.
        process_effects (bool): High pass filter enabling state flag.
        clean_dataset (bool): Gating routine implementation variable.
        clean_strength (float): Scaling factor for background noise floor attenuation.
        chunk_len (float): Fragment duration scale.
        overlap_len (float): Splitting block margin overlap offsets.
        normalization_mode (str): System sound volume modifier parameters.
        architecture (str): Target baseline neural infrastructure configuration. Defaults to "RVC".
    """

    start_time = time.time()
    logger.info(translations["start_preprocess"].format(num_processes=num_processes))
    pp = PreProcess(sr, exp_dir, per, architecture)
    dataset_length, idx = 0, 0
    files = []

    # Walk through the dataset file directory structure
    for root, _, filenames in os.walk(input_root):
        try:
            # Map sub-folder names to speaker IDs (root folder equals default speaker ID 0)
            sid = 0 if root == input_root else int(os.path.basename(root))
            for f in filenames:
                if f.lower().endswith(tuple(file_types)):
                    files.append((os.path.join(root, f), idx, sid))
                    idx += 1
        except ValueError:
            raise ValueError(f"{translations['not_integer']} '{os.path.basename(root)}'.")

    # Launch multi-core parallel process execution pools
    with tqdm(total=len(files), ncols=100, unit="f") as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(
                    process_file, 
                    (
                        pp, 
                        file, 
                        cut_preprocess, 
                        process_effects, 
                        clean_dataset, 
                        clean_strength, 
                        chunk_len, 
                        overlap_len, 
                        normalization_mode
                    )
                ) 
                for file in files
            ]

            # Aggregate process completions safely and compute aggregate dataset timeline lengths
            for future in as_completed(futures):
                try:
                    dataset_length += future.result() 
                except Exception as e:
                    raise RuntimeError(f"{translations['process_error']}: {e}")
                pbar.update(1)

    elapsed_time = time.time() - start_time
    logger.info(translations["dataset_duration"].format(duration=format_duration(dataset_length)))
    logger.info(translations["preprocess_success"].format(elapsed_time=f"{elapsed_time:.2f}"))

def main():
    """
    Orchestrates the program execution lifecycle, logs runtime parameters,
    and secures cross-process execution tracking using PID locks.
    """

    args = parse_arguments()
    experiment_directory = os.path.join(configs["logs_path"], args.model_name)

    num_processes = args.cpu_cores
    num_processes = 2 if num_processes is None else int(num_processes)

    (
        dataset, 
        sample_rate, 
        cut_preprocess, 
        preprocess_effects, 
        clean_dataset, 
        clean_strength, 
        chunk_len, 
        overlap_len, 
        normalization_mode,
        architecture
    ) = (
        args.dataset_path, 
        args.sample_rate, 
        args.cut_preprocess, 
        args.process_effects, 
        args.clean_dataset, 
        args.clean_strength, 
        args.chunk_len, 
        args.overlap_len, 
        args.normalization_mode,
        args.architecture
    )

    os.makedirs(experiment_directory, exist_ok=True)
    # Dictionary containing configurations to display for verification
    log_data = {
        translations['modelname']: args.model_name, 
        translations['export_process']: experiment_directory, 
        translations['dataset_folder']: dataset, 
        translations['pretrain_sr']: sample_rate, 
        translations['cpu_core']: num_processes, 
        translations['split_audio']: cut_preprocess, 
        translations['preprocess_effect']: preprocess_effects, 
        translations['clear_audio']: clean_dataset,
        translations['clean_strength']: clean_strength,
        translations["architecture"]: architecture
    }

    for key, value in log_data.items():
        logger.debug(f"{key}: {value}")

    # Write a temporary PID tracking file to register the current process instance locks
    pid_path = os.path.join(experiment_directory, "preprocess_pid.txt")
    with open(pid_path, "w") as pid_file:
        pid_file.write(str(os.getpid()))
    
    try:
        preprocess_training_set(
            dataset, 
            sample_rate, 
            num_processes, 
            experiment_directory, 
            config.per_preprocess, 
            cut_preprocess, 
            preprocess_effects, 
            clean_dataset, 
            clean_strength, 
            chunk_len, 
            overlap_len, 
            normalization_mode,
            architecture
        )
    except Exception as e:
        logger.error(f"{translations['process_audio_error']} {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
    # Clean up the runtime PID file upon normal script termination or error handling routines
    if os.path.exists(pid_path): os.remove(pid_path)
    logger.info(f"{translations['preprocess_model_success']} {args.model_name}")

if __name__ == "__main__": 
    # Force 'spawn' multiprocessing method to guarantee cross-platform stability
    mp.set_start_method("spawn", force=True)
    main()