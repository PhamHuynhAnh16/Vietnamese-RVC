import os
import sys
import torch
import warnings
import contextlib

import numpy as np

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.utils import clear_gpu_cache
from main.library.uvr5_lib.demucs import apply
from main.library.uvr5_lib import common_separator
from main.library.uvr5_lib.demucs.hdemucs import HDemucs
from main.library.uvr5_lib.demucs.htdemucs import HTDemucs

if not config.debug_mode: warnings.filterwarnings("ignore")

DEMUCS_4_SOURCE_MAPPER = {
    common_separator.CommonSeparator.BASS_STEM: 0, 
    common_separator.CommonSeparator.DRUM_STEM: 1, 
    common_separator.CommonSeparator.OTHER_STEM: 2, 
    common_separator.CommonSeparator.VOCAL_STEM: 3
}

def load_model(modelname, model_path, segment_size = None):
    """
    Loads and initializes a Demucs neural network model (HDemucs or HTDemucs) from a file checkpoint.

    Args:
        modelname (str): Name/type of the model (e.g., starts with 'hdemucs' or 'htdemucs_ft').
        model_path (str): File system path to the torch checkpoint (.pth / .ckpt).
        segment_size (str or int, optional): Segment length parameter. Defaults to None ("Default").

    Returns:
        tuple: (model, hybrid)
            - model (torch.nn.Module or torch.nn.ModuleList): The initialized torch model(s).
            - hybrid (bool): True if the architecture uses a FT fine-tuned ensemble model array.
    """

    # Conditionally suppress torch load alerts if debug mode is disabled
    with warnings.catch_warnings() if not config.debug_mode else contextlib.nullcontext():
        if not config.debug_mode: warnings.simplefilter("ignore")
        package = torch.load(model_path, map_location="cpu", weights_only=True)
    
    # Select architecture implementation based on prefix match
    demucs_fn = HDemucs if modelname.startswith("hdemucs") else HTDemucs
    hybrid = False

    # Check if the model is an ensemble fine-tuned model (contains 4 individual sub-models)
    if modelname.startswith("htdemucs_ft"):
        model_list = []
        for i in range(4):
            model = demucs_fn(**package[f"kwargs-{i}"])
            model.load_state_dict(package[f"state-{i}"])
            # Set custom segment size or fall back to default architecture value
            model.segment = None if segment_size == "Default" or segment_size is None else int(segment_size)
            model_list.append(model)

        model = torch.nn.ModuleList(model_list)
        hybrid = True
    else:
        # Standard standalone model checkpoint
        model = demucs_fn(**package["kwargs"])
        model.load_state_dict(package["state"])
        model.segment = None if segment_size == "Default" or segment_size is None else int(segment_size)

    return model, hybrid

class DemucsSeparator(common_separator.CommonSeparator):
    """
    Audio separator specialized for Meta's Demucs (Hybrid / Hybrid Transformer) architectures.
    Inherits global I/O and normalization utilities from CommonSeparator.
    """

    def __init__(self, common_config, arch_config):
        """
        Initializes Demucs architecture configurations, loads parameters, and moves weights to device.

        Args:
            common_config (dict): Core separation pipeline configurations.
            arch_config (dict): Hyperparameters specific to Demucs inference (shifts, overlaps, etc.).
        """

        super().__init__(config=common_config)
        # Demucs-specific inference settings
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
        # Force fallback to CPU if specified in global application configurations
        if config.configs.get("demucs_cpu_mode", False): self.torch_device = torch.device("cpu")
        # Default weights configuration used during prediction blending
        self.weights = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        # Instantiate model structure and populate weights
        self.model, self.hybrid = load_model(self.model_name, self.model_path, self.segment_size)
        self.model.to(self.torch_device).eval()
        # Optimize memory usage with FP16 precision if enabled globally
        if config.is_half: self.model.half()

    def cleanup(self):
        """
        Deletes the model from application memory and forces an explicit cache flush on the GPU.
        """

        del self.model
        self.model = None
        clear_gpu_cache()

    def separate(self, audio_file_path):
        """
        Executes source separation on an audio file, maps output layers, and saves individual stems.

        Args:
            audio_file_path (str): Input file path of the source mix.

        Returns:
            list: List of created output file paths.
        """

        source = self.demix_demucs(audio_file_path)
        source_length = len(source)
        output_files = []

        # Adapt source mapping dictionary based on the number of stems returned by the model
        if source_length == 2: self.demucs_source_map = {common_separator.CommonSeparator.INST_STEM: 0, common_separator.CommonSeparator.VOCAL_STEM: 1}
        elif source_length == 6: self.demucs_source_map = {common_separator.CommonSeparator.BASS_STEM: 0, common_separator.CommonSeparator.DRUM_STEM: 1, common_separator.CommonSeparator.OTHER_STEM: 2, common_separator.CommonSeparator.VOCAL_STEM: 3, common_separator.CommonSeparator.GUITAR_STEM: 4, common_separator.CommonSeparator.PIANO_STEM: 5}
        else: self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        # Export each mapped audio stem to disk
        for stem_name, stem_value in self.demucs_source_map.items():
            # Build clean out-path string: base_name_(StemName)_ModelName.ext
            stem_path = os.path.join(f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_({stem_name})_{self.model_name}.{self.output_format.lower()}")
            # Transpose array back (Channels x Samples -> Samples x Channels) for saving
            self.final_process(stem_path, source[stem_value].T, stem_name)
            output_files.append(stem_path)

        # Free GPU resource footprint
        self.cleanup()
        return output_files

    def demix_demucs(self, audio_file_path):
        """
        Prepares raw audio tensors, standardizes signals, and infers isolated multi-stem sources.

        Args:
            audio_file_path (str): Target mix sound track location.

        Returns:
            np.ndarray: Matrix of separated audio sources.
        """

        processed = {}
        # Load file and initialize as single-precision PyTorch tensor
        mix = torch.tensor(self.prepare_mix(audio_file_path), dtype=torch.float32)
        ref = mix.mean(0)

        # Normalize variance and mean values across signal matrices
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix

        # Disable autograd engine calculation for inference performance gains
        with torch.inference_mode():
            sources = apply.apply_model(
                model=self.model, 
                mix=mix_infer[None], # Add batch dimension
                shifts=self.shifts, 
                split=self.segments_enabled, 
                overlap=self.overlap, 
                static_shifts=max(self.shifts, 1), 
                set_progress_bar=None, 
                device=self.torch_device, 
                progress=True,
                hybrid=self.hybrid,
                weights=self.weights
            )[0] # Extract result from single batch element

        # Reverse scaling normalization transformations and copy array to host RAM
        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        # Swap internal audio channel arrangements to match canonical targets if required
        sources[[0, 1]] = sources[[1, 0]]
        processed[mix] = sources[:, :, 0:None].copy()

        # Concatenate and compile dictionary into unified stacked numpy array tracks
        return np.concatenate([s[:, :, 0:None] for s in list(processed.values())], axis=-1)