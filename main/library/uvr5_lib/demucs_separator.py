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
    with warnings.catch_warnings() if not config.debug_mode else contextlib.nullcontext():
        if not config.debug_mode: warnings.simplefilter("ignore")
        package = torch.load(model_path, map_location="cpu", weights_only=True)
    
    demucs_fn = HDemucs if modelname.startswith("hdemucs") else HTDemucs
    hybrid = False

    if modelname.startswith("htdemucs_ft"):
        model_list = []
        for i in range(4):
            model = demucs_fn(**package[f"kwargs-{i}"])
            model.load_state_dict(package[f"state-{i}"])
            model.segment = None if segment_size == "Default" or segment_size is None else int(segment_size)
            model_list.append(model)

        model = torch.nn.ModuleList(model_list)
        hybrid = True
    else:
        model = demucs_fn(**package["kwargs"])
        model.load_state_dict(package["state"])
        model.segment = None if segment_size == "Default" or segment_size is None else int(segment_size)

    return model, hybrid

class DemucsSeparator(common_separator.CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
        if config.configs.get("demucs_cpu_mode", False): self.torch_device = torch.device("cpu")
        self.weights = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        self.demucs_model_instance, self.hybrid = load_model(self.model_name, self.model_path, self.segment_size)
        self.demucs_model_instance.to(self.torch_device).eval()

    def cleanup(self):
        del self.demucs_model_instance
        self.demucs_model_instance = None
        clear_gpu_cache()

    def separate(self, audio_file_path):
        source = self.demix_demucs(audio_file_path)
        output_files = []

        source_length = len(source)

        if source_length == 2: self.demucs_source_map = {common_separator.CommonSeparator.INST_STEM: 0, common_separator.CommonSeparator.VOCAL_STEM: 1}
        elif source_length == 6: 
            self.demucs_source_map = {
                common_separator.CommonSeparator.BASS_STEM: 0, 
                common_separator.CommonSeparator.DRUM_STEM: 1, 
                common_separator.CommonSeparator.OTHER_STEM: 2, 
                common_separator.CommonSeparator.VOCAL_STEM: 3, 
                common_separator.CommonSeparator.GUITAR_STEM: 4, 
                common_separator.CommonSeparator.PIANO_STEM: 5
            }
        else: self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        for stem_name, stem_value in self.demucs_source_map.items():
            if self.output_single_stem is not None and stem_name.lower() != self.output_single_stem.lower(): continue
            stem_path = os.path.join(f"{os.path.splitext(os.path.basename(audio_file_path))[0]}_({stem_name})_{self.model_name}.{self.output_format.lower()}")

            self.final_process(stem_path, source[stem_value].T, stem_name)
            output_files.append(stem_path)

        self.cleanup()
        return output_files

    def demix_demucs(self, audio_file_path):
        processed = {}

        mix = torch.tensor(self.prepare_mix(audio_file_path), dtype=torch.float32)
        ref = mix.mean(0)

        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix

        with torch.inference_mode():
            sources = apply.apply_model(
                model=self.demucs_model_instance, 
                mix=mix_infer[None], 
                shifts=self.shifts, 
                split=self.segments_enabled, 
                overlap=self.overlap, 
                static_shifts=max(self.shifts, 1), 
                set_progress_bar=None, 
                device=self.torch_device, 
                progress=True,
                hybrid=self.hybrid,
                weights=self.weights
            )[0]

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        processed[mix] = sources[:, :, 0:None].copy()

        return np.concatenate([s[:, :, 0:None] for s in list(processed.values())], axis=-1)