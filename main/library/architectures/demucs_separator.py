import os
import sys
import yaml
import torch

import numpy as np

from pathlib import Path
from hashlib import sha256

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
from main.library.uvr5_separator import spec_utils, common_separator
from main.library.uvr5_separator.demucs import hdemucs, states, apply

translations = Config().translations

DEMUCS_4_SOURCE = ["drums", "bass", "other", "vocals"]

DEMUCS_2_SOURCE_MAPPER = {
    common_separator.CommonSeparator.INST_STEM: 0, 
    common_separator.CommonSeparator.VOCAL_STEM: 1
}

DEMUCS_4_SOURCE_MAPPER = {
    common_separator.CommonSeparator.BASS_STEM: 0, 
    common_separator.CommonSeparator.DRUM_STEM: 1, 
    common_separator.CommonSeparator.OTHER_STEM: 2, 
    common_separator.CommonSeparator.VOCAL_STEM: 3
}

DEMUCS_6_SOURCE_MAPPER = {
    common_separator.CommonSeparator.BASS_STEM: 0,
    common_separator.CommonSeparator.DRUM_STEM: 1,
    common_separator.CommonSeparator.OTHER_STEM: 2,
    common_separator.CommonSeparator.VOCAL_STEM: 3,
    common_separator.CommonSeparator.GUITAR_STEM: 4,
    common_separator.CommonSeparator.PIANO_STEM: 5,
}

sys.path.insert(0, os.path.join(os.getcwd(), "main", "library", "uvr5_separator"))

class DemucsSeparator(common_separator.CommonSeparator):
    def __init__(self, common_config, arch_config):
        super().__init__(config=common_config)
        self.segment_size = arch_config.get("segment_size", "Default")
        self.shifts = arch_config.get("shifts", 2)
        self.overlap = arch_config.get("overlap", 0.25)
        self.segments_enabled = arch_config.get("segments_enabled", True)
        self.logger.debug(translations["demucs_info"].format(segment_size=self.segment_size, segments_enabled=self.segments_enabled))
        self.logger.debug(translations["demucs_info_2"].format(shifts=self.shifts, overlap=self.overlap))
        self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER
        self.audio_file_path = None
        self.audio_file_base = None
        self.demucs_model_instance = None
        self.logger.info(translations["start_demucs"])

    def separate(self, audio_file_path):
        self.logger.debug(translations["start_separator"])
        source = None
        inst_source = {}
        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]
        self.logger.debug(translations["prepare_mix"])
        mix = self.prepare_mix(self.audio_file_path)
        self.logger.debug(translations["demix"].format(shape=mix.shape))
        self.logger.debug(translations["cancel_mix"])
        self.demucs_model_instance = hdemucs.HDemucs(sources=DEMUCS_4_SOURCE)
        self.demucs_model_instance = get_demucs_model(name=os.path.splitext(os.path.basename(self.model_path))[0], repo=Path(os.path.dirname(self.model_path)))
        self.demucs_model_instance = apply.demucs_segments(self.segment_size, self.demucs_model_instance)
        self.demucs_model_instance.to(self.torch_device)
        self.demucs_model_instance.eval()
        self.logger.debug(translations["model_review"])
        source = self.demix_demucs(mix)
        del self.demucs_model_instance
        self.clear_gpu_cache()
        self.logger.debug(translations["del_gpu_cache_after_demix"])
        output_files = []
        self.logger.debug(translations["process_output_file"])

        if isinstance(inst_source, np.ndarray):
            self.logger.debug(translations["process_ver"])
            inst_source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]] = spec_utils.reshape_sources(inst_source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]], source[self.demucs_source_map[common_separator.CommonSeparator.VOCAL_STEM]])
            source = inst_source

        if isinstance(source, np.ndarray):
            source_length = len(source)
            
            self.logger.debug(translations["source_length"].format(source_length=source_length))
            self.logger.debug(translations["set_map"].format(part=source_length))

            match source_length:
                case 2: self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
                case 6: self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER
                case _: self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        self.logger.debug(translations["process_all_part"])
        for stem_name, stem_value in self.demucs_source_map.items():
            if self.output_single_stem is not None:
                if stem_name.lower() != self.output_single_stem.lower():
                    self.logger.debug(translations["skip_part"].format(stem_name=stem_name, output_single_stem=self.output_single_stem))
                    continue

            stem_path = os.path.join(f"{self.audio_file_base}_({stem_name})_{self.model_name}.{self.output_format.lower()}")
            self.final_process(stem_path, source[stem_value].T, stem_name)
            output_files.append(stem_path)

        return output_files

    def demix_demucs(self, mix):
        self.logger.debug(translations["starting_demix_demucs"])
        processed = {}
        mix = torch.tensor(mix, dtype=torch.float32)
        ref = mix.mean(0)
        mix = (mix - ref.mean()) / ref.std()
        mix_infer = mix

        with torch.no_grad():
            self.logger.debug(translations["model_infer"])
            sources = apply.apply_model(model=self.demucs_model_instance, mix=mix_infer[None], shifts=self.shifts, split=self.segments_enabled, overlap=self.overlap, static_shifts=1 if self.shifts == 0 else self.shifts, set_progress_bar=None, device=self.torch_device, progress=True)[0]

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]
        processed[mix] = sources[:, :, 0:None].copy()
        return np.concatenate([s[:, :, 0:None] for s in list(processed.values())], axis=-1)

class ModelOnlyRepo:
    def has_model(self, sig):
        pass

    def get_model(self, sig):
        pass

class LocalRepo(ModelOnlyRepo):
    def __init__(self, root):
        self.root = root
        self.scan()

    def scan(self):
        self._models, self._checksums = {}, {}

        for file in self.root.iterdir():
            if file.suffix == ".th":
                if "-" in file.stem:
                    xp_sig, checksum = file.stem.split("-")
                    self._checksums[xp_sig] = checksum
                else: xp_sig = file.stem

                if xp_sig in self._models: raise RuntimeError(translations["del_all_but_one"].format(xp_sig=xp_sig))
                self._models[xp_sig] = file

    def has_model(self, sig):
        return sig in self._models

    def get_model(self, sig):
        try:
            file = self._models[sig]
        except KeyError:
            raise RuntimeError(translations["not_found_model_signature"].format(sig=sig))
        
        if sig in self._checksums: check_checksum(file, self._checksums[sig])
        return states.load_model(file)

class BagOnlyRepo:
    def __init__(self, root, model_repo):
        self.root = root
        self.model_repo = model_repo
        self.scan()

    def scan(self):
        self._bags = {}
        for file in self.root.iterdir():
            if file.suffix == ".yaml": self._bags[file.stem] = file

    def get_model(self, name):
        try:
            yaml_file = self._bags[name]
        except KeyError:
            raise RuntimeError(translations["name_not_pretrained"].format(name=name))
        
        bag = yaml.safe_load(open(yaml_file))
        return apply.BagOfModels([self.model_repo.get_model(sig) for sig in bag["models"]], bag.get("weights"), bag.get("segment"))

def check_checksum(path, checksum):
    sha = sha256()

    with open(path, "rb") as file:
        while 1:
            buf = file.read(2**20)
            if not buf: break
            sha.update(buf)

    actual_checksum = sha.hexdigest()[: len(checksum)]
    if actual_checksum != checksum: raise RuntimeError(translations["invalid_checksum"].format(path=path, checksum=checksum, actual_checksum=actual_checksum))

def get_demucs_model(name, repo = None):
    model_repo = LocalRepo(repo)
    return (model_repo.get_model(name) if model_repo.has_model(name) else BagOnlyRepo(repo, model_repo).get_model(name)).eval()