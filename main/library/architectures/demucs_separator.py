import os
import sys
import yaml
import torch

import numpy as np
import typing as tp

from pathlib import Path
from hashlib import sha256

now_dir = os.getcwd()
sys.path.append(now_dir)

from main.configs.config import Config
from main.library.uvr5_separator import spec_utils
from main.library.uvr5_separator.demucs.hdemucs import HDemucs
from main.library.uvr5_separator.demucs.states import load_model
from main.library.uvr5_separator.demucs.apply import BagOfModels, Model
from main.library.uvr5_separator.common_separator import CommonSeparator
from main.library.uvr5_separator.demucs.apply import apply_model, demucs_segments


translations = Config().translations

DEMUCS_4_SOURCE = ["drums", "bass", "other", "vocals"]

DEMUCS_2_SOURCE_MAPPER = {
    CommonSeparator.INST_STEM: 0, 
    CommonSeparator.VOCAL_STEM: 1
}

DEMUCS_4_SOURCE_MAPPER = {
    CommonSeparator.BASS_STEM: 0, 
    CommonSeparator.DRUM_STEM: 1, 
    CommonSeparator.OTHER_STEM: 2, 
    CommonSeparator.VOCAL_STEM: 3
}

DEMUCS_6_SOURCE_MAPPER = {
    CommonSeparator.BASS_STEM: 0,
    CommonSeparator.DRUM_STEM: 1,
    CommonSeparator.OTHER_STEM: 2,
    CommonSeparator.VOCAL_STEM: 3,
    CommonSeparator.GUITAR_STEM: 4,
    CommonSeparator.PIANO_STEM: 5,
}


REMOTE_ROOT = Path(__file__).parent / "remote"

PRETRAINED_MODELS = {
    "demucs": "e07c671f",
    "demucs48_hq": "28a1282c",
    "demucs_extra": "3646af93",
    "demucs_quantized": "07afea75",
    "tasnet": "beb46fac",
    "tasnet_extra": "df3777b2",
    "demucs_unittest": "09ebc15f",
}


sys.path.insert(0, os.path.join(os.getcwd(), "main", "library", "uvr5_separator"))

AnyModel = tp.Union[Model, BagOfModels]


class DemucsSeparator(CommonSeparator):
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
        stem_source = None

        inst_source = {}

        self.audio_file_path = audio_file_path
        self.audio_file_base = os.path.splitext(os.path.basename(audio_file_path))[0]

        self.logger.debug(translations["prepare_mix"])
        mix = self.prepare_mix(self.audio_file_path)

        self.logger.debug(translations["demix"].format(shape=mix.shape))

        self.logger.debug(translations["cancel_mix"])

        self.demucs_model_instance = HDemucs(sources=DEMUCS_4_SOURCE)
        self.demucs_model_instance = get_demucs_model(name=os.path.splitext(os.path.basename(self.model_path))[0], repo=Path(os.path.dirname(self.model_path)))
        self.demucs_model_instance = demucs_segments(self.segment_size, self.demucs_model_instance)
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
            source_reshape = spec_utils.reshape_sources(inst_source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]], source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]])
            inst_source[self.demucs_source_map[CommonSeparator.VOCAL_STEM]] = source_reshape
            source = inst_source

        if isinstance(source, np.ndarray):
            source_length = len(source)
            self.logger.debug(translations["source_length"].format(source_length=source_length))

            match source_length:
                case 2:
                    self.logger.debug(translations["set_map"].format(part="2"))
                    self.demucs_source_map = DEMUCS_2_SOURCE_MAPPER
                case 6:
                    self.logger.debug(translations["set_map"].format(part="6"))
                    self.demucs_source_map = DEMUCS_6_SOURCE_MAPPER
                case _:
                    self.logger.debug(translations["set_map"].format(part="2"))
                    self.demucs_source_map = DEMUCS_4_SOURCE_MAPPER

        self.logger.debug(translations["process_all_part"])

        for stem_name, stem_value in self.demucs_source_map.items():
            if self.output_single_stem is not None:
                if stem_name.lower() != self.output_single_stem.lower():
                    self.logger.debug(translations["skip_part"].format(stem_name=stem_name, output_single_stem=self.output_single_stem))
                    continue

            stem_path = os.path.join(f"{self.audio_file_base}_({stem_name})_{self.model_name}.{self.output_format.lower()}")
            stem_source = source[stem_value].T

            self.final_process(stem_path, stem_source, stem_name)
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
            sources = apply_model(model=self.demucs_model_instance, mix=mix_infer[None], shifts=self.shifts, split=self.segments_enabled, overlap=self.overlap, static_shifts=1 if self.shifts == 0 else self.shifts, set_progress_bar=None, device=self.torch_device, progress=True)[0]

        sources = (sources * ref.std() + ref.mean()).cpu().numpy()
        sources[[0, 1]] = sources[[1, 0]]

        processed[mix] = sources[:, :, 0:None].copy()

        sources = list(processed.values())
        sources = [s[:, :, 0:None] for s in sources]
        sources = np.concatenate(sources, axis=-1)
        return sources


class ModelOnlyRepo:
    def has_model(self, sig: str) -> bool:
        raise NotImplementedError()

    def get_model(self, sig: str) -> Model:
        raise NotImplementedError()


class RemoteRepo(ModelOnlyRepo):
    def __init__(self, models: tp.Dict[str, str]):
        self._models = models

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            url = self._models[sig]
        except KeyError:
            raise RuntimeError(translations["not_found_model_signature"].format(sig=sig))
        
        pkg = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)
        return load_model(pkg)


class LocalRepo(ModelOnlyRepo):
    def __init__(self, root: Path):
        self.root = root
        self.scan()

    def scan(self):
        self._models = {}
        self._checksums = {}

        for file in self.root.iterdir():
            if file.suffix == ".th":
                if "-" in file.stem:
                    xp_sig, checksum = file.stem.split("-")
                    self._checksums[xp_sig] = checksum
                else: xp_sig = file.stem

                if xp_sig in self._models: raise RuntimeError(translations["del_all_but_one"].format(xp_sig=xp_sig))
                
                self._models[xp_sig] = file

    def has_model(self, sig: str) -> bool:
        return sig in self._models

    def get_model(self, sig: str) -> Model:
        try:
            file = self._models[sig]
        except KeyError:
            raise RuntimeError(translations["not_found_model_signature"].format(sig=sig))
        
        if sig in self._checksums: check_checksum(file, self._checksums[sig])

        return load_model(file)


class BagOnlyRepo:
    def __init__(self, root: Path, model_repo: ModelOnlyRepo):
        self.root = root
        self.model_repo = model_repo
        self.scan()

    def scan(self):
        self._bags = {}

        for file in self.root.iterdir():
            if file.suffix == ".yaml": self._bags[file.stem] = file

    def has_model(self, name: str) -> bool:
        return name in self._bags

    def get_model(self, name: str) -> BagOfModels:
        try:
            yaml_file = self._bags[name]
        except KeyError:
            raise RuntimeError(translations["name_not_pretrained"].format(name=name))
        
        bag = yaml.safe_load(open(yaml_file))
        signatures = bag["models"]
        models = [self.model_repo.get_model(sig) for sig in signatures]

        weights = bag.get("weights")
        segment = bag.get("segment")

        return BagOfModels(models, weights, segment)


class AnyModelRepo:
    def __init__(self, model_repo: ModelOnlyRepo, bag_repo: BagOnlyRepo):
        self.model_repo = model_repo
        self.bag_repo = bag_repo

    def has_model(self, name_or_sig: str) -> bool:
        return self.model_repo.has_model(name_or_sig) or self.bag_repo.has_model(name_or_sig)

    def get_model(self, name_or_sig: str) -> AnyModel:
        if self.model_repo.has_model(name_or_sig): return self.model_repo.get_model(name_or_sig)
        else: return self.bag_repo.get_model(name_or_sig)


def check_checksum(path: Path, checksum: str):
    sha = sha256()

    with open(path, "rb") as file:
        while 1:
            buf = file.read(2**20)
            if not buf: break

            sha.update(buf)

    actual_checksum = sha.hexdigest()[: len(checksum)]

    if actual_checksum != checksum: raise RuntimeError(translations["invalid_checksum"].format(path=path, checksum=checksum, actual_checksum=actual_checksum))


def _parse_remote_files(remote_file_list) -> tp.Dict[str, str]:
    root: str = ""
    models: tp.Dict[str, str] = {}

    for line in remote_file_list.read_text().split("\n"):
        line = line.strip()

        if line.startswith("#"): continue
        elif line.startswith("root:"): root = line.split(":", 1)[1].strip()
        else:
            sig = line.split("-", 1)[0]
            assert sig not in models

            models[sig] = "https://dl.fbaipublicfiles.com/demucs/mdx_final/" + root + line

    return models


def get_demucs_model(name: str, repo: tp.Optional[Path] = None):
    if name == "demucs_unittest": return HDemucs(channels=4, sources=DEMUCS_4_SOURCE)

    model_repo: ModelOnlyRepo

    if repo is None:
        models = _parse_remote_files(REMOTE_ROOT / "files.txt")
        model_repo = RemoteRepo(models)
        bag_repo = BagOnlyRepo(REMOTE_ROOT, model_repo)
    else:
        if not repo.is_dir(): print(translations["repo_must_be_folder"].format(repo=repo))

        model_repo = LocalRepo(repo)
        bag_repo = BagOnlyRepo(repo, model_repo)

    any_repo = AnyModelRepo(model_repo, bag_repo)

    model = any_repo.get_model(name)
    model.eval()

    return model