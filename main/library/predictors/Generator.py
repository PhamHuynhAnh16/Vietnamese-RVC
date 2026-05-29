import os
import re
import sys
import math
import torch
import parselmouth

import numba as nb
import numpy as np

from scipy.signal import medfilt
from librosa import yin, pyin, piptrack

sys.path.append(os.getcwd())

from main.library.utils import circular_write
from main.library.predictors.CREPE.filter import mean, median
from main.library.predictors.WORLD.SWIPE import swipe, stonemask
from main.app.variables import config, configs, logger, translations

NOTE = [49.00, 51.91, 55.00, 58.27, 61.74, 65.41, 69.30, 73.42, 77.78, 82.41, 87.31, 92.50, 98.00, 103.83, 110.00, 116.54, 123.47, 130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185.00, 196.00, 207.65, 220.00, 233.08, 246.94, 261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392.00, 415.30, 440.00, 466.16, 493.88, 523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880.00, 932.33, 987.77, 1046.50]

def medfilts(x, kernel_size = 3):
    if kernel_size % 2 == 0: raise ValueError
    if x.ndim != 1: raise ValueError

    if torch.is_tensor(x):
        return torch.nn.functional.pad(
            x.unsqueeze(0).unsqueeze(0), 
            (kernel_size // 2, kernel_size // 2), 
            mode="constant", 
            value=0.0
        ).squeeze(0).squeeze(0).unfold(0, kernel_size, 1).median(dim=-1).values
    else:
        return medfilt(x, kernel_size)

def autotune_f0(note_dict, f0, f0_autotune_strength):
    if torch.is_tensor(f0):
        notes = torch.as_tensor(note_dict, dtype=f0.dtype, device=f0.device)
        nearest = notes[torch.cdist(f0[:, None], notes[:, None]).argmin(dim=1)]
    else:
        notes = np.asarray(note_dict, dtype=f0.dtype)
        nearest = notes[np.abs(f0[:, None] - notes).argmin(axis=1)]

    return f0 + (nearest - f0) * f0_autotune_strength


def extract_median_f0(f0):
    f0 = np.where(f0 == 0, np.nan, f0)

    return float(
        np.median(
            np.interp(
                np.arange(len(f0)), 
                np.where(~np.isnan(f0))[0], 
                f0[~np.isnan(f0)]
            )
        )
    )

def proposal_f0_up_key(f0, target_f0 = 155.0, limit = 12):
    if torch.is_tensor(f0): f0 = f0.cpu().numpy()

    try:
        return max(
            -limit, 
            min(
                limit, int(np.round(12 * np.log2(target_f0 / extract_median_f0(f0))))
            )
        )
    except ValueError:
        return 0

@nb.jit(nopython=True)
def post_process(
    f0, 
    f0_up_key, 
    manual_x_pad, 
    f0_mel_min, 
    f0_mel_max, 
    manual_f0 = None
):
    f0 *= pow(2, f0_up_key / 12)

    if manual_f0 is not None:
        start_frame = int(manual_x_pad * 100)

        replace_f0 = np.interp(
            np.linspace(
                manual_f0[:, 0][0], 
                manual_f0[:, 0][-1], 
                len(manual_f0[:, 0])
            ), 
            manual_f0[:, 0], 
            manual_f0[:, 1]
        )

        end_frame = start_frame + len(replace_f0)
        if end_frame > len(f0):
            end_frame = len(f0)
            replace_f0 = replace_f0[:end_frame - start_frame]

        f0[start_frame:end_frame] = replace_f0

    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > 255] = 255

    return np.rint(f0_mel).astype(np.int32), f0

def realtime_post_process(
    f0, 
    pitch, 
    pitchf, 
    f0_up_key = 0, 
    f0_min = 50.0, 
    f0_max = 1100.0
):
    f0 *= 2 ** (f0_up_key / 12)

    f0_mel = 1127.0 * (1.0 + f0 / 700.0).log()
    f0_mel = torch.clip((f0_mel - f0_min) * 254 / (f0_max - f0_min) + 1, 1, 255, out=f0_mel)
    f0_coarse = torch.round(f0_mel, out=f0_mel).long()

    if pitch is not None and pitchf is not None:
        f0_coarse = f0_coarse[3:-1] if f0_coarse.shape[0] > 4 else f0_coarse
        f0 = f0[3:-1] if f0.shape[0] > 4 else f0

        circular_write(f0_coarse, pitch)
        circular_write(f0, pitchf)
    else:
        pitch = f0_coarse
        pitchf = f0

    return pitch.unsqueeze(0), pitchf.unsqueeze(0)

class Generator:
    def __init__(
        self, 
        sample_rate = 16000, 
        hop_length = 160, 
        f0_min = 50, 
        f0_max = 1100, 
        alpha = 0.5, 
        is_half = False, 
        device = "cpu", 
        predictor_onnx = False, 
        return_tensor = False
    ):
        self.alpha = alpha
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.f0_mel_min = 1127 * math.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * math.log(1 + self.f0_max / 700)

        self.window = 160
        self.batch_size = 512
        self.ref_freqs = NOTE
        self.frame_period = self.window / self.sample_rate * 1000
        self.time_step = self.frame_period / 1000

        self.cache_args = None
        self.cache_flags = None
        self.cache_f0_method = None
        self.providers = config.providers
        self.is_half = is_half
        self.device = device

        self.compile_model = config.compile_all
        self.compile_mode = config.compile_mode
        self.predictor_onnx = predictor_onnx
        self.return_tensor = return_tensor

        self.resize = configs.get("turn_on_resize_f0_for_all", False)
        self._resize_f0 = self._resize_tensor_f0 if self.return_tensor else self._resize_array_f0
        self.resize_dtype = torch.float64 if self.device.startswith(("cpu", "cuda")) else torch.float32

    def calculator(
        self, 
        x_pad, 
        f0_method, 
        x, 
        f0_up_key = 0, 
        p_len = None, 
        filter_radius = 3, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        manual_f0 = None, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0
    ):
        if p_len is None: p_len = x.shape[0] // self.window
        compute_fn = self.get_f0_hybrid if "hybrid" in f0_method else self.compute_f0

        f0 = compute_fn(
            f0_method, 
            x, 
            p_len, 
            filter_radius if filter_radius % 2 != 0 else filter_radius + 1
        )
        
        if proposal_pitch: 
            up_key = proposal_f0_up_key(
                f0, 
                proposal_pitch_threshold, 
                configs["limit_f0"]
            )

            logger.debug(translations["proposal_f0"].format(up_key=up_key))
            f0_up_key += up_key

        if f0_autotune: 
            logger.debug(translations["startautotune"])

            f0 = autotune_f0(
                self.ref_freqs, 
                f0, 
                f0_autotune_strength
            )

        return post_process(f0, f0_up_key, x_pad, self.f0_mel_min, self.f0_mel_max, manual_f0)

    def realtime_calculator(
        self, 
        audio, 
        f0_method, 
        pitch, 
        pitchf, 
        f0_up_key = 0, 
        filter_radius = 3, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0
    ):
        p_len = audio.shape[0] // self.window

        f0 = self.compute_f0(
            f0_method,
            audio,
            p_len,
            filter_radius if filter_radius % 2 != 0 else filter_radius + 1
        )

        if f0_autotune: 
            f0 = autotune_f0(
                self.ref_freqs, 
                f0, 
                f0_autotune_strength
            )

        if proposal_pitch: 
            up_key = proposal_f0_up_key(
                f0, 
                proposal_pitch_threshold, 
                configs["limit_f0"]
            )

            f0_up_key += up_key

        return realtime_post_process(f0, pitch, pitchf, f0_up_key, self.f0_min, self.f0_max)

    def _resize_array_f0(self, x, target_len, resize = False):
        if torch.is_tensor(x): x = x.cpu().numpy()
        if not resize or len(x) == target_len: return x

        source = np.array(x)
        source[source < 0.001] = np.nan

        return np.nan_to_num(
            np.interp(
                np.arange(0, len(source) * target_len, len(source)) / target_len, 
                np.arange(0, len(source)), 
                source
            )
        )

    def _resize_tensor_f0(self, x, target_len, resize = False):
        if not torch.is_tensor(x): x = torch.from_numpy(x).to(self.device)
        if not resize or len(x) == target_len: return x

        source = torch.as_tensor(x, dtype=self.resize_dtype, device=self.device).clone()
        source[source < 0.001] = torch.nan
        n = source.shape[0]

        xp = torch.arange(target_len, dtype=self.resize_dtype, device=self.device) * (n / target_len)
        x0 = xp.floor().long()

        y0 = source[x0]
        y1 = source[(x0 + 1).clamp(max=n - 1)]
        w = xp - x0.to(self.resize_dtype)

        return torch.nan_to_num(
            torch.where(
                w == 0, 
                y0, 
                y0 + (y1 - y0) * w
            )
        ) 
    
    def compute_f0(self, f0_method, x, p_len, filter_radius):
        if f0_method != self.cache_f0_method:
            self.cache_f0_method = f0_method
            self.cache_args = f0_method.split("-")
            self.cache_flags = set(self.cache_args)

        if "pm" in self.cache_flags:
            f0 = self.get_f0_pm(
                x, 
                p_len, 
                filter_radius=filter_radius, 
                mode=self.cache_args[1]
            )
        elif self.cache_args[0] in {"harvest", "dio"}:
            f0 = self.get_f0_pyworld(
                x, 
                p_len, 
                filter_radius, 
                self.cache_args[0], 
                use_stonemask="stonemask" in self.cache_flags
            )
        elif "crepe" in self.cache_flags:
            f0 = (
                self.get_f0_mangio_crepe(
                    x, 
                    p_len, 
                    self.cache_args[2]
                )
            ) if "mangio" in self.cache_flags else (
                self.get_f0_crepe(
                    x, 
                    p_len, 
                    self.cache_args[1], 
                    filter_radius=filter_radius
                )
            )
        elif "fcpe" in self.cache_flags:
            f0 = self.get_f0_fcpe(
                x, 
                p_len, 
                legacy="legacy" in self.cache_flags and "previous" not in self.cache_flags, 
                previous="previous" in self.cache_flags, 
                filter_radius=filter_radius
            )
        elif "rmvpe" in self.cache_flags:
            f0 = self.get_f0_rmvpe(
                x, 
                p_len, 
                clipping="clipping" in self.cache_flags, 
                filter_radius=filter_radius, 
                hpa="hpa" in self.cache_flags,
                previous="previous" in self.cache_flags,
                mix="mix" in self.cache_flags
            )
        elif self.cache_args[0] in {"yin", "pyin", "piptrack"}:
            f0 = self.get_f0_librosa(
                x, 
                p_len, 
                mode=self.cache_args[0],
                filter_radius=filter_radius
            )
        elif "swipe" in self.cache_flags:
            f0 = self.get_f0_swipe(
                x, 
                p_len, 
                filter_radius=filter_radius, 
                use_stonemask="stonemask" in self.cache_flags
            )
        elif "penn" in self.cache_flags:
            f0 = (
                self.get_f0_mangio_penn(
                    x, 
                    p_len
                )
            ) if self.cache_args[0] == "mangio" else (
                self.get_f0_penn(
                    x, 
                    p_len, 
                    filter_radius=filter_radius
                )
            )
        elif "djcm" in self.cache_flags:
            f0 = self.get_f0_djcm(
                x, 
                p_len, 
                clipping="clipping" in self.cache_flags, 
                svs="svs" in self.cache_flags, 
                filter_radius=filter_radius
            )
        elif "pesto" in self.cache_flags:
            f0 = self.get_f0_pesto(
                x, 
                p_len
            )
        elif "swift" in self.cache_flags:
            f0 = self.get_f0_swift(
                x, 
                p_len, 
                filter_radius=filter_radius
            )
        else:
            raise ValueError(translations["option_not_valid"])
        
        if isinstance(f0, tuple): f0 = f0[0]
        if "medfilt" in self.cache_flags or "svs" in self.cache_flags: f0 = medfilts(f0, kernel_size=5)

        return f0
    
    def get_f0_hybrid(self, methods_str, x, p_len, filter_radius):
        logger.debug(translations["hybrid_calc"].format(f0_method=methods_str))

        methods_str = re.search(r"hybrid\[(.+)\]", methods_str)
        if methods_str: methods = [method.strip() for method in methods_str.group(1).split("+")]

        n = len(methods)
        f0_stack = []

        for method in methods:
            f0_stack.append(
                self._resize_f0(
                    self.compute_f0(
                        method, 
                        x, 
                        p_len, 
                        filter_radius
                    ),
                    p_len,
                    True
                )
            )
        
        f0_mix = np.zeros(p_len)

        if not f0_stack: return f0_mix
        if len(f0_stack) == 1: return f0_stack[0]

        weights = (1 - np.abs(np.arange(n) / (n - 1) - (1 - self.alpha))) ** 2
        weights /= weights.sum()

        stacked = np.vstack(f0_stack)
        voiced_mask = np.any(stacked > 0, axis=0)

        f0_mix[voiced_mask] = np.exp(
            np.nansum(
                np.log(stacked + 1e-6) * weights[:, None], axis=0
            )[voiced_mask]
        )

        return f0_mix

    def get_f0_pm(self, x, p_len, filter_radius=3, mode="ac"):
        if torch.is_tensor(x): x = x.cpu().numpy()

        pm = parselmouth.Sound(x, self.sample_rate)

        pm_fn = {
            "ac": pm.to_pitch_ac, 
            "cc": pm.to_pitch_cc, 
            "shs": pm.to_pitch_shs
        }.get(mode, pm.to_pitch_ac)

        pitch = (
            pm_fn(
                time_step=self.time_step, 
                voicing_threshold=filter_radius / 10 * 2, 
                pitch_floor=self.f0_min, 
                pitch_ceiling=self.f0_max
            )
        ) if mode != "shs" else (
            pm_fn(
                time_step=self.time_step,
                minimum_pitch=self.f0_min,
                maximum_frequency_component=self.f0_max
            )
        )

        f0 = pitch.selected_array["frequency"]
        pad_size = (p_len - len(f0) + 1) // 2

        if pad_size > 0 or p_len - len(f0) - pad_size > 0:  f0 = np.pad(f0, [[pad_size, p_len - len(f0) - pad_size]], mode="constant")
        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_mangio_crepe(self, x, p_len, model="full"):
        if not hasattr(self, "mangio_crepe"):
            from main.library.predictors.CREPE.CREPE import CREPE

            self.mangio_crepe = CREPE(
                os.path.join(
                    configs["predictors_path"], 
                    f"crepe_{model}.{'onnx' if self.predictor_onnx else 'pth'}"
                ), 
                model_size=model, 
                hop_length=self.hop_length, 
                batch_size=self.hop_length * 2, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                return_periodicity=False,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        x = x.float()
        x /= x.abs().quantile(0.999)
        audio = x.unsqueeze(dim=0)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0 = self.mangio_crepe.compute_f0(audio.detach(), pad=True).squeeze(0)

        return self._resize_f0(f0, p_len, True)
    
    def get_f0_crepe(self, x, p_len, model="full", filter_radius=3):
        if not hasattr(self, "crepe"):
            from main.library.predictors.CREPE.CREPE import CREPE

            self.crepe = CREPE(
                os.path.join(
                    configs["predictors_path"], 
                    f"crepe_{model}.{'onnx' if self.predictor_onnx else 'pth'}"
                ), 
                model_size=model, 
                hop_length=self.window, 
                batch_size=self.batch_size, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                sample_rate=self.sample_rate, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                return_periodicity=True,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        f0, pd = self.crepe.compute_f0(x.float().unsqueeze(0), pad=True)
        f0, pd = mean(f0, filter_radius), median(pd, filter_radius)
        f0[pd < 0.1] = 0

        return self._resize_f0(f0[0], p_len, self.resize)
    
    def get_f0_fcpe(self, x, p_len, legacy=False, previous=False, filter_radius=3):
        if not hasattr(self, "fcpe"): 
            from main.library.predictors.FCPE.FCPE import FCPE

            self.fcpe = FCPE(
                configs, 
                os.path.join(
                    configs["predictors_path"], 
                    (
                        "fcpe_legacy" 
                        if legacy else 
                        ("fcpe" if previous else "ddsp_200k")
                    ) + (".onnx" if self.predictor_onnx else ".pt")
                ), 
                hop_length=self.hop_length, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                dtype=torch.float32, 
                device=self.device, 
                threshold=(
                    filter_radius / 100
                ) if legacy else (
                    filter_radius / 1000 * 2
                ), 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                legacy=legacy,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )
        
        f0 = self.fcpe.compute_f0(x, p_len)
        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_rmvpe(self, x, p_len, clipping=False, filter_radius=3, hpa=False, previous=False, mix=False):
        if not hasattr(self, "rmvpe"): 
            from main.library.predictors.RMVPE.RMVPE import RMVPE

            self.rmvpe = RMVPE(
                os.path.join(
                    configs["predictors_path"], 
                    (
                        (
                            "hpa-rmvpe-76000" 
                            if previous else 
                            "hpa-rmvpe-112000"
                        ) if hpa else ("rmvpe-mix" if mix else "rmvpe")
                    ) + (".onnx" if self.predictor_onnx else ".pt")
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.predictor_onnx, 
                providers=self.providers,
                hpa=hpa,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                enable_chunk=configs.get("enable_rmvpe_chunk", False) or self.device.startswith(("privateuseone", "ocl")),
                chunk_size=configs.get("rmvpe_chunk_size", 8000),
                return_tensor=self.return_tensor,
                f0_min=self.f0_min, 
                f0_max=self.f0_max
            )
            if clipping: self.rmvpe.infer_from_audio = self.rmvpe.infer_from_audio_with_pitch

        f0 = self.rmvpe.infer_from_audio(x, thred=filter_radius / 100)
        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_pyworld(self, x, p_len, filter_radius, model="harvest", use_stonemask=True):
        if not hasattr(self, "pw"): 
            from main.library.predictors.WORLD.WORLD import PYWORLD

            self.pw = PYWORLD(
                os.path.join(configs["predictors_path"], "world"), 
                os.path.join(configs["binary_path"], "world.bin"),
                harvest=model == "harvest"
            )
        
        if torch.is_tensor(x): x = x.cpu().numpy()
        x = x.astype(np.double)

        f0, t = self.pw.infer(
            x, 
            fs=self.sample_rate, 
            f0_ceil=self.f0_max, 
            f0_floor=self.f0_min, 
            frame_period=self.frame_period
        )

        if use_stonemask:
            f0 = self.pw.stonemask(
                x, 
                self.sample_rate, 
                t, 
                f0
            )

        if filter_radius > 2 and model == "harvest": f0 = medfilts(f0, filter_radius)
        elif model == "dio":
            for index, pitch in enumerate(f0):
                f0[index] = round(pitch, 1)

        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_swipe(self, x, p_len, filter_radius=3, use_stonemask=True):
        if torch.is_tensor(x): x = x.cpu().numpy()

        f0, t = swipe(
            x.astype(np.float32), 
            self.sample_rate, 
            f0_floor=self.f0_min, 
            f0_ceil=self.f0_max, 
            frame_period=self.frame_period,
            sTHR=filter_radius / 10
        )

        if use_stonemask:
            f0 = stonemask(
                x, 
                self.sample_rate, 
                t, 
                f0
            )

        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_librosa(self, x, p_len, mode="yin", filter_radius=3):
        if torch.is_tensor(x): x = x.cpu().numpy()

        if mode != "piptrack":
            if_yin = mode == "yin"
            yin_fn = yin if if_yin else pyin

            f0 = yin_fn(
                x.astype(np.float32), 
                sr=self.sample_rate, 
                fmin=self.f0_min, 
                fmax=self.f0_max, 
                hop_length=self.hop_length
            )

            if not if_yin: f0 = f0[0]
        else:
            pitches, magnitudes = piptrack(
                y=x.astype(np.float32),
                sr=self.sample_rate,
                fmin=self.f0_min,
                fmax=self.f0_max,
                hop_length=self.hop_length,
                threshold=filter_radius / 10
            )

            f0 = pitches[np.argmax(magnitudes, axis=0), range(magnitudes.shape[1])]

        return self._resize_f0(f0, p_len, True)

    def get_f0_penn(self, x, p_len, filter_radius=3):
        if not hasattr(self, "penn"):
            from main.library.predictors.PENN.PENN import PENN

            self.penn = PENN(
                os.path.join(
                    configs["predictors_path"], 
                    f"fcn.{'onnx' if self.predictor_onnx else 'pt'}"
                ), 
                hop_length=self.window, 
                batch_size=self.batch_size, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        f0, pd = self.penn.compute_f0(x.float().unsqueeze(0))
        f0, pd = mean(f0, filter_radius), median(pd, filter_radius)
        f0[pd < 0.1] = 0

        f0 = medfilts(f0[0], kernel_size=5)
        return self._resize_f0(f0, p_len, True)

    def get_f0_mangio_penn(self, x, p_len):
        if not hasattr(self, "mangio_penn"):
            from main.library.predictors.PENN.PENN import PENN

            self.mangio_penn = PENN(
                os.path.join(
                    configs["predictors_path"], 
                    f"fcn.{'onnx' if self.predictor_onnx else 'pt'}"
                ), 
                hop_length=self.hop_length, 
                batch_size=self.hop_length * 2, 
                f0_min=self.f0_min, 
                f0_max=self.f0_max, 
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx, 
                interp_unvoiced_at=0.1,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        x = x.float()
        x /= x.abs().quantile(0.999)
        audio = x.unsqueeze(dim=0)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0 = self.mangio_penn.compute_f0(audio.detach()).squeeze(0)

        f0 = medfilts(f0, kernel_size=5)
        return self._resize_f0(f0, p_len, True)

    def get_f0_djcm(self, x, p_len, clipping=False, svs=False, filter_radius=3):
        if not hasattr(self, "djcm"): 
            from main.library.predictors.DJCM.DJCM import DJCM
            
            self.djcm = DJCM(
                os.path.join(
                    configs["predictors_path"], 
                    (
                        "djcm-svs" 
                        if svs else 
                        "djcm"
                    ) + (".onnx" if self.predictor_onnx else ".pt")
                ), 
                is_half=self.is_half, 
                device=self.device, 
                onnx=self.predictor_onnx, 
                svs=svs,
                providers=self.providers,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                return_tensor=self.return_tensor,
                f0_min=self.f0_min, 
                f0_max=self.f0_max
            )
            if clipping: self.djcm.infer_from_audio = self.djcm.infer_from_audio_with_pitch
            
        f0 = self.djcm.infer_from_audio(x, thred=filter_radius / 10)
        return self._resize_f0(f0, p_len, self.resize)
    
    def get_f0_swift(self, x, p_len, filter_radius=3):
        if not hasattr(self, "swift"): 
            from main.library.predictors.SWIFT.SWIFT import SWIFT

            self.swift = SWIFT(
                os.path.join(
                    configs["predictors_path"], 
                    "swift.onnx"
                ), 
                fmin=self.f0_min, 
                fmax=self.f0_max, 
                confidence_threshold=filter_radius / 4 + 0.137
            )
        
        if torch.is_tensor(x): x = x.cpu().numpy()

        pitch_hz, _, _ = self.swift.detect_from_array(x)
        return self._resize_f0(pitch_hz, p_len, self.resize)

    def get_f0_pesto(self, x, p_len):
        if not hasattr(self, "pesto"):
            from main.library.predictors.PESTO.PESTO import PESTO

            self.pesto = PESTO(
                os.path.join(
                    configs["predictors_path"], 
                    f"pesto.{'onnx' if self.predictor_onnx else 'pt'}"
                ), 
                step_size=self.frame_period, 
                reduction = "alwa", 
                sample_rate=self.sample_rate, 
                device=self.device, 
                providers=self.providers, 
                onnx=self.predictor_onnx,
                compile_model=self.compile_model,
                compile_mode=self.compile_mode,
                chunk_size=30 * self.sample_rate
            )

        if not torch.is_tensor(x): x = torch.from_numpy(x).to(device=self.device)

        x = x.float()
        x /= x.abs().quantile(0.999)
        audio = x.unsqueeze(dim=0)

        if audio.ndim == 2 and audio.shape[0] > 1: audio = audio.mean(dim=0, keepdim=True).detach()
        f0 = self.pesto.compute_f0(audio.detach())[0].squeeze(0)

        return self._resize_f0(f0, p_len, self.resize)