import os
import sys
import torch

import numpy as np

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.spec import Spectrogram

SAMPLE_RATE, N_CLASS = 16000, 360

class DJCM:
    def __init__(
        self, 
        model_path, 
        device = "cpu", 
        is_half = False, 
        onnx = False, 
        svs = False, 
        providers = ["CPUExecutionProvider"], 
        batch_size = 1, 
        segment_len = 5.12, 
        compile_model = False,
        compile_mode = None,
        return_tensor = False,
        f0_min=50, 
        f0_max=1100
    ):
        super(DJCM, self).__init__()
        window_length = 2048 if svs else 1024

        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.DJCM.model import DJCMM

            model = DJCMM(1, 1, 1, svs=svs, window_length=window_length, n_class=N_CLASS)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()

            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.is_half = is_half
        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // int(SAMPLE_RATE // 100))
        self.dtype = torch.float16 if is_half else torch.float32

        self.spec_extractor = Spectrogram(int(SAMPLE_RATE // 100), window_length).to(device)
        self.cents_mapping = np.pad(20 * np.arange(N_CLASS) + 1997.3794084376191, (4, 4))
        if return_tensor: self.cents_mapping = torch.as_tensor(self.cents_mapping, dtype=self.dtype, device=device)

        self.infer = self._infer_onnx if onnx else self._infer_torch
        self.offsets = torch.arange(-4, 5, device=device) if return_tensor else np.arange(-4, 5)
        self.to_local_average_cents = self._to_local_average_cents_tensor if return_tensor else self._to_local_average_cents_array

    def infer_from_audio(self, audio, thred=0.03):
        if not torch.is_tensor(audio): audio = torch.from_numpy(audio).to(self.device)
        if audio.ndim > 1: audio = audio.squeeze()

        with torch.inference_mode():
            segments = self.pad_audio(audio)

            hidden = torch.cat([
                seg[self.seg_frames // 4: int(self.seg_frames * 0.75)] 
                for seg in torch.cat([
                    self.infer(self.spec_extractor(segments[i:i + self.batch_size].float())) 
                    for i in range(0, len(segments), self.batch_size)
                ], dim=0) 
            ], dim=0)[:(audio.shape[-1] // int(SAMPLE_RATE // 100) + 1)].squeeze(0)

            return self.decode(hidden, thred)
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03):
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0

        return f0

    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        seg_len = self.seg_len
        hop = seg_len // 2

        audio_len = audio.shape[-1]
        left_pad = seg_len // 4

        return torch.nn.functional.pad(
            audio, 
            (left_pad, (((audio_len + seg_len - 1) // seg_len + 1) * seg_len + hop) - audio_len - left_pad)
        ).unfold(0, seg_len, hop).unsqueeze(1).contiguous()

    def _infer_torch(self, spec):
        if self.is_half: spec = spec.half()
        return self.model(spec)

    def _infer_onnx(self, spec):
        spec = spec.cpu().numpy().astype(np.float32)

        return torch.as_tensor(
            self.model.run(
                [self.model.get_outputs()[0].name], 
                {self.model.get_inputs()[0].name: spec}
            )[0], 
            device=self.device,
            dtype=self.dtype
        )

    def _to_local_average_cents_array(self, salience, thred=0.05):
        salience = salience.cpu().numpy().astype(np.float32)

        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4

        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[np.arange(salience.shape[0])[:, None], idx]
        devided = np.sum(local_salience * self.cents_mapping[idx], axis=1) / np.sum(local_salience, axis=1)
        devided[np.max(salience, axis=1) <= thred] = 0

        return devided

    def _to_local_average_cents_tensor(self, salience, thred=0.05):
        center = torch.argmax(salience, dim=1)
        salience = torch.nn.functional.pad(salience, (4, 4))
        center += 4
        
        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[torch.arange(salience.shape[0], device=salience.device)[:, None], idx]
        devided = (local_salience * self.cents_mapping[idx]).sum(dim=1) / local_salience.sum(dim=1)
        devided = torch.where(salience.max(dim=1).values <= thred, torch.zeros_like(devided), devided)

        return devided