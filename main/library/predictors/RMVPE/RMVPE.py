import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.predictors.RMVPE.mel import MelSpectrogram

N_MELS, N_CLASS = 128, 360

class RMVPE:
    def __init__(self, model_path, is_half, device=None, providers=None, onnx=False, hpa=False, compile_model=False, compile_mode=None, enable_chunk = False, chunk_size = 8000, return_tensor = False, f0_min = 50, f0_max = 1100):
        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.RMVPE.e2e import E2E

            model = E2E(4, 1, (2, 2), 5, 4, 1, 16, hpa=hpa)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()

            if is_half: model = model.half()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        self.is_half = is_half
        self.chunk_size = chunk_size
        self.dtype = torch.float16 if is_half else torch.float32

        self.mel_extractor = MelSpectrogram(N_MELS, 16000, 1024, 160, 1024, 30, 8000).to(device)
        self.cents_mapping = np.pad(20 * np.arange(N_CLASS) + 1997.3794084376191, (4, 4))
        if return_tensor: self.cents_mapping = torch.as_tensor(self.cents_mapping, dtype=self.dtype, device=device)

        self.infer = self._infer_onnx if onnx else self._infer_torch
        self.mel2hidden = self._mel2hidden_chunk if enable_chunk else self._mel2hidden
        self.offsets = torch.arange(-4, 5, device=device) if return_tensor else np.arange(-4, 5)
        self.to_local_average_cents = self._to_local_average_cents_tensor if return_tensor else self._to_local_average_cents_array

    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0

        return f0

    def infer_from_audio(self, audio, thred=0.03):
        audio = torch.from_numpy(audio).float().to(self.device) if not torch.is_tensor(audio) else audio
        hidden = self.mel2hidden(self.mel_extractor(audio.unsqueeze(0), center=True)).squeeze(0)

        return self.decode(hidden, thred=thred)
    
    def infer_from_audio_with_pitch(self, audio, thred=0.03):
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < self.f0_min) | (f0 > self.f0_max)] = 0  

        return f0

    def _mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            return self.infer(F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect"))[:, :n_frames]

    def _mel2hidden_chunk(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = F.pad(mel, (0, 32 * ((n_frames - 1) // 32 + 1) - n_frames), mode="reflect")
            hidden = torch.cat([self.infer(mel[..., start:min(start + self.chunk_size, mel.shape[-1])]) for start in range(0, mel.shape[-1], self.chunk_size)], dim=1)

            return hidden[:, :n_frames]

    def _infer_torch(self, mel):
        if self.is_half: mel = mel.half()
        return self.model(mel)
    
    def _infer_onnx(self, mel):
        mel = mel.cpu().numpy().astype(np.float32)

        return torch.as_tensor(
            self.model.run(
                [self.model.get_outputs()[0].name], 
                {self.model.get_inputs()[0].name: mel}
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
        salience = F.pad(salience, (4, 4))
        center += 4
        
        idx = center[:, None] + self.offsets[None, :]
        local_salience = salience[torch.arange(salience.shape[0], device=salience.device)[:, None], idx]
        devided = (local_salience * self.cents_mapping[idx]).sum(dim=1) / local_salience.sum(dim=1)
        devided = torch.where(salience.max(dim=1).values <= thred, torch.zeros_like(devided), devided)

        return devided