import os
import sys
import torch

import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.algorithm.viterbi import viterbi
from main.library.predictors.PENN.core import bins_to_cents, cents_to_frequency
from main.library.predictors.PENN.core import PITCH_BINS, CENTS_PER_BIN, frequency_to_bins, entropy, interpolate

SAMPLE_RATE, WINDOW_SIZE = 16000, 1024

class Viterbi:
    def __init__(
        self, 
        pitch_bins=1440, 
        hop_length=80, 
        sample_rate=16000, 
        local_pitch_window_size=19, 
        max_octaves_per_second=32, 
        cents_per_bin=5
    ):
        self.pitch_bins = pitch_bins
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.window_size = local_pitch_window_size
        self.max_octaves_per_second = max_octaves_per_second
        self.cents_per_bin = cents_per_bin

    def __call__(self, logits):
        if not hasattr(self, 'transition'):
            idx = torch.arange(self.pitch_bins, device=logits.device, dtype=logits.dtype)
            transition = (-0.5 * ((idx[:, None] - idx[None, :]).abs() / ((self.max_octaves_per_second * 1200 * (self.hop_length / self.sample_rate)) / self.cents_per_bin)) ** 2).exp()
            self.transition = transition / transition.sum(dim=1, keepdim=True)

        distributions = F.softmax(logits, dim=1)
        bins = viterbi(distributions, self.transition)

        pitch = self.local_expected_value_from_bins(bins.T, logits)
        return pitch
    
    def local_expected_value_from_bins(self, bins, logits):
        batch, pitch_bins, frames = logits.shape
        logits_flat = logits.reshape(-1, pitch_bins)
        bins_flat = bins.reshape(-1)

        half_win = self.window_size // 2
        steps = torch.arange(-half_win, half_win + 1, device=bins.device)
        indices = (bins_flat[:, None] + steps[None, :]).clamp(0, pitch_bins - 1)

        return cents_to_frequency((F.softmax(logits_flat.gather(1, indices), dim=1) * bins_to_cents(indices)).sum(dim=1, keepdim=True)).reshape(batch, frames)

class PENN:
    def __init__(
        self, 
        model_path, 
        hop_length = 80, 
        batch_size = None, 
        f0_min = 31, 
        f0_max = 1984, 
        interp_unvoiced_at = None, 
        device = None, 
        providers = None, 
        onnx = False,
        compile_model = False,
        compile_mode = None
    ):
        self.device = device
        self.hop_length = hop_length
        self.batch_size = batch_size or 384
        self.f0_min_bin = frequency_to_bins(torch.tensor(f0_min))
        self.f0_max_bin = frequency_to_bins(torch.tensor(f0_max), torch.ceil)
        self.interp_unvoiced_at = interp_unvoiced_at
        self.decoder = Viterbi(
            PITCH_BINS, 
            hop_length, 
            SAMPLE_RATE, 
            19, 
            32, 
            CENTS_PER_BIN
        )

        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.PENN.fcn import FCN

            model = FCN(256, PITCH_BINS, (2, 2))
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True)['model'])
            model.to(device).eval()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        self.infer = self._infer_onnx if onnx else self._infer_torch
    
    def preprocess(self, audio):
        audio = F.pad(audio.to(self.device), (WINDOW_SIZE // 2, WINDOW_SIZE // 2))

        frames = audio.unfold(-1, WINDOW_SIZE, self.hop_length).squeeze(0)
        frames = frames.unsqueeze(1).contiguous()

        return frames

    def postprocess(self, logits):
        logits[:, :self.f0_min_bin] = -float('inf')
        logits[:, self.f0_max_bin:] = -float('inf')

        pitch = self.decoder(logits)
        periodicity = entropy(logits)

        return pitch.T, periodicity.T
    
    def compute_f0(self, audio):
        with torch.inference_mode():
            frames = self.preprocess(audio)
            pitch, periodicity = self.postprocess(torch.cat([self.infer(frames[i:i + self.batch_size]).detach() for i in range(0, frames.shape[0], self.batch_size)], dim=0))

            if self.interp_unvoiced_at is not None:
                pitch = interpolate(pitch, periodicity, self.interp_unvoiced_at)
                return pitch

            return pitch, periodicity
    
    def _infer_onnx(self, frames):
        return torch.tensor(
            self.model.run(
                [self.model.get_outputs()[0].name], 
                {self.model.get_inputs()[0].name: frames.cpu().numpy()}
            )[0],
            device=self.device
        )

    def _infer_torch(self, frames):
        return self.model(frames)