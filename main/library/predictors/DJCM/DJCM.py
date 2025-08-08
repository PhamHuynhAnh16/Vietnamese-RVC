import os
import sys
import torch

import numpy as np

from scipy.signal import medfilt

sys.path.append(os.getcwd())

from main.library.predictors.DJCM.model import DJCMM
from main.library.predictors.DJCM.utils import N_CLASS, SAMPLE_RATE, HOP_SIZE

class DJCM:
    def __init__(self, model_path, device = "cpu", is_half = False, onnx = False, providers = ["CPUExecutionProvider"], batch_size = 1, segment_len = 5.12):
        super(DJCM, self).__init__()
        self.onnx = onnx

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = DJCMM(1, 1, 1)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model = model.to(device).eval()
            self.model = model.half() if is_half else model.float()

        self.batch_size = batch_size
        self.seg_len = int(segment_len * SAMPLE_RATE)
        self.seg_frames = int(self.seg_len // HOP_SIZE)
        self.is_half = is_half
        self.device = device

        cents_mapping = 20 * np.arange(N_CLASS) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))

    def audio2hidden(self, audio):
        if self.onnx:
            hidden = torch.as_tensor(
                self.model.run([self.model.get_outputs()[0].name], {self.model.get_inputs()[0].name: audio.cpu().numpy().astype(np.float32)})[0], device=self.device
            )
        else:
            hidden = self.model(
                audio.half() if self.is_half else audio.float()
            )

        return hidden

    def infer_from_audio(self, audio, thred=0.03):
        with torch.inference_mode():
            padded_audio = self.pad_audio(audio)
            hidden = self.inference(padded_audio)[:(audio.shape[-1] // HOP_SIZE + 1)]

            f0 = self.decode(hidden.squeeze(0).cpu().numpy(), thred)
            f0 = medfilt(f0, kernel_size=3)

            return f0
        
    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        f0 = self.infer_from_audio(audio, thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0

        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience, todo_cents_mapping = [], []
        starts = center - 4
        ends = center + 5

        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])

        todo_salience = np.array(todo_salience)
        devided = np.sum(todo_salience * np.array(todo_cents_mapping), 1) / np.sum(todo_salience, 1)
        devided[np.max(salience, axis=1) <= thred] = 0

        return devided
        
    def decode(self, hidden, thred=0.03):
        f0 = 10 * (2 ** (self.to_local_average_cents(hidden, thred=thred) / 1200))
        f0[f0 == 10] = 0
        return f0

    def pad_audio(self, audio):
        audio_len = audio.shape[-1]

        seg_nums = int(np.ceil(audio_len / self.seg_len)) + 1
        pad_len = int(seg_nums * self.seg_len - audio_len + self.seg_len // 2)

        left_pad = np.zeros(int(self.seg_len // 4), dtype=np.float32)
        right_pad = np.zeros(int(pad_len - self.seg_len // 4), dtype=np.float32)
        padded_audio = np.concatenate([left_pad, audio, right_pad], axis=-1)

        segments = [padded_audio[start: start + int(self.seg_len)] for start in range(0, len(padded_audio) - int(self.seg_len) + 1, int(self.seg_len // 2))]
        segments = np.stack(segments, axis=0)
        segments = torch.from_numpy(segments).unsqueeze(1).to(self.device)

        return segments

    def inference(self, segments):
        hidden_segments = torch.cat([
            self.audio2hidden(segments[i:i + self.batch_size]) 
            for i in range(0, len(segments), self.batch_size)
        ], dim=0)

        hidden = torch.cat([
            seg[self.seg_frames // 4: int(self.seg_frames * 0.75)]
            for seg in hidden_segments
        ], dim=0)

        return hidden