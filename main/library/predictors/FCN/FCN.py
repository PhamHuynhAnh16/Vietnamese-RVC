import os
import sys
import math
import torch
import librosa
import torchaudio

import numpy as np

sys.path.append(os.getcwd())

from main.library.predictors.FCN.model import MODEL
from main.library.predictors.FCN.convert import frequency_to_bins, seconds_to_samples, bins_to_frequency

PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 1440, 16000, 1024

class FCN:
    def __init__(self, model_path, hop_length=160, batch_size=None, f0_min=50, f0_max=1100, device=None, sample_rate=16000, providers=None, onnx=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hopsize = hop_length / SAMPLE_RATE
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.onnx = onnx
        self.f0_min = f0_min
        self.f0_max = f0_max

        if self.onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            self.model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            model = MODEL()
            ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt['model'])
            model.eval()
            self.model = model.to(device)
    
    def entropy(self, logits):
        distribution = torch.nn.functional.softmax(logits, dim=1)
        return (1 + 1 / math.log(PITCH_BINS) * (distribution * torch.log(distribution + 1e-7)).sum(dim=1))
    
    def expected_frames(self, samples, center):
        hopsize_resampled = seconds_to_samples(self.hopsize, self.sample_rate)

        if center == 'half-window':
            window_size_resampled = WINDOW_SIZE / SAMPLE_RATE * self.sample_rate
            samples = samples - (window_size_resampled - hopsize_resampled)
        elif center == 'half-hop':
            samples = samples
        elif center == 'zero':
            samples = samples + hopsize_resampled
        else: raise ValueError

        return max(1, int(samples / hopsize_resampled))

    def resample(self, audio, target_rate=SAMPLE_RATE):
        if self.sample_rate == target_rate: return audio

        resampler = torchaudio.transforms.Resample(self.sample_rate, target_rate)
        resampler = resampler.to(audio.device)

        return resampler(audio)
    
    def preprocess(self, audio, center='half-window'):
        total_frames = self.expected_frames(audio.shape[-1], center)
        if self.sample_rate != SAMPLE_RATE: audio = self.resample(audio)

        hopsize = seconds_to_samples(self.hopsize, SAMPLE_RATE)
        if center in ['half-hop', 'zero']:
            if center == 'half-hop': padding = int((WINDOW_SIZE - hopsize) / 2)
            else: padding = int(WINDOW_SIZE / 2)

            audio = torch.nn.functional.pad(audio, (padding, padding), mode='reflect')

        if isinstance(hopsize, int) or hopsize.is_integer():
            hopsize = int(round(hopsize))
            start_idxs = None
        else:
            start_idxs = torch.round(torch.tensor([hopsize * i for i in range(total_frames + 1)])).int()

        batch_size = total_frames if self.batch_size is None else self.batch_size

        for i in range(0, total_frames, batch_size):
            batch = min(total_frames - i, batch_size)

            if start_idxs is None:
                start = i * hopsize
                end = start + int((batch - 1) * hopsize) + WINDOW_SIZE
                end = min(end, audio.shape[-1])
                batch_audio = audio[:, start:end]

                if end - start < WINDOW_SIZE:
                    padding = WINDOW_SIZE - (end - start)
                    remainder = (end - start) % hopsize

                    if remainder: padding += end - start - hopsize
                    batch_audio = torch.nn.functional.pad(batch_audio, (0, padding))

                frames = torch.nn.functional.unfold(batch_audio[:, None, None], kernel_size=(1, WINDOW_SIZE), stride=(1, hopsize)).permute(2, 0, 1)
            else:
                frames = torch.zeros(batch, 1, WINDOW_SIZE)

                for j in range(batch):
                    start = start_idxs[i + j]
                    end = min(start + WINDOW_SIZE, audio.shape[-1])
                    frames[j, :, : end - start] = audio[:, start:end]

            yield frames

    def viterbi(self, logits):
        if not hasattr(self, 'transition'):
            xx, yy = np.meshgrid(range(PITCH_BINS), range(PITCH_BINS))
            transition = np.maximum(12 - abs(xx - yy), 0)
            self.transition = transition / transition.sum(axis=1, keepdims=True)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)

        bins = torch.tensor(np.array([librosa.sequence.viterbi(sequence, self.transition).astype(np.int64) for sequence in probs.cpu().numpy()]), device=probs.device)
        return bins_to_frequency(bins)

    def postprocess(self, logits):
        with torch.inference_mode():
            minidx = frequency_to_bins(torch.tensor(self.f0_min))
            maxidx = frequency_to_bins(torch.tensor(self.f0_max), torch.ceil)

            logits = logits.detach()
            logits[:, :minidx] = -float('inf')
            logits[:, maxidx:] = -float('inf')

            pitch = self.viterbi(logits)
            periodicity = self.entropy(logits)

            return pitch.T, periodicity.T

    def compute_f0(self, audio, center = 'half-window'):
        if self.batch_size is not None: logits = []

        for frames in self.preprocess(audio, center):
            if self.onnx:
                inferred = torch.tensor(
                    self.model.run(
                        [self.model.get_outputs()[0].name], 
                        {
                            self.model.get_inputs()[0].name: frames.cpu().numpy()
                        }
                    )[0]
                ).detach()
            else:
                with torch.no_grad():
                    inferred = self.model(frames.to(self.device)).detach()

            logits.append(inferred)

        pitch, periodicity = self.postprocess(torch.cat(logits, 0).to(self.device))
        return pitch, periodicity