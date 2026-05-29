import os
import sys
import torch

sys.path.append(os.getcwd())

from main.library.predictors.FCPE.stft import STFT

class Wav2Mel:
    def __init__(
        self, 
        device=None, 
        dtype=torch.float32
    ):
        self.sample_rate = 16000
        self.hop_size = 160
        self.device = device
        self.dtype = dtype
        self.stft = STFT(16000, 128, 1024, 1024, 160, 0, 8000)

    def extract_mel(self, audio):
        audio = audio.to(self.dtype).to(self.device)
        mel = self.stft.get_mel(audio).transpose(1, 2)

        n_frames = int(audio.shape[1] // self.hop_size) + 1
        mel = (torch.cat((mel, mel[:, -1:, :]), 1) if n_frames > int(mel.shape[1]) else mel)

        return mel[:, :n_frames, :] if n_frames < int(mel.shape[1]) else mel

    def __call__(self, audio):
        return self.extract_mel(audio)