import os
import sys
import torch

from contextlib import nullcontext

sys.path.append(os.getcwd())

from main.app.variables import config
from main.library.speaker_diarization.whisper import Whisper, ModelDimensions, log_mel_spectrogram, pad_or_trim

class WhisperModel(torch.nn.Module):
    def __init__(
        self, 
        model_path
    ):
        super().__init__()
        checkpoint = torch.load(model_path, map_location="cpu")
        dims = ModelDimensions(**checkpoint["dims"])
        self._final_proj = torch.nn.Linear(dims.n_text_state, 768)
        self.final_proj = torch.nn.Linear(768, 256)
        self.model = Whisper(dims)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del self.model.decoder
        self.rank = 0

    def forward(self, audio):
        ppgln = audio.shape[1] // 320

        mel = log_mel_spectrogram(
            pad_or_trim(audio[0])
        ).to(audio.device)

        autocast_enabled = config.is_half and audio.device.type == "cuda"
        autocast_dtype = (
            torch.float32 
            if not autocast_enabled else 
            torch.float16
        )

        autocasts = torch.amp.autocast(
            audio.device.type, 
            enabled=autocast_enabled, 
            dtype=autocast_dtype
        ) if not audio.device.type.startswith("ocl") else nullcontext()

        with torch.no_grad():
            with autocasts:
                ppg_raw = self.model.encoder(mel.unsqueeze(0))
                ppg_projected = self._final_proj(ppg_raw)

                ppg = ppg_projected.data.float()
                ppg = ppg[:, :ppgln, :]

        return [ppg]
    
    def extract_features(self, source, padding_mask = None, output_layer = None):
        if self.rank == 0:
            self.rank += 1

            self.model.to(padding_mask.device).eval()
            self.model.encoder.to(padding_mask.device).eval()

            if config.is_half: 
                self.model.half()
                self.model.encoder.half()

        return self.forward(source)