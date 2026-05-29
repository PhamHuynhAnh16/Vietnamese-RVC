import os
import sys
import torch
import librosa
import scipy.stats

sys.path.append(os.getcwd())

from main.library.algorithm.viterbi import viterbi

CENTS_PER_BIN, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 360, 16000, 1024

class CREPE:
    def __init__(
        self, 
        model_path, 
        model_size="full", 
        hop_length=512, 
        batch_size=None, 
        f0_min=50, 
        f0_max=1100, 
        device=None, 
        sample_rate=16000, 
        providers=None, 
        onnx=False, 
        return_periodicity=False,
        compile_model = False,
        compile_mode = None
    ):
        self.device = device
        self.hop_length = hop_length
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.return_periodicity = return_periodicity

        self.f0_min_bin = (((1200 * torch.tensor(f0_min / 10).log2()) - 1997.3794084376191) / CENTS_PER_BIN).floor().int()
        self.f0_max_bin = (((1200 * torch.tensor(f0_max / 10).log2()) - 1997.3794084376191) / CENTS_PER_BIN).ceil().int()
        self.eps = torch.tensor(1e-10, device=device)

        if onnx:
            import onnxruntime as ort

            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3
            model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        else:
            from main.library.predictors.CREPE.model import CREPEE

            model = CREPEE(model_size)
            model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
            model.to(device).eval()
            if compile_model: model = torch.compile(model, mode=compile_mode)

        self.model = model
        self.infer = self._infer_onnx if onnx else self._infer_torch

    def bins_to_frequency(self, bins):
        if str(bins.device).startswith(("ocl", "privateuseone")): bins = bins.to(torch.float32)

        cents = CENTS_PER_BIN * bins + 1997.3794084376191
        cents = (
            cents + cents.new_tensor(
                scipy.stats.triang.rvs(
                    c=0.5, 
                    loc=-CENTS_PER_BIN, 
                    scale=2 * CENTS_PER_BIN, 
                    size=cents.size()
                )
            )
        ) / 1200

        return 10 * 2 ** cents

    def viterbi(self, logits):
        if not hasattr(self, 'transition'):
            idx = torch.arange(360, device=logits.device, dtype=logits.dtype)
            transition = (12 - (idx[:, None] - idx[None, :]).abs()).clamp(min=0)
            self.transition = transition / transition.sum(dim=1, keepdim=True)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(logits, dim=1)
            bins = viterbi(probs, self.transition)

        return bins, self.bins_to_frequency(bins)
    
    def preprocess(self, audio, pad=True):
        hop_length = (self.sample_rate // 100) if self.hop_length is None else self.hop_length

        if self.sample_rate != SAMPLE_RATE:
            audio = torch.tensor(
                librosa.resample(
                    audio.detach().cpu().numpy().squeeze(0), 
                    orig_sr=self.sample_rate, 
                    target_sr=SAMPLE_RATE, 
                    res_type="soxr_vhq"
                ), 
                device=audio.device
            ).unsqueeze(0)

            hop_length = int(hop_length * SAMPLE_RATE / self.sample_rate)

        if pad:
            total_frames = 1 + int(audio.size(1) // hop_length)
            audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
        else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)

        batch_size = total_frames if self.batch_size is None else self.batch_size

        for i in range(0, total_frames, batch_size):
            frames = torch.nn.functional.unfold(
                audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], 
                kernel_size=(1, WINDOW_SIZE), 
                stride=(1, hop_length)
            ).transpose(1, 2)
            
            if self.device.startswith(("ocl", "privateuseone")): frames = frames.contiguous()
            frames = frames.reshape(-1, WINDOW_SIZE).to(self.device)

            frames -= frames.mean(dim=1, keepdim=True)
            frames /= self.eps.max(frames.std(dim=1, keepdim=True))

            yield frames

    def periodicity(self, probabilities, bins):
        probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
        periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
        
        return periodicity.reshape(probabilities.size(0), probabilities.size(2))

    def postprocess(self, probabilities):
        probabilities = probabilities.detach()
        probabilities[:, :self.f0_min_bin] = -float('inf')
        probabilities[:, self.f0_max_bin:] = -float('inf')

        bins, pitch = self.viterbi(probabilities)

        if not self.return_periodicity: return pitch
        return pitch, self.periodicity(probabilities, bins)

    def compute_f0(self, audio, pad=True):
        results = []

        with torch.no_grad():
            for frames in self.preprocess(audio, pad):
                result = self.postprocess(self.infer(audio, frames))
                results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))
        
        if self.return_periodicity:
            pitch, periodicity = zip(*results)
            return torch.cat(pitch, 1), torch.cat(periodicity, 1)
        
        return torch.cat(results, 1)
    
    def _infer_torch(self, audio, frames):
        return self.model(
            frames, 
            embed=False
        ).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2)

    def _infer_onnx(self, audio, frames):
        return torch.tensor(
            self.model.run(
                [self.model.get_outputs()[0].name], 
                {
                    self.model.get_inputs()[0].name: frames.cpu().numpy()
                }
            )[0].transpose(1, 0)[None],
            device=self.device
        )