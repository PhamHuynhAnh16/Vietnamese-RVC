import os
import scipy
import torch
import librosa
import functools

import numpy as np

CENTS_PER_BIN, MAX_FMAX, PITCH_BINS, SAMPLE_RATE, WINDOW_SIZE = 20, 2006, 360, 16000, 1024  


class Crepe(torch.nn.Module):
    def __init__(self, model='full'):
        super().__init__()
        if model == 'full':
            in_channels = [1, 1024, 128, 128, 128, 256]
            out_channels = [1024, 128, 128, 128, 256, 512]
            self.in_features = 2048
        elif model == 'tiny':
            in_channels = [1, 128, 16, 16, 16, 32]
            out_channels = [128, 16, 16, 16, 32, 64]
            self.in_features = 256
        kernel_sizes = [(512, 1)] + 5 * [(64, 1)]
        strides = [(4, 1)] + 5 * [(1, 1)]
        batch_norm_fn = functools.partial(torch.nn.BatchNorm2d, eps=0.0010000000474974513, momentum=0.0)
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0], kernel_size=kernel_sizes[0], stride=strides[0])
        self.conv1_BN = batch_norm_fn(num_features=out_channels[0])
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1], kernel_size=kernel_sizes[1], stride=strides[1])
        self.conv2_BN = batch_norm_fn(num_features=out_channels[1])
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2], kernel_size=kernel_sizes[2], stride=strides[2])
        self.conv3_BN = batch_norm_fn(num_features=out_channels[2])
        self.conv4 = torch.nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3], kernel_size=kernel_sizes[3], stride=strides[3])
        self.conv4_BN = batch_norm_fn(num_features=out_channels[3])
        self.conv5 = torch.nn.Conv2d(in_channels=in_channels[4], out_channels=out_channels[4], kernel_size=kernel_sizes[4], stride=strides[4])
        self.conv5_BN = batch_norm_fn(num_features=out_channels[4])
        self.conv6 = torch.nn.Conv2d(in_channels=in_channels[5], out_channels=out_channels[5], kernel_size=kernel_sizes[5], stride=strides[5])
        self.conv6_BN = batch_norm_fn(num_features=out_channels[5])
        self.classifier = torch.nn.Linear(in_features=self.in_features, out_features=PITCH_BINS)

    def forward(self, x, embed=False):
        x = self.embed(x)
        if embed: return x
        return torch.sigmoid(self.classifier(self.layer(x, self.conv6, self.conv6_BN).permute(0, 2, 1, 3).reshape(-1, self.in_features)))

    def embed(self, x):
        x = x[:, None, :, None]
        return self.layer(self.layer(self.layer(self.layer(self.layer(x, self.conv1, self.conv1_BN, (0, 0, 254, 254)), self.conv2, self.conv2_BN), self.conv3, self.conv3_BN), self.conv4, self.conv4_BN), self.conv5, self.conv5_BN)

    def layer(self, x, conv, batch_norm, padding=(0, 0, 31, 32)):
        return torch.nn.functional.max_pool2d(batch_norm(torch.nn.functional.relu(conv(torch.nn.functional.pad(x, padding)))), (2, 1), (2, 1))

def viterbi(logits):
    if not hasattr(viterbi, 'transition'):
        xx, yy = np.meshgrid(range(360), range(360))
        transition = np.maximum(12 - abs(xx - yy), 0)
        viterbi.transition = transition / transition.sum(axis=1, keepdims=True)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(logits, dim=1)
    bins = torch.tensor(np.array([librosa.sequence.viterbi(sequence, viterbi.transition).astype(np.int64) for sequence in probs.cpu().numpy()]), device=probs.device)
    return bins, bins_to_frequency(bins)

def predict(audio, sample_rate, hop_length=None, fmin=50, fmax=MAX_FMAX, model='full', decoder=viterbi, return_periodicity=False, batch_size=None, device='cpu', pad=True):
    results = []
    with torch.no_grad():
        for frames in preprocess(audio, sample_rate, hop_length, batch_size, device, pad):
            result = postprocess(infer(frames, model, device, embed=False).reshape(audio.size(0), -1, PITCH_BINS).transpose(1, 2), fmin, fmax, decoder, return_periodicity)
            results.append((result[0].to(audio.device), result[1].to(audio.device)) if isinstance(result, tuple) else result.to(audio.device))
    if return_periodicity:
        pitch, periodicity = zip(*results)
        return torch.cat(pitch, 1), torch.cat(periodicity, 1)
    return torch.cat(results, 1)

def bins_to_frequency(bins):
    cents = CENTS_PER_BIN * bins + 1997.3794084376191
    return 10 * 2 ** ((cents + cents.new_tensor(scipy.stats.triang.rvs(c=0.5, loc=-CENTS_PER_BIN, scale=2 * CENTS_PER_BIN, size=cents.size()))) / 1200)

def frequency_to_bins(frequency, quantize_fn=torch.floor):
    return quantize_fn(((1200 * torch.log2(frequency / 10)) - 1997.3794084376191) / CENTS_PER_BIN).int()

def infer(frames, model='full', device='cpu', embed=False):
    if not hasattr(infer, 'model') or not hasattr(infer, 'capacity') or (hasattr(infer, 'capacity') and infer.capacity != model): load_model(device, model)
    infer.model = infer.model.to(device)
    return infer.model(frames, embed=embed)

def load_model(device, capacity='full'):
    infer.capacity = capacity
    infer.model = Crepe(capacity)
    infer.model.load_state_dict(torch.load(os.path.join("assets", "models", "predictors", f"crepe_{capacity}.pth"), map_location=device))
    infer.model = infer.model.to(torch.device(device))
    infer.model.eval()

def postprocess(probabilities, fmin=0, fmax=MAX_FMAX, decoder=viterbi, return_periodicity=False):
    probabilities = probabilities.detach()
    probabilities[:, :frequency_to_bins(torch.tensor(fmin))] = -float('inf')
    probabilities[:, frequency_to_bins(torch.tensor(fmax), torch.ceil):] = -float('inf')
    bins, pitch = decoder(probabilities)
    if not return_periodicity: return pitch
    return pitch, periodicity(probabilities, bins)

def preprocess(audio, sample_rate, hop_length=None, batch_size=None, device='cpu', pad=True):
    hop_length = sample_rate // 100 if hop_length is None else hop_length
    if sample_rate != SAMPLE_RATE:
        audio = torch.tensor(librosa.resample(audio.detach().cpu().numpy().squeeze(0), orig_sr=sample_rate, target_sr=SAMPLE_RATE), device=audio.device).unsqueeze(0)
        hop_length = int(hop_length * SAMPLE_RATE / sample_rate)
    if pad:
        total_frames = 1 + int(audio.size(1) // hop_length)
        audio = torch.nn.functional.pad(audio, (WINDOW_SIZE // 2, WINDOW_SIZE // 2))
    else: total_frames = 1 + int((audio.size(1) - WINDOW_SIZE) // hop_length)
    batch_size = total_frames if batch_size is None else batch_size
    for i in range(0, total_frames, batch_size):
        frames = torch.nn.functional.unfold(audio[:, None, None, max(0, i * hop_length):min(audio.size(1), (i + batch_size - 1) * hop_length + WINDOW_SIZE)], kernel_size=(1, WINDOW_SIZE), stride=(1, hop_length))
        frames = frames.transpose(1, 2).reshape(-1, WINDOW_SIZE).to(device)
        frames -= frames.mean(dim=1, keepdim=True)
        frames /= torch.max(torch.tensor(1e-10, device=frames.device), frames.std(dim=1, keepdim=True))
        yield frames

def periodicity(probabilities, bins):
    probs_stacked = probabilities.transpose(1, 2).reshape(-1, PITCH_BINS)
    periodicity = probs_stacked.gather(1, bins.reshape(-1, 1).to(torch.int64))
    return periodicity.reshape(probabilities.size(0), probabilities.size(2))