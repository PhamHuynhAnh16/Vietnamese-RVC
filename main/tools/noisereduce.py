import torch
import tempfile
import numpy as np

from tqdm.auto import tqdm
from joblib import Parallel, delayed
from torch.nn.functional import conv1d, conv2d

from main.configs.config import Config
translations = Config().translations

@torch.no_grad()
def amp_to_db(x, eps = torch.finfo(torch.float64).eps, top_db = 40):
    x_db = 20 * torch.log10(x.abs() + eps)
    return torch.max(x_db, (x_db.max(-1).values - top_db).unsqueeze(-1))

@torch.no_grad()
def temperature_sigmoid(x, x0, temp_coeff):
    return torch.sigmoid((x - x0) / temp_coeff)

@torch.no_grad()
def linspace(start, stop, num = 50, endpoint = True, **kwargs):
    return torch.linspace(start, stop, num, **kwargs) if endpoint else torch.linspace(start, stop, num + 1, **kwargs)[:-1]

def _smoothing_filter(n_grad_freq, n_grad_time):
    smoothing_filter = np.outer(np.concatenate([np.linspace(0, 1, n_grad_freq + 1, endpoint=False), np.linspace(1, 0, n_grad_freq + 2)])[1:-1], np.concatenate([np.linspace(0, 1, n_grad_time + 1, endpoint=False), np.linspace(1, 0, n_grad_time + 2)])[1:-1])
    return smoothing_filter / np.sum(smoothing_filter)

class SpectralGate:
    def __init__(self, y, sr, prop_decrease, chunk_size, padding, n_fft, win_length, hop_length, time_constant_s, freq_mask_smooth_hz, time_mask_smooth_ms, tmp_folder, use_tqdm, n_jobs):
        self.sr = sr
        self.flat = False
        y = np.array(y)

        if len(y.shape) == 1:
            self.y = np.expand_dims(y, 0)
            self.flat = True
        elif len(y.shape) > 2: raise ValueError(translations["waveform"])
        else: self.y = y

        self._dtype = y.dtype
        self.n_channels, self.n_frames = self.y.shape
        self._chunk_size = chunk_size
        self.padding = padding
        self.n_jobs = n_jobs
        self.use_tqdm = use_tqdm
        self._tmp_folder = tmp_folder
        self._n_fft = n_fft
        self._win_length = self._n_fft if win_length is None else win_length
        self._hop_length = (self._win_length // 4) if hop_length is None else hop_length
        self._time_constant_s = time_constant_s
        self._prop_decrease = prop_decrease

        if (freq_mask_smooth_hz is None) & (time_mask_smooth_ms is None): self.smooth_mask = False
        else: self._generate_mask_smoothing_filter(freq_mask_smooth_hz, time_mask_smooth_ms)

    def _generate_mask_smoothing_filter(self, freq_mask_smooth_hz, time_mask_smooth_ms):
        if freq_mask_smooth_hz is None: n_grad_freq = 1
        else:
            n_grad_freq = int(freq_mask_smooth_hz / (self.sr / (self._n_fft / 2)))
            if n_grad_freq < 1: raise ValueError(translations["freq_mask_smooth_hz"].format(hz=int((self.sr / (self._n_fft / 2)))))

        if time_mask_smooth_ms is None: n_grad_time = 1
        else:
            n_grad_time = int(time_mask_smooth_ms / ((self._hop_length / self.sr) * 1000))
            if n_grad_time < 1: raise ValueError(translations["time_mask_smooth_ms"].format(ms=int((self._hop_length / self.sr) * 1000)))

        if (n_grad_time == 1) & (n_grad_freq == 1): self.smooth_mask = False
        else:
            self.smooth_mask = True
            self._smoothing_filter = _smoothing_filter(n_grad_freq, n_grad_time)

    def _read_chunk(self, i1, i2):
        i1b = 0 if i1 < 0 else i1  
        i2b = self.n_frames if i2 > self.n_frames else i2 
        chunk = np.zeros((self.n_channels, i2 - i1))
        chunk[:, i1b - i1: i2b - i1] = self.y[:, i1b:i2b]
        return chunk

    def filter_chunk(self, start_frame, end_frame):
        i1 = start_frame - self.padding
        return self._do_filter(self._read_chunk(i1, (end_frame + self.padding)))[:, start_frame - i1: end_frame - i1]

    def _get_filtered_chunk(self, ind):
        start0 = ind * self._chunk_size
        end0 = (ind + 1) * self._chunk_size
        return self.filter_chunk(start_frame=start0, end_frame=end0)

    def _do_filter(self, chunk):
        pass

    def _iterate_chunk(self, filtered_chunk, pos, end0, start0, ich):
        filtered_chunk[:, pos: pos + end0 - start0] = self._get_filtered_chunk(ich)[:, start0:end0]
        pos += end0 - start0

    def get_traces(self, start_frame=None, end_frame=None):
        if start_frame is None: start_frame = 0
        if end_frame is None: end_frame = self.n_frames

        if self._chunk_size is not None:
            if end_frame - start_frame > self._chunk_size:
                ich1 = int(start_frame / self._chunk_size)
                ich2 = int((end_frame - 1) / self._chunk_size)

                with tempfile.NamedTemporaryFile(prefix=self._tmp_folder) as fp:
                    filtered_chunk = np.memmap(fp, dtype=self._dtype, shape=(self.n_channels, int(end_frame - start_frame)), mode="w+")
                    pos_list, start_list, end_list = [], [], []
                    pos = 0

                    for ich in range(ich1, ich2 + 1):
                        start0 = (start_frame - ich * self._chunk_size) if ich == ich1 else 0
                        end0 = end_frame - ich * self._chunk_size if ich == ich2 else self._chunk_size
                        pos_list.append(pos)
                        start_list.append(start0)
                        end_list.append(end0)
                        pos += end0 - start0

                    Parallel(n_jobs=self.n_jobs)(delayed(self._iterate_chunk)(filtered_chunk, pos, end0, start0, ich) for pos, start0, end0, ich in zip(tqdm(pos_list, disable=not (self.use_tqdm)), start_list, end_list, range(ich1, ich2 + 1)))
                    return filtered_chunk.astype(self._dtype).flatten() if self.flat else filtered_chunk.astype(self._dtype)

        filtered_chunk = self.filter_chunk(start_frame=0, end_frame=end_frame)
        return filtered_chunk.astype(self._dtype).flatten() if self.flat else filtered_chunk.astype(self._dtype)

class TG(torch.nn.Module):
    @torch.no_grad()
    def __init__(self, sr, nonstationary = False, n_std_thresh_stationary = 1.5, n_thresh_nonstationary = 1.3, temp_coeff_nonstationary = 0.1, n_movemean_nonstationary = 20, prop_decrease = 1.0, n_fft = 1024, win_length = None, hop_length = None, freq_mask_smooth_hz = 500, time_mask_smooth_ms = 50):
        super().__init__()
        self.sr = sr
        self.nonstationary = nonstationary
        assert 0.0 <= prop_decrease <= 1.0
        self.prop_decrease = prop_decrease
        self.n_fft = n_fft
        self.win_length = self.n_fft if win_length is None else win_length
        self.hop_length = self.win_length // 4 if hop_length is None else hop_length
        self.n_std_thresh_stationary = n_std_thresh_stationary
        self.temp_coeff_nonstationary = temp_coeff_nonstationary
        self.n_movemean_nonstationary = n_movemean_nonstationary
        self.n_thresh_nonstationary = n_thresh_nonstationary
        self.freq_mask_smooth_hz = freq_mask_smooth_hz
        self.time_mask_smooth_ms = time_mask_smooth_ms
        self.register_buffer("smoothing_filter", self._generate_mask_smoothing_filter())

    @torch.no_grad()
    def _generate_mask_smoothing_filter(self):
        if self.freq_mask_smooth_hz is None and self.time_mask_smooth_ms is None: return None
        n_grad_freq = (1 if self.freq_mask_smooth_hz is None else int(self.freq_mask_smooth_hz / (self.sr / (self.n_fft / 2))))
        if n_grad_freq < 1: raise ValueError(translations["freq_mask_smooth_hz"].format(hz=int((self.sr / (self._n_fft / 2)))))

        n_grad_time = (1 if self.time_mask_smooth_ms is None else int(self.time_mask_smooth_ms / ((self.hop_length / self.sr) * 1000)))
        if n_grad_time < 1: raise ValueError(translations["time_mask_smooth_ms"].format(ms=int((self._hop_length / self.sr) * 1000)))
        if n_grad_time == 1 and n_grad_freq == 1: return None

        smoothing_filter = torch.outer(torch.cat([linspace(0, 1, n_grad_freq + 1, endpoint=False), linspace(1, 0, n_grad_freq + 2)])[1:-1], torch.cat([linspace(0, 1, n_grad_time + 1, endpoint=False), linspace(1, 0, n_grad_time + 2)])[1:-1]).unsqueeze(0).unsqueeze(0)
        return smoothing_filter / smoothing_filter.sum()

    @torch.no_grad()
    def _stationary_mask(self, X_db, xn = None):
        XN_db = amp_to_db(torch.stft(xn, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True, pad_mode="constant", center=True, window=torch.hann_window(self.win_length).to(xn.device))).to(dtype=X_db.dtype) if xn is not None else X_db
        std_freq_noise, mean_freq_noise = torch.std_mean(XN_db, dim=-1)
        return torch.gt(X_db, (mean_freq_noise + std_freq_noise * self.n_std_thresh_stationary).unsqueeze(2))

    @torch.no_grad()
    def _nonstationary_mask(self, X_abs):
        X_smoothed = (conv1d(X_abs.reshape(-1, 1, X_abs.shape[-1]), torch.ones(self.n_movemean_nonstationary, dtype=X_abs.dtype, device=X_abs.device).view(1, 1, -1), padding="same").view(X_abs.shape) / self.n_movemean_nonstationary)
        return temperature_sigmoid(((X_abs - X_smoothed) / X_smoothed), self.n_thresh_nonstationary, self.temp_coeff_nonstationary)

    def forward(self, x, xn = None):
        assert x.ndim == 2
        if x.shape[-1] < self.win_length * 2: raise Exception(f"{translations['x']} {self.win_length * 2}")
        assert xn is None or xn.ndim == 1 or xn.ndim == 2
        if xn is not None and xn.shape[-1] < self.win_length * 2: raise Exception(f"{translations['xn']} {self.win_length * 2}")

        X = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True, pad_mode="constant", center=True, window=torch.hann_window(self.win_length).to(x.device))
        sig_mask = self._nonstationary_mask(X.abs()) if self.nonstationary else self._stationary_mask(amp_to_db(X), xn)

        sig_mask = self.prop_decrease * (sig_mask * 1.0 - 1.0) + 1.0
        if self.smoothing_filter is not None: sig_mask = conv2d(sig_mask.unsqueeze(1), self.smoothing_filter.to(sig_mask.dtype), padding="same")

        Y = X * sig_mask.squeeze(1)
        return torch.istft(Y, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, center=True, window=torch.hann_window(self.win_length).to(Y.device)).to(dtype=x.dtype)

class StreamedTorchGate(SpectralGate):
    def __init__(self, y,sr, stationary=False, y_noise=None, prop_decrease=1.0, time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50, thresh_n_mult_nonstationary=2, sigmoid_slope_nonstationary=10, n_std_thresh_stationary=1.5, tmp_folder=None, chunk_size=600000, padding=30000, n_fft=1024, win_length=None, hop_length=None, clip_noise_stationary=True, use_tqdm=False, n_jobs=1, device="cpu"):
        super().__init__(y=y, sr=sr, chunk_size=chunk_size, padding=padding, n_fft=n_fft, win_length=win_length, hop_length=hop_length, time_constant_s=time_constant_s, freq_mask_smooth_hz=freq_mask_smooth_hz, time_mask_smooth_ms=time_mask_smooth_ms, tmp_folder=tmp_folder, prop_decrease=prop_decrease, use_tqdm=use_tqdm, n_jobs=n_jobs)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        if y_noise is not None:
            if y_noise.shape[-1] > y.shape[-1] and clip_noise_stationary: y_noise = y_noise[: y.shape[-1]]
            y_noise = torch.from_numpy(y_noise).to(device)
            if len(y_noise.shape) == 1: y_noise = y_noise.unsqueeze(0)

        self.y_noise = y_noise
        self.tg = TG(sr=sr, nonstationary=not stationary, n_std_thresh_stationary=n_std_thresh_stationary, n_thresh_nonstationary=thresh_n_mult_nonstationary, temp_coeff_nonstationary=1 / sigmoid_slope_nonstationary, n_movemean_nonstationary=int(time_constant_s / self._hop_length * sr), prop_decrease=prop_decrease, n_fft=self._n_fft, win_length=self._win_length, hop_length=self._hop_length, freq_mask_smooth_hz=freq_mask_smooth_hz, time_mask_smooth_ms=time_mask_smooth_ms).to(device)

    def _do_filter(self, chunk):
        if type(chunk) is np.ndarray: chunk = torch.from_numpy(chunk).to(self.device)
        return self.tg(x=chunk, xn=self.y_noise).cpu().detach().numpy()

def reduce_noise(y, sr, stationary=False, y_noise=None, prop_decrease=1.0, time_constant_s=2.0, freq_mask_smooth_hz=500, time_mask_smooth_ms=50, thresh_n_mult_nonstationary=2, sigmoid_slope_nonstationary=10, tmp_folder=None, chunk_size=600000, padding=30000, n_fft=1024, win_length=None, hop_length=None, clip_noise_stationary=True, use_tqdm=False, device="cpu"):
    return StreamedTorchGate(y=y, sr=sr, stationary=stationary, y_noise=y_noise, prop_decrease=prop_decrease, time_constant_s=time_constant_s, freq_mask_smooth_hz=freq_mask_smooth_hz, time_mask_smooth_ms=time_mask_smooth_ms, thresh_n_mult_nonstationary=thresh_n_mult_nonstationary, sigmoid_slope_nonstationary=sigmoid_slope_nonstationary, tmp_folder=tmp_folder, chunk_size=chunk_size, padding=padding, n_fft=n_fft, win_length=win_length, hop_length=hop_length, clip_noise_stationary=clip_noise_stationary, use_tqdm=use_tqdm, n_jobs=1, device=device).get_traces()