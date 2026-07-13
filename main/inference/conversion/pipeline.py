import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

from scipy import signal

sys.path.append(os.getcwd())

from main.library.audio.features import change_rms
from main.app.variables import translations, logger
from main.library.utils import extract_features, clear_gpu_cache

# Initialize a 5th-order Butterworth high-pass filter to remove sub-bass noise below 48Hz
bh, ah = signal.butter(
    N=5, 
    Wn=48, 
    btype="high", 
    fs=16000
)

class Pipeline:
    """
    Handles the structured step-by-step processing for voice conversion.
    Manages audio padding, chunking/slicing for long audio signals to optimize VRAM,
    and coordinates the feature extraction and generator inference.
    """

    def __init__(
        self, 
        tgt_sr, 
        config, 
        net_g,
        hubert_model,
        f0_generator, 
        version,
        sid,
        dtype
    ):
        """
        Initializes pipeline hyperparameters, padding configurations, and core models.

        Args:
            tgt_sr (int): Target sampling rate for the output audio (e.g., 40000, 48000).
            config (object): Configuration object containing hardware devices and padding thresholds.
            net_g (torch.nn.Module): The generator network used for synthesizing the converted audio.
            hubert_model (torch.nn.Module): Semantic speech feature extraction model (e.g., HuBERT/ContentVec).
            f0_generator (object): Pitch extraction utility (e.g., Harvest, Crepe, PM). Can be None for f0-less models.
            version (str): Model framework version flag ('v1' or 'v2').
            sid (torch.Tensor): Speaker ID target index for multi-speaker synthesis.
            dtype (torch.dtype): PyTorch tensor precision type (e.g., torch.float32, torch.float16).
        """
        self.window = 160
        self.sample_rate = 16000
        # Padding configurations for boundary artifact prevention
        self.x_pad = config.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad = self.sample_rate * self.x_pad
        self.t_max = self.sample_rate * config.x_max
        self.t_query = self.sample_rate * config.x_query
        self.t_center = self.sample_rate * config.x_center
        self.t_pad2 = self.t_pad * 2
        # Model and runtime environment settings
        self.sid = sid
        self.dtype = dtype
        self.net_g = net_g
        self.tgt_sr = tgt_sr
        self.version = version
        self.device = config.device
        self.f0_generator = f0_generator
        self.hubert_model = hubert_model
        self.pitch_guidance = f0_generator is not None

    def voice_conversion(
        self, 
        audio, 
        pitch, 
        pitchf, 
        index, 
        big_tsr, 
        index_rate, 
        protect, 
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5
    ):
        """
        Executes raw model-level inference for a single chunk of audio.

        Args:
            audio (np.ndarray): Input audio slice array.
            pitch (torch.Tensor): Discretized pitch values.
            pitchf (torch.Tensor): Continuous pitch float values.
            index (object): The Index Wrapper class handles the retrieval process.
            big_tsr (torch.Tensor): Saved base feature vectors corresponding to FAISS index.
            index_rate (float): Blending ratio for index retrieval features.
            protect (float): Consonant and breath sound protection scale factor.
            embedders_mix (bool): Flag to mix multiple hidden layers of the embedder.
            embedders_mix_layers (int): Number of layers to mix.
            embedders_mix_ratio (float): Ratio weighting for layer mixing.

        Returns:
            np.ndarray: Converted raw float32 audio waveform chunk.
        """

        # Convert raw numpy audio chunk into a PyTorch tensor on target device
        feats = torch.from_numpy(audio).to(self.device).to(self.dtype)
        feats = feats.mean(-1) if feats.dim() == 2 else feats # Convert stereo to mono if needed
        assert feats.dim() == 1, feats.dim()

        with torch.inference_mode():
            # Step 1: Extract semantic speech embedding features using the HuBERT/ContentVec model
            feats = extract_features(
                self.hubert_model, 
                feats.view(1, -1), 
                self.version, 
                mix=embedders_mix, 
                mix_layers=embedders_mix_layers, 
                mix_ratio=embedders_mix_ratio
            )

            # Save a clean copy of unblended features if the consonant protection mechanism is triggered
            feats0 = feats.clone() if protect < 0.5 and self.pitch_guidance else None

            # Step 2: Apply FAISS index feature mapping if index file is provided and index_rate > 0
            if index is not None and big_tsr is not None and index_rate != 0:
                # Perform K-Nearest Neighbors search (k=8) on the feature tensor
                score, ix = index.search(feats[0], k=8)
                weight = (1 / score).square() # Inverse distance squared weighting
                # Compute weighted average of retrieved index vectors
                query = (big_tsr[ix] * (weight / weight.sum(dim=1, keepdim=True)).unsqueeze(2)).sum(dim=1)
                # Blend the retrieved query features with raw extracted features based on index_rate
                feats = query.unsqueeze(0) * index_rate + (1.0 - index_rate) * feats

            # Step 3: Up-sample the features by a factor of 2 to match the frame size expectations of the generator
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
            p_len = min(audio.shape[0] // self.window, feats.shape[1])

            # Align pitch trajectory dimensions with feature sequence length
            if self.pitch_guidance: pitch, pitchf = pitch[:, :p_len], pitchf[:, :p_len]

            # Step 4: Execute consonant and breath protection blending
            if feats0 is not None:
                pitchff = pitchf.clone()
                pitchff[pitchf > 0] = 1 # Voiced frames take full conversion features
                pitchff[pitchf < 1] = protect # Unvoiced frames blend original features based on 'protect'
                pitchff = pitchff.unsqueeze(-1)

                # Up-sample the copy of original features to match current sequence length
                feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
                # Blend the features linearly using the protection mask
                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            # Step 5: Synthesize final output audio waveform using the generator model
            p_len = torch.tensor([p_len], device=self.device).long()
            if self.pitch_guidance: pitchf = pitchf.to(self.dtype)
            feats = feats.to(self.dtype) 

            audio1 = self.net_g.infer(
                feats, 
                p_len, 
                pitch, 
                pitchf,
                self.sid
            )[0, 0].cpu().float().numpy()

        # Explicitly delete temporary tensors and clean memory cache
        del feats, feats0, p_len
        clear_gpu_cache()
        return audio1
    
    def pipeline(
        self, 
        audio, 
        f0_up_key, 
        f0_method, 
        index,
        big_tsr, 
        index_rate, 
        filter_radius, 
        rms_mix_rate, 
        protect, 
        f0_autotune, 
        f0_autotune_strength, 
        f0_file=None, 
        pbar=None, 
        proposal_pitch=False, 
        proposal_pitch_threshold=255.0, 
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
    ):
        """
        The orchestrator method. Filters, segments long audios at silence points, 
        extracts pitch trajectories, loops over chunks, and concatenates final outputs.

        Args:
            audio (np.ndarray): Target 1D input audio array sampled at 16kHz.
            f0_up_key (int): Pitch shift key value in semitones (positive for higher pitch, negative for lower).
            f0_method (str): Pitch extraction algorithm method (e.g., 'harvest', 'crepe', 'rmvpe').
            index (object): The Index Wrapper class handles the retrieval process.
            big_tsr (torch.Tensor): Saved base feature vectors corresponding to FAISS index.
            index_rate (float): Blending ratio for index retrieval features (0.0 to 1.0).
            filter_radius (int): Median filtering radius applied to smooth out the extracted F0 curve.
            rms_mix_rate (float): Volume envelope adjustment rate (1.0 keeps target volume, lower values blend source volume).
            protect (float): Consonant and breath sound protection factor (0.0 to 0.5).
            f0_autotune (bool): Enable snapping the F0 pitch sequence to the nearest musical notes.
            f0_autotune_strength (float): Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
            f0_file (str, optional): Path to a raw text file containing external comma-separated pitch values. Defaults to None.
            pbar (object): Tqdm tracker object for state tracking.
            proposal_pitch (bool): Enable automatic pitch key shifting calculation based on median F0 alignment.
            proposal_pitch_threshold (float): The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.
            embedders_mix (bool): Flag to mix multiple hidden layers of the embedder. Defaults to False.
            embedders_mix_layers (int): Number of layers to mix. Defaults to 9.
            embedders_mix_ratio (float): Ratio weighting for layer mixing. Defaults to 0.5.

        Returns:
            np.ndarray: Final concatenated, normalized, and converted full length audio waveform matrix.
        """

        s = 0
        t, inp_f0 = None, None
        opt_ts, audio_opt = [], []
        # Apply the high-pass filter to eliminate DC offsets and low-frequency rumble
        audio = signal.filtfilt(bh, ah, audio)
        # Apply a minor boundary padding to secure stable audio edge processing
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        pbar.update(1)
        # Step 1: Detect optimal splitting locations for long audios to prevent VRAM spikes
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            # Compute short-term moving energy window across the waveform
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            # Find the absolute minimum volume index within query bounds to act as a split joint
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0]
                )

        # Re-pad audio fully using pipeline configuration dimensions
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        pbar.update(1)

        # Step 2: Read manual external F0 curve inputs if provided via text format
        if f0_file and os.path.exists(f0_file) and f0_file.endswith(".txt"):
            try:
                with open(f0_file, "r") as f:
                    raw_lines = f.read()

                    if len(raw_lines) > 0:
                        inp_f0 = []

                        for line in raw_lines.strip("\n").split("\n"):
                            inp_f0.append([float(i) for i in line.split(",")])

                        inp_f0 = np.array(inp_f0, dtype=np.float32)
            except:
                logger.error(translations["error_readfile"])
                inp_f0 = None

        pbar.update(1)

        # Step 3: Extract fundamental pitch (F0) tracking arrays
        if self.pitch_guidance:
            pitch, pitchf = self.f0_generator.calculator(
                self.x_pad, 
                f0_method, 
                audio_pad, 
                f0_up_key, 
                p_len, 
                filter_radius, 
                f0_autotune, 
                f0_autotune_strength, 
                manual_f0=inp_f0, 
                proposal_pitch=proposal_pitch, 
                proposal_pitch_threshold=proposal_pitch_threshold
            )

            if self.device == "mps": pitchf = pitchf.astype(np.float32)
            # Move pitch arrays into PyTorch Tensors on target hardware
            pitch, pitchf = (
                torch.tensor(pitch[:p_len], device=self.device).unsqueeze(0).long(), 
                torch.tensor(pitchf[:p_len], device=self.device).unsqueeze(0).float()
            )

        pbar.update(1)
        pbar.total = pbar.total + len(opt_ts)
        pbar.refresh()

        # Step 4: Sequentially process each audio slice identified in step 1
        for t in opt_ts:
            t = t // self.window * self.window
            start = s // self.window
            end = (t + self.t_pad2) // self.window

            # Convert chunk and crop padding fields out from the output block
            audio_opt.append(
                self.voice_conversion(
                    audio_pad[s : t + self.t_pad2 + self.window], 
                    pitch[:, start:end] if self.pitch_guidance else None, 
                    pitchf[:, start:end] if self.pitch_guidance else None, 
                    index, 
                    big_tsr, 
                    index_rate, 
                    protect, 
                    embedders_mix=embedders_mix, 
                    embedders_mix_layers=embedders_mix_layers, 
                    embedders_mix_ratio=embedders_mix_ratio
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )

            s = t
            pbar.update(1)
        
        # Step 5: Convert the remaining trailing segment of the audio track
        start_opt = (t // self.window) if t is not None else 0
        audio_opt.append(
            self.voice_conversion(
                audio_pad[t:], 
                pitch[:, start_opt:] if self.pitch_guidance else None, 
                pitchf[:, start_opt:] if self.pitch_guidance else None, 
                index, 
                big_tsr, 
                index_rate, 
                protect, 
                embedders_mix=embedders_mix, 
                embedders_mix_layers=embedders_mix_layers, 
                embedders_mix_ratio=embedders_mix_ratio
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        pbar.update(1)
        # Concatenate all generated slices back into a continuous signal matrix
        audio_opt = np.concatenate(audio_opt)

        # Step 6: Blend source volume amplitude envelope back if requested (rms_mix_rate != 1)
        if rms_mix_rate != 1:
            audio_opt = change_rms(
                audio, 
                self.sample_rate, 
                audio_opt, 
                self.tgt_sr, 
                rms_mix_rate,
                device=self.device
            ).cpu().numpy()

        # Final peak normalization control block (0.99 threshold boundary scale)
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1: audio_opt /= audio_max
        if self.pitch_guidance: del pitch, pitchf

        clear_gpu_cache()
        return audio_opt