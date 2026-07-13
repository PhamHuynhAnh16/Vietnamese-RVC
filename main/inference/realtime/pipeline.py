import os
import sys
import torch

import numpy as np
import torch.nn.functional as F

sys.path.append(os.getcwd())

from main.library.audio.features import change_rms
from main.app.variables import config, logger, translations, configs
from main.library.utils import load_embedders_model, extract_features, load_faiss_index, load_model

class RealtimeVoiceConverter:
    """
    Handles loading, setting up, and running inference for Retrieval-based Voice Conversion (RVC) 
    and SVC acoustic models in real-time environments.
    """

    def __init__(self, weight_root, noise_scale = 0.35):
        """
        Initializes the RealtimeVoiceConverter instance.

        Args:
            weight_root (str): Path to the model weight file (.pth or checkpoint object).
            noise_scale (float): Noise scale factor for the SynthesizerSVC model architecture.
        """

        self.model = None
        self.net_g = None
        self.tgt_sr = None
        self.use_f0 = None
        self.version = None
        self.architecture = None
        # Trigger model configuration and device deployment
        self.setup(weight_root, noise_scale)

    def setup(self, weight_root, noise_scale = 0.35):
        """
        Loads the model weights and constructs the target generator architecture (Synthesizer/SynthesizerSVC).

        Args:
            weight_root (str): Path to the model weight file.
            noise_scale (float): Scaling factor for noise standard deviation in SVC models.

        Returns:
            RealtimeVoiceConverter: The configured instance itself.
        """

        # Load raw dictionary weights or pre-instantiated checkpoints
        model = load_model(weight_root)

        if weight_root.endswith(".pth"):
            from main.library.algorithm.synthesizers import Synthesizer, SynthesizerSVC

            # Extract target sample rate from the last position of the config list
            self.tgt_sr = model["config"][-1]
            # Dynamic adjustment of the embedding layer weight dimension based on checkpoint data
            model["config"][-3] = model["weight"]["emb_g.weight"].shape[0]

            # Fetch model metadata attributes with safe default values
            self.use_f0 = model.get("f0", 1)
            self.version = model.get("version", "v1")
            self.vocoder = model.get("vocoder", "Default")
            self.architecture = model.get("architecture", "RVC")
            
            # Instantiate appropriate generator based on the specified model architecture
            if self.architecture == "RVC":
                net_g = Synthesizer(
                    *model["config"], 
                    use_f0=self.use_f0, 
                    text_enc_hidden_dim=768 if self.version == "v2" else 256, 
                    vocoder=self.vocoder, 
                    checkpointing=False
                )
            else:
                net_g = SynthesizerSVC(
                    *model["config"], 
                    text_enc_hidden_dim=768 if self.version == "v2" else 256, 
                    vocoder=self.vocoder,
                    noise_scale=noise_scale
                )

            # Load weights into generator, send to target device, and set to evaluation mode
            net_g.load_state_dict(model["weight"], strict=False)
            net_g.eval().to(config.device).to(torch.float16 if config.is_half else torch.float32)
            # Remove weight normalization for faster inference execution
            net_g.remove_weight_norm()
            # Delete encoder queues or variables not utilized during inference stage to free VRAM
            del net_g.enc_q

            # Apply torch.compile if general optimization flags are enabled
            if config.compile_all: net_g = torch.compile(net_g, mode=config.compile_mode)

            self.net_g = net_g
            self.model = model
        else:
            # For ONNX models
            self.model = model
            self.net_g = self.model.to(config.device)
            self.tgt_sr = self.model.cpt.get("tgt_sr", 32000)
            self.use_f0 = self.model.cpt.get("f0", 1)
            self.version = self.model.cpt.get("version", "v1")
            self.architecture = self.model.cpt.get("architecture", "RVC")

        return self
    
    def inference(
        self, 
        feats, 
        p_len, 
        sid, 
        pitch, 
        pitchf, 
        rate
    ):
        """
        Executes raw core neural network forward generation.

        Args:
            feats (torch.Tensor): Extracted audio content features.
            p_len (torch.Tensor): Length of the pitch sequence/frames.
            sid (torch.Tensor): Speaker ID tensor.
            pitch (torch.Tensor): Coarse pitch sequence (integer indices).
            pitchf (torch.Tensor): Continuous fundamental frequency (F0) float sequence.
            rate (torch.Tensor): rate tensor.

        Returns:
            torch.Tensor: Normalized raw synthesized audio tensor.
        """

        # Execute generator inference and extract the first element batch/channel
        output = self.net_g.infer(
            feats, 
            p_len, 
            pitch, 
            pitchf,
            sid,
            rate
        )[0, 0]
        
        # Hard clamp output signal to avoid digital clipping artifacts
        return output.clamp_(-1.0, 1.0)
    
class Pipeline:
    """
    Coordinates the full conversion routine pipeline: including pitch prediction, 
    content embedding extraction, index searching/retrieval, and audio synthesis.
    """

    def __init__(
        self, 
        weight_root, 
        index_path=None, 
        f0_method = "rmvpe", 
        predictor_onnx=False, 
        embedders_mode="fairseq", 
        embedder_model="hubert_base", 
        noise_scale = 0.35,
        sample_rate=16000, 
        hop_length=160,
        nprobe=1, 
        sid = 0
    ):
        """
        Initializes the Pipeline setup with models, parameters, index indexes, and execution settings.
        """

        # Instantiate internal voice converter generator unit
        self.vc = RealtimeVoiceConverter(weight_root, noise_scale)

        self.f0_method = f0_method
        self.use_f0 = self.vc.use_f0
        self.tgt_sr = self.vc.tgt_sr
        self.version = self.vc.version
        # Initialize core modular blocks: Feature Extractors, Pitch Predictors, and FAISS Index
        self.embedder = self.setup_embedder(embedder_model, embedders_mode)
        self.predictor = self.setup_predictor(sample_rate, hop_length, predictor_onnx)
        # Clean up path layout string formatting and load matching FAISS cluster index vectors
        self.index, self.big_tsr = load_faiss_index(index_path.strip().strip('"').strip("\n").strip('"').strip().replace("trained", "added"), nprobe=nprobe, cpu_mode=False)

        # Configure system tensor variables cache to reduce memory initialization latency during stream loops
        self.sid = sid
        self.device = config.device
        self.is_half = config.is_half
        self.dtype = torch.float16 if self.is_half else torch.float32

        self.p_len = torch.zeros(1, device=self.device, dtype=torch.int64)
        self.rate = torch.zeros(1, device=self.device, dtype=self.dtype)
        self.torch_sid = torch.tensor([sid], device=self.device, dtype=torch.int64)
    
    def setup_predictor(self, sample_rate, hop_length, predictor_onnx = False):
        """
        Sets up the fundamental frequency (F0) estimator model if pitch tracking is enabled.
        """

        if self.use_f0:
            from main.library.predictors.Generator import Generator

            # Initialize pitch tracking generator context
            predictor = Generator(
                sample_rate=sample_rate, 
                hop_length=hop_length, 
                f0_min=configs.get("f0_min", 50), 
                f0_max=configs.get("f0_max", 1100), 
                alpha=0, 
                is_half=config.is_half, 
                device=config.device, 
                predictor_onnx=predictor_onnx, 
                return_tensor=True
            ) 
        else: predictor = None

        return predictor
    
    def setup_embedder(self, embedder_model, embedders_mode):
        """
        Loads and prepares content extraction embedder models (e.g., HuBERT).
        """

        embedder = load_embedders_model(
            embedder_model, 
            embedders_mode=embedders_mode
        )

        # Optimize and match tensor types for torch modules
        if isinstance(embedder, torch.nn.Module): 
            dtype = torch.float16 if config.is_half else torch.float32
            embedder = embedder.to(config.device).to(dtype).eval()
            if config.compile_all: embedder = torch.compile(embedder, mode=config.compile_mode)

        return embedder
    
    def inference(
        self, 
        audio, 
        pitch = None, 
        pitchf = None, 
        f0_up_key = 0, 
        index_rate = 0.5, 
        p_len = 0, 
        skip_head = None, 
        return_length = None, 
        protect = 0.5, 
        filter_radius = 3, 
        rms_mix_rate = 1, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0,
        torchgate = None,
        board = None,
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        block_size_16k = None,
    ):
        """
        Runs complete conversion inference on an audio chunk.
        """

        with torch.inference_mode():     
            assert audio.dim() == 1, audio.dim()
            formant_length = int(np.ceil(return_length * 1.0))
            # Calculate rolling audio slice boundaries and window sizes for real-time streams
            shift = (block_size_16k or skip_head * self.predictor.window) // self.predictor.window
            f0_frame = (block_size_16k + 800 if block_size_16k else skip_head * self.predictor.window + 800)
            # Adjust RMVPE frame block dimensions dynamically to match algorithm alignment constraints
            if "rmvpe" in self.f0_method: f0_frame = 5120 * ((f0_frame - 1) // 5120 + 1) - self.predictor.window

            if self.use_f0:
                # Evaluate pitch tracking using the calculated optimal audio slice window length
                pitch_new, pitchf_new = self.predictor.realtime_calculator(
                    audio[-min(f0_frame, audio.shape[0]):],
                    self.f0_method, 
                    None, 
                    None, 
                    f0_up_key, 
                    filter_radius, 
                    f0_autotune, 
                    f0_autotune_strength, 
                    proposal_pitch, 
                    proposal_pitch_threshold
                )

                pitch_new, pitchf_new = pitch_new.squeeze(0), pitchf_new.squeeze(0)
                # Shift pitch history frames backwards to maintain overlapping buffer structures
                if shift > 0:
                    pitch[:-shift] = pitch[shift:].clone()
                    pitchf[:-shift] = pitchf[shift:].clone()

                # Trim edge crossfade artifacts from raw real-time pitch estimations
                interior_pitch = pitch_new[3:-1] if pitch_new.shape[0] > 4 else pitch_new
                interior_pitchf = pitchf_new[3:-1] if pitchf_new.shape[0] > 4 else pitchf_new
                # Overwrite tail ends of buffers with the clean interior estimates
                pitch[-interior_pitch.shape[0]:] = interior_pitch
                pitchf[-interior_pitchf.shape[0]:] = interior_pitchf
            else: pitch, pitchf = None, None
            
            # Extract phonetic content embeddings from raw waveform tensor inputs
            feats = extract_features(
                self.embedder, 
                audio.view(1, -1), 
                self.version, 
                mix=embedders_mix, 
                mix_layers=embedders_mix_layers, 
                mix_ratio=embedders_mix_ratio
            )

            # Pad features sequence via trailing edge replication
            feats = torch.cat((feats, feats[:, -1:, :]), 1)
            feats0 = feats.detach().clone() if protect < 0.5 and self.use_f0 else None

            # Retrieve target vector traits from index to enhance accent/timbre fidelity
            if self.index is not None and self.big_tsr  is not None and index_rate != 0:
                try:
                    skip_offset = skip_head // 2
                    # Perform FAISS KNN query lookup on feature representations
                    score, ix = self.index.search(feats[0][skip_offset :], k=8)
                    weight = (1 / score).square()
                    query = (self.big_tsr[ix] * (weight / weight.sum(dim=1, keepdim=True)).unsqueeze(2)).sum(dim=1)
                    # Linearly blend original raw content embeddings with target vector index matches
                    feats[0][skip_offset :] = query.unsqueeze(0) * index_rate + (1.0 - index_rate) * feats[0][skip_offset :]
                except AssertionError:
                    logger.warning(translations["index_assertion"])
                    self.index = self.big_tsr = None

            # Resample content embedding dimensions to match the generator timing scale
            feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)[:, :p_len, :]
            # Align pitch timelines and scale by the formant length ratio
            if self.use_f0: pitch, pitchf = pitch[-p_len:].unsqueeze(0), pitchf[-p_len:].unsqueeze(0) * (formant_length / return_length)

            # Apply consonant protection mechanisms to preserve unvoiced pronunciation dynamics
            if feats0 is not None:
                pitchff = pitchf.detach().clone()
                pitchff[pitchf > 0] = 1
                pitchff[pitchf < 1] = protect
                pitchff = pitchff.unsqueeze(-1)

                feats0 = F.interpolate(
                    feats0.permute(0, 2, 1), 
                    scale_factor=2
                ).permute(0, 2, 1)[:, :p_len, :]

                feats = (feats * pitchff + feats0 * (1 - pitchff)).to(feats0.dtype)

            if self.use_f0: pitchf = pitchf.to(self.dtype)
            feats = feats.to(self.dtype)

            # Re-fill memory-cached structural metadata tensors prior to model forwarding
            self.rate.fill_(return_length / p_len)
            self.p_len.fill_(p_len)

            # Synthesize structural representations into target audio waveform
            out_audio = self.vc.inference(
                feats, 
                self.p_len, 
                self.torch_sid, 
                pitch, 
                pitchf,
                self.rate
            )

            # Apply volume/RMS mixing to restore the dynamics of the source audio
            if rms_mix_rate != 1: 
                out_audio = change_rms(
                    audio[-(return_length * self.predictor.window):], 
                    self.predictor.sample_rate, 
                    out_audio, 
                    self.tgt_sr, 
                    rms_mix_rate,
                    device=self.device
                )

            # Pass signal output through noise gate if available
            if torchgate is not None: 
                out_audio = torchgate(
                    out_audio.unsqueeze(0)
                ).squeeze(0)

            # Apply custom internal audio master board effects if present
            if board is not None: 
                out_audio = torch.as_tensor(
                    board(
                        out_audio.cpu().numpy(), 
                        self.tgt_sr
                    ), 
                    device=config.device
                )

            return out_audio.float()