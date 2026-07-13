import os
import sys
import time
import torch

import numpy as np
import torch.nn.functional as F
import torchaudio.transforms as tat

sys.path.append(os.getcwd())

from main.app.variables import config
from main.inference.realtime.pipeline import Pipeline
from main.library.utils import circular_write, check_assets, phase_vocoder

class Realtime:
    """
    Handles real-time audio block resampling, buffering, VAD, and model inference.

    This class serves as the functional interface for processing streaming raw
    audio chunks through the designated RVC conversion pipeline. It handles dynamic
    resampling, safety input volume thresholding, Voice Activity Detection (VAD),
    TorchGate noise reduction, and custom VST pedalboard effect applications.
    """

    def __init__(
        self, 
        model_path, 
        index_path = None, 
        f0_method = "rmvpe", 
        predictor_onnx = False, 
        embedder_model = "hubert_base", 
        embedders_mode = "fairseq", 
        sample_rate = 16000, 
        hop_length = 160, 
        silent_threshold = 0, 
        input_sample_rate = 48000, 
        output_sample_rate = 48000, 
        vad_enabled = False, 
        vad_sensitivity = 3, 
        vad_frame_ms = 30, 
        clean_audio=False, 
        clean_strength=0.7, 
        post_process = False, 
        sid = 0,
        noise_scale = 0.35,
        nprobe=1,
        **kwargs
    ):
        """Initializes the backend real-time inference processor."""

        # Check and download required model assets/weights if missing
        check_assets(
            f0_method, 
            embedder_model, 
            predictor_onnx=predictor_onnx, 
            embedders_mode=embedders_mode
        )
        # Storage for processing pipeline configurations
        self.model_path = model_path
        self.index_path = index_path
        self.f0_method = f0_method
        self.predictor_onnx = predictor_onnx
        self.embedder_model = embedder_model
        self.embedders_mode = embedders_mode
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.post_process = post_process
        self.sid = sid
        self.noise_scale = noise_scale
        self.nprobe = nprobe
        self.kwargs = kwargs
        # Initialize stream pipelines and buffers to be reallocated later
        self.pipeline = None
        self.audio_buffer = None
        self.convert_buffer = None
        self.pitch_buffer = None
        self.pitchf_buffer = None
        # Sizing boundaries used for context padding and feature offsets
        self.return_length = 0
        self.skip_head = 0
        self.silence_front = 0
        # Processing properties and hardware settings
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        self.vad_enabled = vad_enabled
        self.vad_sensitivity = vad_sensitivity
        self.vad_frame_ms = vad_frame_ms
        self.clean_audio = clean_audio
        self.clean_strength = clean_strength
        # Convert decibel threshold to raw numerical amplitude sensitivity limit
        self.input_sensitivity = 10 ** (silent_threshold / 20)
        self.window_size = sample_rate // 100
        self.dtype = torch.float16 if config.is_half else torch.float32

        # Conditional initialization for Voice Activity Detection
        if self.vad_enabled:
            from main.inference.realtime.vad_utils import VADProcessor

            self.vad = VADProcessor(
                sensitivity_mode=self.vad_sensitivity, 
                sample_rate=self.sample_rate, 
                frame_duration_ms=self.vad_frame_ms
            )
        else: self.vad = None

        # Build structural pedalboard chain if audio post-processing is enabled
        self.board = self.setup_pedalboard(**self.kwargs) if self.post_process else None
        # Instantiate core extraction and neural synthesis pipeline handler
        self.pipeline = Pipeline(
            weight_root=self.model_path, 
            index_path=self.index_path, 
            f0_method=self.f0_method, 
            predictor_onnx=self.predictor_onnx, 
            embedders_mode=self.embedders_mode, 
            embedder_model=self.embedder_model, 
            noise_scale=self.noise_scale,
            sample_rate=self.sample_rate, 
            hop_length=self.hop_length, 
            nprobe=self.nprobe,
            sid=self.sid
        )

        # Initialize output stream gate noise filter if requested
        if self.clean_audio:
            from main.library.audio.noisereduce import TorchGate

            self.tg = TorchGate(
                self.pipeline.tgt_sr, 
                prop_decrease=self.clean_strength
            ).to(config.device)
        else: self.tg = None

        # Define resampling modules for streaming IO
        self.resample_in = tat.Resample(
            orig_freq=self.input_sample_rate,
            new_freq=self.sample_rate,
            dtype=torch.float32
        ).to(config.device)
        self.resample_out = tat.Resample(
            orig_freq=self.pipeline.tgt_sr,
            new_freq=self.output_sample_rate,
            dtype=torch.float32
        ).to(config.device)

    def setup_pedalboard(self, **kwargs):
        """
        Constructs a serial audio effect processing rack using pedalboard components.

        Args:
            **kwargs: Configuration flags and scalar parameters for effects.

        Returns:
            An instantiated pedalboard.Pedalboard effect chain.
        """

        from pedalboard import (
            Pedalboard, 
            Chorus, 
            Distortion, 
            Reverb, 
            PitchShift, 
            Delay, 
            Limiter, 
            Gain, 
            Bitcrush, 
            Clipping, 
            Compressor, 
            Phaser, 
            HighpassFilter
        )

        # Initialize base chain with a Highpass filter to eliminate low-end rumble
        board = Pedalboard([HighpassFilter()])

        if kwargs["chorus"]:
            board.append(
                Chorus(
                    depth=kwargs["chorus_depth"], 
                    rate_hz=kwargs["chorus_rate"], 
                    mix=kwargs["chorus_mix"], 
                    centre_delay_ms=kwargs["chorus_delay"], 
                    feedback=kwargs["chorus_feedback"]
                )
            )
        
        if kwargs["distortion"]:
            board.append(
                Distortion(
                    drive_db=kwargs["distortion_gain"]
                )
            )

        if kwargs["reverb"]:
            board.append(
                Reverb(
                    room_size=kwargs["reverb_room_size"],
                    damping=kwargs["reverb_damping"],
                    wet_level=kwargs["reverb_wet_level"],
                    dry_level=kwargs["reverb_dry_level"],
                    width=kwargs["reverb_width"],
                    freeze_mode=int(kwargs["reverb_freeze_mode"])
                )
            )

        if kwargs["pitch_shift"]:
            board.append(
                PitchShift(
                    semitones=kwargs["pitch_shift_semitones"]
                )
            )

        if kwargs["delay"]:
            board.append(
                Delay(
                    delay_seconds=kwargs["delay_seconds"],
                    feedback=kwargs["delay_feedback"],
                    mix=kwargs["delay_mix"]
                )
            )

        if kwargs["compressor"]:
            board.append(
                Compressor(
                    threshold_db=kwargs["compressor_threshold"],
                    ratio=kwargs["compressor_ratio"],
                    attack_ms=kwargs["compressor_attack"],
                    release_ms=kwargs["compressor_release"]
                )
            )

        if kwargs["limiter"]:
            board.append(
                Limiter(
                    threshold_db=kwargs["limiter_threshold"],
                    release_ms=kwargs["limiter_release"]
                )
            )

        if kwargs["gain"]:
            board.append(
                Gain(
                    gain_db=kwargs["gain_db"]
                )
            )

        if kwargs["bitcrush"]:
            board.append(
                Bitcrush(
                    bit_depth=kwargs["bitcrush_bit_depth"]
                )
            )

        if kwargs["clipping"]:
            board.append(
                Clipping(
                    threshold_db=kwargs["clipping_threshold"]
                )
            )

        if kwargs["phaser"]: 
            board.append(
                Phaser(
                    rate_hz=kwargs["phaser_rate_hz"], 
                    depth=kwargs["phaser_depth"], 
                    centre_frequency_hz=kwargs["phaser_centre_frequency_hz"], 
                    feedback=kwargs["phaser_feedback"], 
                    mix=kwargs["phaser_mix"]
                )
            )

        return board

    def realloc(
        self, 
        block_frame, 
        extra_frame, 
        crossfade_frame, 
        sola_search_frame
    ):
        """
        Dynamically computes and allocates internal Tensor buffers based on frame layout.

        Args:
            block_frame: Frames per processing chunk.
            extra_frame: Extra historical frame lookback padding.
            crossfade_frame: Frame overlap segment length for mixing.
            sola_search_frame: Search window size used for SOLA cross-correlation.
        """

        # Rescale physical hardware frame measurements to the model's internal 16kHz context
        block_frame_16k = int(
            block_frame / self.input_sample_rate * self.sample_rate
        )
        crossfade_frame_16k = int(
            crossfade_frame / self.input_sample_rate * self.sample_rate
        )

        sola_search_frame_16k = int(
            sola_search_frame / self.input_sample_rate * self.sample_rate
        )
        extra_frame_16k = int(
            extra_frame / self.input_sample_rate * self.sample_rate
        )

        # Aggregate composite frame sizing required for model inference windowing
        convert_size_16k = (
            block_frame_16k + 
            sola_search_frame_16k + 
            extra_frame_16k + 
            crossfade_frame_16k
        )

        # Align conversion window bounds exactly to integer multiples of the internal window_size
        if (modulo := convert_size_16k % self.window_size) != 0: 
            convert_size_16k = convert_size_16k + (self.window_size - modulo)

        # Calculate logical feature dimensions based on step size
        self.convert_feature_size_16k = convert_size_16k // self.window_size
        self.block_frame_16k = block_frame_16k
        self.skip_head = extra_frame_16k // self.window_size
        self.return_length = self.convert_feature_size_16k - self.skip_head
        self.silence_front = extra_frame_16k - (self.window_size * 5) if self.silence_front else 0
        # Compute minimum initialization blocks required to fill up sliding buffers
        self.warmup_blocks = int(np.ceil(convert_size_16k / block_frame_16k)) + 1

        # Physically instantiate processing buffers on target accelerator memory
        audio_buffer_size = block_frame_16k + crossfade_frame_16k

        self.audio_buffer = torch.zeros(
            audio_buffer_size, 
            dtype=self.dtype, 
            device=config.device
        )
        self.convert_buffer = torch.zeros(
            convert_size_16k, 
            dtype=self.dtype, 
            device=config.device
        )
        self.pitch_buffer = torch.zeros(
            self.convert_feature_size_16k + 1, 
            dtype=torch.int64, 
            device=config.device
        )
        self.pitchf_buffer = torch.zeros(
            self.convert_feature_size_16k + 1, 
            dtype=self.dtype, 
            device=config.device
        )

    def inference(
        self, 
        audio_in, 
        f0_up_key = 0, 
        index_rate = 0.5, 
        protect = 0.5, 
        filter_radius = 3, 
        rms_mix_rate = 1, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0,
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5
    ):
        """
        Runs the streaming audio block payload through the RVC pipeline.

        Args:
            audio_in: Input NumPy vector containing the current block chunk.
            f0_up_key: Explicit semitone key shift factor.
            index_rate: Index retrieval mapping ratio.
            protect: Speech preservation masking strength factor.
            filter_radius: Median pitch smoothing window parameter.
            rms_mix_rate: Volume structural matching index.
            f0_autotune: Snaps fundamental frequency (F0) to musical notes.
            f0_autotune_strength: Correction intensity factor for note snapping.
            proposal_pitch: Automated pitch shift estimation flag via F0 median alignment.
            proposal_pitch_threshold: Boundary limit value for computed proposal pitch key shift.
            embedders_mix: Flag to apply composite multiple embeddings.
            embedders_mix_layers: Target specific layer layout from embedding networks.
            embedders_mix_ratio: Blending interpolation parameter for combined features.

        Returns:
            A tuple containing:
                - Optional[torch.Tensor]: Converted output audio tensor or None if bypassed.
                - float: Root Mean Square (RMS) volume calculation of the input block.
        """

        # Resample incoming audio chunk to the pipeline's expected 16kHz processing rate
        audio_in_16k = self.resample_in(
            torch.as_tensor(
                audio_in, 
                dtype=torch.float32, 
                device=config.device
            )
        ).to(self.dtype)

        # Enqueue new audio values safely inside the sliding circular storage ring
        circular_write(audio_in_16k, self.audio_buffer)
        # Compute the instantaneous root-mean-square amplitude of the current segment
        vol_t = self.audio_buffer.square().mean().sqrt()
        vol = max(vol_t.item(), 0)

        tg = self.tg
        board = self.board

        def inference_with_silent():
            """Helper to execute silent/dummy inference pass to update internal model memory states."""

            self.pipeline.inference(
                self.convert_buffer,
                self.pitch_buffer,
                self.pitchf_buffer,
                f0_up_key,
                index_rate,
                self.convert_feature_size_16k,
                self.skip_head,
                self.return_length,
                protect,
                filter_radius,
                rms_mix_rate,
                f0_autotune, 
                f0_autotune_strength, 
                proposal_pitch, 
                proposal_pitch_threshold,
                tg,
                board,
                embedders_mix,
                embedders_mix_layers,
                embedders_mix_ratio,
                block_size_16k=self.block_frame_16k
            )

            return None, vol

        # Handle system buffering warmup blocks sequentially without generating audio
        if self.warmup_blocks > 0:
            self.warmup_blocks -= 1
            circular_write(audio_in_16k, self.convert_buffer)
            return inference_with_silent()

        # Check Voice Activity Detection to bypass calculation during silent intervals
        if self.vad is not None:
            is_speech = self.vad.is_speech(audio_in_16k.cpu().numpy().copy())
            if not is_speech: return inference_with_silent()

        # Gate processing if block volume falls below ambient gate floor sensitivity parameters
        if vol < self.input_sensitivity: return inference_with_silent()
        # Update historical conversion tracking buffer
        circular_write(audio_in_16k, self.convert_buffer)

        # Dispatch audio frames into backend neural acoustic generator
        audio_model = self.pipeline.inference(
            self.convert_buffer,
            self.pitch_buffer,
            self.pitchf_buffer,
            f0_up_key,
            index_rate,
            self.convert_feature_size_16k,
            self.skip_head,
            self.return_length,
            protect,
            filter_radius,
            rms_mix_rate,
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold,
            tg,
            board,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            block_size_16k=self.block_frame_16k
        )

        # Upsample converted output back to targeted hardware execution sample rates
        audio_out = self.resample_out(audio_model * vol_t.sqrt())
        return audio_out, vol
    
class VoiceChanger:
    """
    Manages real-time cross-fading, SOLA alignment, and stream recording.

    This class provides stream stitching routines using the Synchronized Overlap-Add
    (SOLA) cross-correlation technique or Phase Vocoder to eliminate block-boundary
    clicking artifacts during live audio transformation.
    """

    def __init__(
        self, 
        read_chunk_size, 
        cross_fade_overlap_size, 
        input_sample_rate, 
        output_sample_rate,
        extra_convert_size,
        model_path, 
        index_path, 
        f0_method, 
        predictor_onnx, 
        embedder_model, 
        embedders_mode, 
        sample_rate, 
        hop_length, 
        silent_threshold,
        vad_enabled,
        vad_sensitivity,
        vad_frame_ms,
        clean_audio, 
        clean_strength,
        post_process, 
        sid,
        noise_scale,
        nprobe,
        record_audio,
        record_audio_path,
        export_format,
        **kwargs
    ):
        """Initializes the structural layout and overlap windows of the VoiceChanger interface."""

        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        # Calculate granular frame sizes based on parameters
        self.block_frame = read_chunk_size * 128
        self.crossfade_frame = int(cross_fade_overlap_size * input_sample_rate)
        self.extra_frame = int(extra_convert_size * input_sample_rate)
        self.sola_search_frame = input_sample_rate // 100
        # Uniform tensor utilized inside cross-correlation norm operations
        self.cor_den_ones = torch.ones(1, 1, self.crossfade_frame, device=config.device, dtype=torch.float32)
        # Instantiate underlying realtime model engine
        self.vc_model = Realtime(
            model_path, 
            index_path, 
            f0_method, 
            predictor_onnx, 
            embedder_model, 
            embedders_mode, 
            sample_rate, 
            hop_length, 
            silent_threshold, 
            self.input_sample_rate, 
            self.output_sample_rate,
            vad_enabled, 
            vad_sensitivity,
            vad_frame_ms,
            clean_audio, 
            clean_strength,
            post_process,
            sid,
            noise_scale,
            nprobe,
            **kwargs
        )
        # IO configuration details for local session capture
        self.record_audio = record_audio
        self.record_audio_path = record_audio_path
        self.export_format = export_format
        self.sola_buffer = None
        # Bootstrap and allocate structural arrays
        self.vc_model.realloc(
            self.block_frame, 
            self.extra_frame, 
            self.crossfade_frame, 
            self.sola_search_frame
        )
        self.generate_strength()
        self.setup_soundfile_record()

    def setup_soundfile_record(self):
        """Initializes the background file recorder for the session."""

        import soundfile as sf

        self.soundfile = sf.SoundFile(
            self.record_audio_path,
            mode="w",
            samplerate=self.output_sample_rate,
            channels=1,
            format=self.export_format.lower(),
        ) if self.record_audio else None

    def generate_strength(self):
        """Generates mathematical window contours for overlap cross-fading."""

        # Calculate sinusoidal fade-in window coefficients
        self.fade_in_window = (
            0.5 * np.pi * torch.linspace(0.0, 1.0, steps=self.crossfade_frame, device=config.device, dtype=torch.float32)
        ).sin() ** 2
        # Complementary inversion yields a matching linear fade-out path
        self.fade_out_window = 1 - self.fade_in_window

        # Storage for historical phase overlaps
        self.sola_buffer = torch.zeros(
            self.crossfade_frame, 
            device=config.device, 
            dtype=torch.float32
        )

    def process_audio(
        self, 
        audio_in, 
        f0_up_key = 0, 
        index_rate = 0.5, 
        protect = 0.5, 
        filter_radius = 3, 
        rms_mix_rate = 1, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0,
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        use_phase_vocoder = True
    ):
        """
        Performs SOLA stitching or Phase Vocoder processing on an audio block.

        Args:
            audio_in: Array containing streaming input.
            f0_up_key: Target pitch modifier key.
            index_rate: Feature retrieval search influence multiplier.
            protect: Envelope conservation factor parameter.
            filter_radius: Pitch track filter smooth index bounds.
            rms_mix_rate: Target loudness preservation mapping index.
            f0_autotune: Snaps fundamental frequency (F0) to musical notes.
            f0_autotune_strength: Correction intensity factor for note snapping.
            proposal_pitch: Automated optimal key estimation flag via F0 median alignment.
            proposal_pitch_threshold: Boundary ceiling/floor limit for calculated proposal pitch shift key.
            embedders_mix: Combined multi-embedding processing toggle.
            embedders_mix_layers: Specific layers targeted for blending extraction.
            embedders_mix_ratio: Inter-embedding scale mix factor.
            use_phase_vocoder: Enable phase vocoder..

        Returns:
            A tuple containing:
                - np.ndarray: Glitch-free output audio chunk.
                - float: Detected block level RMS metric values.
        """

        block_size = audio_in.shape[0]
        # Fetch output blocks from underlying neural transformer model
        audio, vol = self.vc_model.inference(
            audio_in, 
            f0_up_key, 
            index_rate, 
            protect, 
            filter_radius, 
            rms_mix_rate, 
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio
        )

        # Handle system-bypass conditions or empty processing responses safely
        if audio is None: 
            self.sola_buffer.zero_()
            audio = np.zeros(block_size, dtype=np.float32)

            if self.record_audio and self.soundfile is not None: self.soundfile.write(audio)
            return audio, vol

        # Identify if current chunk marks the start of a transmission sequence
        is_onset = not self.sola_buffer.any()
        # Compute cross-correlation between current stream buffer and past frames using 1D convolution
        conv_input = audio[None, None, : self.crossfade_frame + self.sola_search_frame].float()
        cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])

        cor_den = (
            F.conv1d(
                conv_input ** 2, 
                self.cor_den_ones
            ) + 1e-8
        ).sqrt()

        # Find the optimal overlap point that maximizes signal cross-correlation alignment
        sola_offset = (cor_nom[0, 0] / cor_den[0, 0]).argmax()
        audio = audio[sola_offset:]

        # Execute window merging routines depending on current execution paths
        if use_phase_vocoder and not config.device.startswith(("privateuseone", "ocl")):
            # Resolve boundary phase discontinuities explicitly using STFT Phase Vocoder processing
            audio[: self.crossfade_frame] = phase_vocoder(
                self.sola_buffer,
                audio[: self.crossfade_frame],
                self.fade_out_window,
                self.fade_in_window
            )
        else:
            if is_onset:
                # Suppress attack transients at the start of speech onset periods to reduce clicks
                n_hops = block_size // 160
                if n_hops >= 1:
                    hop_energy = audio[: n_hops * 160].reshape(n_hops, 160).abs().max(dim=1).values
                    peak = hop_energy.max().item()
                    onset_sample = 0

                    if peak > 1e-4:
                        above = (hop_energy > 0.1 * peak).nonzero(as_tuple=False)
                        if len(above) > 0: onset_sample = int(above[0].item()) * 160
                else: onset_sample = 0

                audio[:onset_sample] = 0.0
                fade_len = min(block_size - onset_sample, self.crossfade_frame)
                if fade_len > 0: audio[onset_sample : onset_sample + fade_len] *= self.fade_in_window[:fade_len]
            else:
                # Apply standard crossfade blending matrices across block junctions
                audio[: self.crossfade_frame] *= self.fade_in_window
                audio[: self.crossfade_frame] += self.sola_buffer * self.fade_out_window

        # Enforce consistent audio lengths by padding missing trailing segments if needed
        min_len = block_size + self.crossfade_frame
        if audio.shape[0] < min_len: audio = torch.cat([audio, torch.zeros(min_len - audio.shape[0], device=config.device, dtype=audio.dtype)])
        # Update historical overlap buffers for the next incoming chunk
        self.sola_buffer[:] = audio[block_size : block_size + self.crossfade_frame]
        audio_output = audio[:block_size].detach().cpu().numpy()

        # Write converted blocks to local session storage files if enabled
        if self.record_audio and self.soundfile is not None: self.soundfile.write(audio_output)
        return audio_output, vol
    
    @torch.no_grad()
    def on_request(
        self, 
        audio_in, 
        f0_up_key = 0, 
        index_rate = 0.5, 
        protect = 0.5, 
        filter_radius = 3, 
        rms_mix_rate = 1, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0,
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        use_phase_vocoder = True
    ):
        """
        Wrapper method that benchmarks processing latency while converting audio blocks.

        Args:
            All arguments correspond directly to `process_audio` parameters.

        Returns:
            A tuple containing:
                - np.ndarray: Stitched transformed output waveform values.
                - float: Calculated block RMS level metric.
                - float: Total duration elapsed during calculation (in milliseconds).
        """

        start = time.perf_counter()

        result, vol = self.process_audio(
            audio_in, 
            f0_up_key, 
            index_rate, 
            protect, 
            filter_radius, 
            rms_mix_rate, 
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            use_phase_vocoder
        )

        end = time.perf_counter()
        # Compute total latency processing metrics back to the streaming execution pipeline
        return result, vol, (end - start) * 1000