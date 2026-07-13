import os
import sys
import threading

import numpy as np

sys.path.append(os.getcwd())

from main.inference.realtime.audio import Audio
from main.app.variables import logger, translations
from main.inference.realtime.realtime import VoiceChanger

class AudioCallbacks:
    """
    Manages audio processing callbacks for real-time voice changing.

    This class bridges the low-level audio stream handling (via Audio class)
    and the core voice conversion inference engine (via VoiceChanger class).
    It supports multi-threaded processing with thread safety using locks.
    """

    def __init__(
        self, 
        pass_through = False, 
        read_chunk_size = 192, 
        cross_fade_overlap_size = 0.1, 
        input_sample_rate = 48000, 
        output_sample_rate = 48000, 
        extra_convert_size = 0.5, 
        model_path = None, 
        index_path = None, 
        f0_method = "rmvpe", 
        predictor_onnx = False, 
        embedder_model = "hubert_base", 
        embedders_mode = "fairseq", 
        sample_rate = 16000, 
        hop_length = 160, 
        silent_threshold = -90, 
        f0_up_key = 0, 
        index_rate = 0.5, 
        protect = 0.5, 
        filter_radius = 3, 
        rms_mix_rate = 1, 
        f0_autotune = False, 
        f0_autotune_strength = 1, 
        proposal_pitch = False, 
        proposal_pitch_threshold = 255.0, 
        input_audio_gain = 1.0, 
        output_audio_gain = 1.0, 
        monitor_audio_gain = 1.0, 
        monitor = False, 
        vad_enabled = False, 
        vad_sensitivity = 3, 
        vad_frame_ms = 30, 
        clean_audio = False, 
        clean_strength = 0.7, 
        post_process = False, 
        sid = 0,
        noise_scale = 0.35,
        nprobe = 1,
        record_audio = False,
        record_audio_path = None,
        export_format = "WAV",
        embedders_mix = False,
        embedders_mix_layers = 9,
        embedders_mix_ratio = 0.5,
        use_phase_vocoder = True,
        **kwargs
    ):
        """
        Initializes the AudioCallbacks handler with voice changer configurations.

        Args:
            pass_through: If True, bypasses conversion and returns raw audio.
            read_chunk_size: Number of audio samples to process per chunk.
            cross_fade_overlap_size: Fraction of overlap used for cross-fading chunks.
            input_sample_rate: Sampling rate of the incoming audio stream.
            output_sample_rate: Sampling rate of the outgoing audio stream.
            extra_convert_size: Extra padding buffer size for context.
            model_path: Path to the target model checkpoint file.
            index_path: Path to the retrieval index file (.index).
            f0_method: Pitch extraction algorithm (e.g., 'rmvpe', 'harvest').
            predictor_onnx: Whether to use ONNX runtime for pitch estimation.
            embedder_model: Name or path of the feature extractor model.
            embedders_mode: Framework mode for embedding (e.g., 'fairseq').
            sample_rate: Internal processing sample rate.
            hop_length: Frame hop size used in analysis.
            silent_threshold: Decibel threshold below which audio is treated as silence.
            f0_up_key: Pitch shift key in semitones (positive or negative).
            index_rate: Influence ratio of the feature index retrieval.
            protect: Protection level for vowels/consonants against artifacts.
            filter_radius: Median filter radius applied to extracted pitch.
            rms_mix_rate: Mix ratio of original audio envelope to converted.
            f0_autotune: Enable snapping the F0 pitch sequence to the nearest musical notes.
            f0_autotune_strength: Blend factor for autotune (0.0 = raw pitch, 1.0 = fully snapped).
            proposal_pitch: Enable automatic pitch key shifting calculation based on median F0 alignment.
            proposal_pitch_threshold: The maximum allowed semitone boundary limit (floor/ceiling) for the proposed pitch shift calculation.
            input_audio_gain: Digital volume multiplier for input stream.
            output_audio_gain: Digital volume multiplier for output stream.
            monitor_audio_gain: Volume multiplier for the dedicated monitoring output device.
            monitor: Enable real-time audio routing to a 3rd output device for monitoring.
            vad_enabled: Enable Voice Activity Detection.
            vad_sensitivity: Aggressiveness level for filtering non-speech (1-3).
            vad_frame_ms: Frame duration in milliseconds for VAD analysis.
            clean_audio: Enable noise reduction/denoising on the output audio stream.
            clean_strength: Intensity parameter for the output noise reduction.
            post_process: Enable additional acoustic post-filtering effects.
            sid: Speaker ID index (for multi-speaker checkpoints).
            noise_scale: Scale factor for exploratory noise during generation.
            nprobe: Number of clusters to query in FAISS index retrieval.
            record_audio: Enable background recording of the session.
            record_audio_path: Directory path where recordings are saved.
            export_format: Audio format for saved file (e.g., 'WAV', 'MP3').
            embedders_mix: Enable blending multiple feature extraction models.
            embedders_mix_layers: Target layers from embedder to extract features from.
            embedders_mix_ratio: Weight ratio when mixing embedding features.
            use_phase_vocoder: Enable phase vocoder.
            **kwargs: Dynamic configurations passed directly to the post-processing engine.
        """

        # Save routing flag and stream sampling properties
        self.pass_through = pass_through
        self.input_sample_rate = input_sample_rate
        self.output_sample_rate = output_sample_rate
        # Initialize thread lock to prevent concurrent modifications during inference
        self.lock = threading.Lock()
        # Instantiate the backend inference handler (kwargs holds post-process configs)
        self.vc = VoiceChanger(
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
        )
        # Initialize the audio hardware / stream IO wrapper (including monitor device routing)
        self.audio = Audio(
            self, 
            f0_up_key, 
            index_rate, 
            protect, 
            filter_radius, 
            rms_mix_rate,
            f0_autotune, 
            f0_autotune_strength, 
            proposal_pitch, 
            proposal_pitch_threshold,
            input_audio_gain, 
            output_audio_gain, 
            monitor_audio_gain,
            monitor,
            embedders_mix,
            embedders_mix_layers,
            embedders_mix_ratio,
            use_phase_vocoder
        )

    def change_voice(
        self, 
        received_data, 
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
        Processes an incoming raw audio chunk and applies the voice transformation.

        Args:
            received_data: Input audio waveform array (float32).
            f0_up_key: Pitch shift key in semitones.
            index_rate: Feature retrieval influence scale.
            protect: Protection metric for speech structural preservation.
            filter_radius: Pitch smoothing filter window radius.
            rms_mix_rate: Volume envelope blending parameter.
            f0_autotune: Snaps fundamental frequency (F0) to musical notes.
            f0_autotune_strength: Correction intensity factor for note snapping.
            proposal_pitch: Enable automated optimal semitone key estimation via F0 median alignment.
            proposal_pitch_threshold: Cap limit boundary for calculation of the calculated key offset.
            embedders_mix: Flag to apply composite multiple embeddings.
            embedders_mix_layers: Total specific model layers to combine.
            embedders_mix_ratio: Mix blending balance factor.
            use_phase_vocoder: Enable phase vocoder.

        Returns:
            A tuple containing:
                - np.ndarray: The modified audio array (or original if bypassed).
                - float: Root Mean Square (RMS) volume metrics of the block.
                - float: Performance tracking timings or metrics from inference.
        """

        # If pass-through mode is active, skip voice morphing entirely
        if self.pass_through:
            # Calculate the Root Mean Square (RMS) volume of raw input data safely
            vol = float(np.sqrt(np.square(received_data).mean(dtype=np.float32)))
            # Return raw data, volume metric, empty performance metrics block, and None placeholder
            return received_data, vol, 0

        try:
            # Acquire lock to ensure only one audio chunk is processed by the model at a time
            with self.lock:
                # Dispatch audio payload to the main Voice Changer inference engine
                audio, vol, perf = self.vc.on_request(
                    received_data, 
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

            return audio, vol, perf
        except RuntimeError as e:
            import traceback

            # Log detailed fallback traceback information for debugging purposes
            logger.debug(traceback.format_exc())
            # Emit localized user-friendly error interface notifications
            logger.error(translations["error_occurred"].format(e=e))

            # Return a muted fallback buffer to prevent stream audio corruption or crash spikes
            return np.zeros(1, dtype=np.float32), 0, 0