import os
import sys
import librosa

import numpy as np
import soundfile as sf

sys.path.append(os.getcwd())

from main.library.uvr5_lib.spec_utils import normalize

class CommonSeparator:
    """
    A foundational separator class responsible for managing configurations, 
    audio I/O operations, stem mapping, and post-processing tasks for audio source separation.
    """

    # Audio stem type constants
    VOCAL_STEM = "Vocals"
    INST_STEM = "Instrumental"
    OTHER_STEM = "Other"
    BASS_STEM = "Bass"
    DRUM_STEM = "Drums"
    GUITAR_STEM = "Guitar"
    PIANO_STEM = "Piano"
    SYNTH_STEM = "Synthesizer"
    STRINGS_STEM = "Strings"
    WOODWINDS_STEM = "Woodwinds"
    BRASS_STEM = "Brass"
    WIND_INST_STEM = "Wind Inst"
    PRIMARY_STEM = "Primary Stem"
    SECONDARY_STEM = "Secondary Stem"
    LEAD_VOCAL_STEM = "lead_only"
    BV_VOCAL_STEM = "backing_only"
    NO_STEM = "No "

    # Maps a primary stem to its logical complementary/secondary counterpart
    STEM_PAIR_MAPPER = {
        VOCAL_STEM: INST_STEM, 
        INST_STEM: VOCAL_STEM, 
        LEAD_VOCAL_STEM: BV_VOCAL_STEM, 
        BV_VOCAL_STEM: LEAD_VOCAL_STEM, 
        PRIMARY_STEM: SECONDARY_STEM
    }

    # Stems that are not categorized as traditional accompaniment
    NON_ACCOM_STEMS = (
        VOCAL_STEM, 
        OTHER_STEM, 
        BASS_STEM, 
        DRUM_STEM, 
        GUITAR_STEM, 
        PIANO_STEM, 
        SYNTH_STEM, 
        STRINGS_STEM, 
        WOODWINDS_STEM, 
        BRASS_STEM, 
        WIND_INST_STEM
    )

    def __init__(self, config):
        """
        Initializes the CommonSeparator with configuration parameters.

        Args:
            config (dict): Configuration dictionary containing hardware settings, 
                           model paths, and audio processing hyperparameters.
        """

        # Logging and Hardware Execution Targets
        self.logger = config.get("logger")
        self.onnx_execution_provider = config.get("onnx_execution_provider")
        self.torch_device = config.get("torch_device")
        # Audio Processing & Post-Processing Hyperparameters
        self.normalization_threshold = config.get("normalization_threshold")
        self.invert_using_spec = config.get("invert_using_spec")
        self.enable_denoise = config.get("enable_denoise")
        self.output_format = config.get("output_format")
        self.sample_rate = config.get("sample_rate")
        # Model Metadata and File Paths
        self.model_name = config.get("model_name")
        self.model_path = config.get("model_path")
        self.model_data = config.get("model_data")
        # Target Output Directory
        self.output_dir = config.get("output_dir")
        # Track active stems for the current task
        self.secondary_stem_name = None
        self.primary_stem_name = None

        # Attempt to extract target stems from model training metadata if available
        if "training" in self.model_data and "instruments" in self.model_data["training"]:
            instruments = self.model_data["training"]["instruments"]
            # If a second instrument exists, use it; otherwise, dynamically find the secondary pair
            if instruments: self.primary_stem_name, self.secondary_stem_name = instruments[0], instruments[1] if len(instruments) > 1 else self.secondary_stem(self.primary_stem_name)

        # Fallback: Extract stem configuration from general keys if training data is missing
        if self.primary_stem_name is None:
            self.primary_stem_name = self.model_data.get("primary_stem", "Vocals")
            self.secondary_stem_name = self.secondary_stem(self.primary_stem_name)

        # File-specific state variables (Reset per audio file)
        self.secondary_stem_output_path = None
        self.primary_stem_output_path = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None

    def secondary_stem(self, primary_stem):
        """
        Determines the appropriate secondary stem name based on the primary stem name.

        Args:
            primary_stem (str): The name of the primary source stem.

        Returns:
            str: The corresponding secondary stem name.
        """

        primary_stem = primary_stem if primary_stem else self.NO_STEM
        return self.STEM_PAIR_MAPPER[primary_stem] if primary_stem in self.STEM_PAIR_MAPPER else primary_stem.replace(self.NO_STEM, "") if self.NO_STEM in primary_stem else f"{self.NO_STEM}{primary_stem}"

    def separate(self, audio_file_path):
        """
        Placeholder method for the core audio separation logic. 
        Must be implemented by subclasses.

        Args:
            audio_file_path (str): Path to the input audio file.
        """

        pass

    def final_process(self, stem_path, source, stem_name):
        """
        Executes final saving procedures on an isolated source stem.

        Args:
            stem_path (str): Relative destination filename/path for the stem.
            source (np.ndarray): Waveform data array of the processed audio stem.
            stem_name (str): Key identifying name for the dictionary output.

        Returns:
            dict: Map matching the stem identifier name to its audio array.
        """

        self.write_audio_soundfile(stem_path, source)
        return {stem_name: source}

    def prepare_mix(self, mix):
        """
        Loads and prepares input audio into a uniform format ready for model processing.

        Args:
            mix (str or np.ndarray): File path string to audio or pre-loaded numpy array.

        Returns:
            np.ndarray: A Fortran-contiguous stereo audio waveform array.
        """

        # Load file path into numpy array via librosa if string is passed
        if not isinstance(mix, np.ndarray): mix, _ = librosa.load(mix, mono=False, sr=self.sample_rate)
        else: mix = mix.T # Transpose if already a numpy array to align axis formatting

        # Convert mono audio (1D) to a stereo format (2D Fortran array)
        if mix.ndim == 1: mix = np.asfortranarray([mix, mix])
        return mix

    def write_audio_soundfile(self, stem_path, stem_source):
        """
        Normalizes, prepares memory layout, and exports the audio stem to disk.

        Args:
            stem_path (str): Target output filename or path.
            stem_source (np.ndarray): Audio data array to write out.
        """

        # Peak normalization based on configuration threshold
        stem_source = normalize(wave=stem_source, max_peak=self.normalization_threshold)
        # Skip export if the audio track contains near-absolute silence
        if np.max(np.abs(stem_source)) < 1e-6: return

        # Ensure output directory exists before writing file
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            stem_path = os.path.join(self.output_dir, stem_path)

        # Convert multi-channel Fortran-contiguous arrays to standard C-contiguous 
        # structure to guarantee full library compatibility with soundfile.write
        if stem_source.shape[1] == 2 and stem_source.flags["F_CONTIGUOUS"]: stem_source = np.ascontiguousarray(stem_source)
        sf.write(stem_path, stem_source, self.sample_rate)

    def clear_file_specific_paths(self):
        """
        Resets instance variables linked to individual files to avoid memory 
        leaks or data cross-contamination during batch processing loops.
        """

        self.secondary_stem_output_path = None
        self.primary_stem_output_path = None
        self.secondary_source = None
        self.audio_file_path = None
        self.audio_file_base = None
        self.primary_source = None