import os
import math
import torch
import random
import librosa
import torchaudio

from io import IOBase

class Audio:
    """
    A comprehensive utility class for audio processing, loading, resampling, 

    downmixing, and precision cropping using PyTorch, torchaudio, and librosa.
    """

    @staticmethod
    def power_normalize(waveform):
        """
        Normalizes the waveform to have a Root Mean Square (RMS) power of 1.

        Args:
            waveform (torch.Tensor): Audio waveform tensor of shape (channels, samples).

        Returns:
            torch.Tensor: Power-normalized waveform tensor.
        """

        # Calculate RMS along the last dimension and avoid division by zero with a small epsilon (1e-8)
        return waveform / (waveform.square().mean(dim=-1, keepdim=True).sqrt() + 1e-8)

    @staticmethod
    def validate_file(file):
        """
        Validates and standardizes various audio file input types into a structured dictionary.

        Args:
            file (Union[str, os.PathLike, IOBase, Dict[str, Any]]): File path, file-like stream, or dictionary containing audio specifications.

        Returns:
            Dict[str, Any]: Standardized file metadata dictionary.

        Raises:
            ValueError: If the input format or internal shape metadata is invalid.
        """

        if isinstance(file, (str, os.PathLike)): 
            # Case 1: Input is a file path string or PathLike object
            file = {
                "audio": str(file), 
                "uri": os.path.splitext(os.path.basename(file))[0] # Use file name without extension as URI
            }
        elif isinstance(file, IOBase): 
            # Case 2: Input is an open file-like stream (IOBase)
            return {
                "audio": file, 
                "uri": "stream"
            }
        else: 
            raise ValueError("Unsupported file type provided.")

        # Validation logic if a raw waveform tensor is already provided in the dictionary
        if "waveform" in file:
            waveform = file["waveform"]
            # Ensure waveform is 2D (channels, samples) and channels < samples
            if len(waveform.shape) != 2 or waveform.shape[0] > waveform.shape[1]: raise ValueError("Waveform must be 2D with shape (channels, samples).")

            # Sample rate is mandatory if an explicit waveform is provided
            sample_rate = file.get("sample_rate", None)
            if sample_rate is None: raise ValueError("sample_rate must be provided alongside waveform.")

            file.setdefault("uri", "waveform")
        elif "audio" in file:
            # Validation logic if an audio path/stream pointer is provided in the dictionary
            if isinstance(file["audio"], IOBase): return file
            # Convert relative paths to absolute paths
            path = os.path.abspath(file["audio"])
            file.setdefault("uri", os.path.splitext(os.path.basename(path))[0])
        else: 
            raise ValueError("The dictionary must contain either 'waveform' or 'audio' keys.")

        return file

    def __init__(
        self, 
        sample_rate = None, 
        mono=None, 
        backend = None
    ):
        """
        Initializes the Audio processor with specific target configurations.

        Args:
            sample_rate (int, optional): Target sample rate for resampling. Defaults to None (no resampling).
            mono (str, optional): Downmix strategy for multi-channel audio. Options: "random", "downmix". Defaults to None.
            backend (str, optional): I/O backend for torchaudio. Defaults to "soundfile".
        """

        super().__init__()
        self.sample_rate = sample_rate
        self.mono = mono

        # Default to 'soundfile' backend if none is specified
        if not backend: backend = "soundfile"
        self.backend = backend

    def downmix_and_resample(self, waveform, sample_rate):
        """
        Performs channel downmixing and sample rate conversion based on instance configurations.

        Args:
            waveform (torch.Tensor): Audio waveform tensor of shape (channels, samples).
            sample_rate (int): Current sample rate of the input waveform.

        Returns:
            Tuple[torch.Tensor, int]: Processed waveform tensor and its new sample rate.
        """

        num_channels = waveform.shape[0]
        # Handle multi-channel downmixing if mono configuration is active
        if num_channels > 1:
            if self.mono == "random":
                # Select one random channel from the available channels
                channel = random.randint(0, num_channels - 1)
                waveform = waveform[channel : channel + 1]
            elif self.mono == "downmix": 
                # Average all channels to create a single mono channel
                waveform = waveform.mean(dim=0, keepdim=True)

        # Resample the waveform if a target sample rate is set and differs from current sample rate
        if (self.sample_rate is not None) and (self.sample_rate != sample_rate):
            waveform = torchaudio.functional.resample(
                waveform, 
                sample_rate, 
                self.sample_rate
            )

            sample_rate = self.sample_rate

        return waveform, sample_rate

    def get_num_samples(self, duration, sample_rate = None):
        """
        Calculates the total number of samples for a given duration.

        Args:
            duration (float): Duration in seconds.
            sample_rate (int, optional): Sample rate to use. Defaults to instance sample_rate.

        Returns:
            int: Calculated total number of samples (floored).

        Raises:
            ValueError: If no valid sample rate is available.
        """

        sample_rate = sample_rate or self.sample_rate
        if sample_rate is None: raise ValueError("Sample rate must be provided or configured globally.")

        return math.floor(duration * sample_rate)

    def __call__(self, file):
        """
        Loads and processes the complete audio file or dictionary configuration.

        Args:
            file (Union[str, os.PathLike, IOBase, Dict[str, Any]]): The input audio source descriptor.

        Returns:
            Tuple[torch.Tensor, int]: Fully processed waveform tensor and its sample rate.
        """

        file = self.validate_file(file)
        # Extract waveform directly if already loaded in dictionary
        if "waveform" in file:
            waveform = file["waveform"]
            sample_rate = file["sample_rate"]
        elif "audio" in file: # Otherwise, read and load the audio data from disk or stream
            try:
                # Primary attempt: Use torchaudio with the specified backend
                waveform, sample_rate = torchaudio.load(file["audio"], backend=self.backend)
            except:
                # Fallback attempt: Use librosa if torchaudio fails (e.g., format or backend issue)
                y, sample_rate = librosa.load(file["audio"], sr=None, mono=False)
                waveform = torch.from_numpy(y)

            # Reset stream pointer position to the beginning if input is an IOBase object
            if isinstance(file["audio"], IOBase): file["audio"].seek(0)

        # Slice specific channel if explicitly requested in the file dictionary
        channel = file.get("channel", None)
        if channel is not None: waveform = waveform[channel : channel + 1]

        # Apply downmixing and resampling configurations
        return self.downmix_and_resample(waveform, sample_rate)

    def crop(self, file, segment, duration = None, mode="raise"):
        """
        Extracts a precise temporal slice (crop) from an audio file or waveform.

        Args:
            file (Union[str, os.PathLike, IOBase, Dict[str, Any]]): The input audio source descriptor.
            segment (Any): An object representing the segment boundaries (must contain .start and .end in seconds).
            duration (float, optional): Exact duration to extract in seconds. Overrides segment.end if specified.
            mode (str, optional): Out-of-bounds handling strategy. Options: "raise", "pad". Defaults to "raise".

        Returns:
            Tuple[torch.Tensor, int]: Cropped waveform slice and its sample rate.

        Raises:
            ValueError: If constraints are violated under "raise" mode.
            RuntimeError: On unrecoverable streaming / I/O reading errors.
        """

        file = self.validate_file(file)
        # Determine total frames and sample rate based on metadata availability
        if "waveform" in file:
            waveform = file["waveform"]
            frames = waveform.shape[1]
            sample_rate = file["sample_rate"]
        elif "torchaudio.info" in file:
            info = file["torchaudio.info"]
            frames = info.num_frames
            sample_rate = info.sample_rate
        else:
            # Heavy fallback: Load metadata attributes using librosa
            info, sr = librosa.load(file["audio"], sr=None)
            frames = info.shape[0]
            sample_rate = sr

        channel = file.get("channel", None)
        start_frame = math.floor(segment.start * sample_rate)
        # Compute targeted frame ranges based on segment limits or requested duration
        if duration:
            num_frames = math.floor(duration * sample_rate)
            end_frame = start_frame + num_frames
        else:
            end_frame = math.floor(segment.end * sample_rate)
            num_frames = end_frame - start_frame

        # Boundary Management - Strategy 1: Strict Validation ("raise" mode)
        if mode == "raise":
            if num_frames > frames: raise ValueError("Requested duration is longer than the file length.")
            # Allow a tiny tolerance window (1ms) before discarding due to precision issues
            if end_frame > frames + math.ceil(0.001 * sample_rate): raise ValueError("Requested crop segment extends past the end of the file.")
            else:
                end_frame = min(end_frame, frames)
                start_frame = end_frame - num_frames

            if start_frame < 0: raise ValueError
        elif mode == "pad":
            # Boundary Management - Strategy 2: Padding Preparation ("pad" mode)
            pad_start = -min(0, start_frame)
            pad_end = max(end_frame, frames) - frames

            start_frame = max(0, start_frame)
            end_frame = min(end_frame, frames)

            num_frames = end_frame - start_frame

        # Slice the audio data based on the source location
        if "waveform" in file: 
            # In-memory slice if the tensor is already fully allocated inside the dict
            data = file["waveform"][:, start_frame:end_frame]
        else:
            try:
                try:
                    # Optimized partial loading using torchaudio with frame offsets
                    data, _ = torchaudio.load(
                        file["audio"], 
                        frame_offset=start_frame, 
                        num_frames=num_frames, 
                        backend=self.backend
                    )
                except:
                    # Fallback partial loading using librosa
                    y, _ = librosa.load(
                        file["audio"], 
                        sr=sample_rate, 
                        offset=start_frame / sample_rate, 
                        duration=num_frames / sample_rate, 
                        mono=False
                    )
                    data = torch.from_numpy(y)
                    # Keep tensor 2D even if loaded as single channel (mono)
                    data = data.unsqueeze(0) if len(data.shape) == 1 else data

                if isinstance(file["audio"], IOBase): file["audio"].seek(0)
            except RuntimeError:
                # Streams cannot be re-read efficiently via fallback, raise immediately
                if isinstance(file["audio"], IOBase): raise RuntimeError("Failed to seek or slice the given streaming IO handle.")

                # Fallback: Load entire file into cache structure, then execute localized index slice
                waveform, sample_rate = self.__call__(file)
                data = waveform[:, start_frame:end_frame]

                file["waveform"] = waveform
                file["sample_rate"] = sample_rate

        # Filter channel tracks post-crop if a channel constraint exists
        if channel is not None: data = data[channel : channel + 1, :]
        # Apply Zero-Padding if mode was set to 'pad'
        if mode == "pad": data = torch.nn.functional.pad(data, (pad_start, pad_end))

        # Pass cropped array through structural normalization/resampling filters
        return self.downmix_and_resample(data, sample_rate)