import webrtcvad

import numpy as np

class VADProcessor:
    """
    A voice activity detector wrapper centered around WebRTC VAD to handle structural multi-channel 
    downmixing, real-time PCM audio conversion, frame chunking, and continuous speech segment parsing.
    """

    def __init__(
        self, 
        sensitivity_mode=3, 
        sample_rate=16000, 
        frame_duration_ms=30
    ):
        """
        Initializes the WebRTC VAD instance and calculates frame alignments based on mathematical constraints.

        Args:
            sensitivity_mode (int): Aggressiveness level filters for filtering out noise (0 to 3). 3 is the most aggressive.
            sample_rate (int): Audio sampling frequency rate. Must strictly be 8000, 16000, 32000, or 48000 Hz.
            frame_duration_ms (int): Target time window length per slice context. Must strictly be 10, 20, or 30 ms.

        Raises:
            ValueError: If sample_rate or frame_duration_ms do not match WebRTC native driver prerequisites.
        """

        # Validate constraints required by the underlying C-extension implementation of WebRTC VAD
        if sample_rate not in [8000, 16000, 32000, 48000]: raise ValueError("WebRTC VAD only supports sample rates of 8000, 16000, 32000, or 48000 Hz.")
        if frame_duration_ms not in [10, 20, 30]: raise ValueError("WebRTC VAD only supports frame lengths of 10, 20, or 30 ms.")

        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms

        self.vad = webrtcvad.Vad(sensitivity_mode)
        # Calculate exactly how many samples represent the requested milliseconds window length
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))

    def is_speech(self, audio_chunk):
        """
        Processes a raw input chunk of arbitrary length to detect the presence of human speech.

        Args:
            audio_chunk (np.ndarray): Input floating-point or integer audio array data.

        Returns:
            bool: True if any sub-frame within the chunk contains active human speech, False otherwise.
        """

        # Downmix stereo or multi-channel shapes down to standard single-channel mono sequence loops
        if audio_chunk.ndim > 1 and audio_chunk.shape[1] == 1: audio_chunk = audio_chunk.flatten()
        elif audio_chunk.ndim > 1: audio_chunk = np.mean(audio_chunk, axis=1)

        # Enforce boundary limitations to guard against potential floating-point out-of-bounds clipping
        if np.max(np.abs(audio_chunk)) > 1.0: audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        # Transform normalized -1.0 to 1.0 float data vectors into signed 16-bit Int PCM bytes
        audio_chunk = (audio_chunk * 32767).astype(np.int16)
        num_frames = len(audio_chunk) // self.frame_length

        # Apply zero-padding on tiny partial trailing data blocks to perfectly fill out one full expected slice frame
        if num_frames == 0 and len(audio_chunk) > 0:
            audio_chunk = np.concatenate((
                audio_chunk, 
                np.zeros(
                    self.frame_length - len(audio_chunk), 
                    dtype=np.int16
                )
            ))
            num_frames = 1
        elif num_frames == 0 and len(audio_chunk) == 0: return False

        try:
            # Step through sequential intervals across the byte stream to look for activity markers
            for i in range(num_frames):
                start = i * self.frame_length
                # If the current sub-frame evaluates as speech, flag the entire buffer segment instantly
                if self.vad.is_speech(
                    audio_chunk[start:start + self.frame_length].tobytes(), 
                    self.sample_rate
                ): 
                    return True
            
            return False
        except Exception:
            return False

    def get_speech(self, y, min_speech_duration=0.5):
        """
        Parses an extended mono audio array to extract localized timestamps pointing to valid 
        human conversation phrases.

        Args:
            y (np.ndarray): Complete contiguous floating-point target audio sequence to track.
            min_speech_duration (float): Minimum duration threshold in seconds required to accept a segment.

        Returns:
            list: List of dictionary segments with explicit timing parameters, formatted as: [{'start': x, 'end': y}].
        """

        duration = len(y) / self.sample_rate
        is_speech_list, segments = [], []

        # Iterate over uniform blocks throughout the audio array sequence
        for i in range(0, len(y), self.frame_length):
            chunk = y[i:i + self.frame_length]
            # Pad the final trailing block with zeros if it falls short of the required frame length
            if len(chunk) < self.frame_length: chunk = np.pad(chunk, (0, self.frame_length - len(chunk)))
            # Perform raw evaluation mapping the individual segment slices to binary Boolean arrays
            is_speech_list.append(self.vad.is_speech((chunk * 32767).astype(np.int16).tobytes(), self.sample_rate))

        start_frame = None
        # Calculate trailing silence frame threshold equivalent to a 500 ms window
        silence_threshold_frames = int(500 / self.frame_duration_ms)
        silence_counter = 0

        # Run state-machine checks across the parsed sequence to establish start/end times
        for idx, is_speech in enumerate(is_speech_list):
            if is_speech:
                if start_frame is None: start_frame = idx
                silence_counter = 0 # Reset non-activity indicators since speech is actively registered
            elif start_frame is not None:
                silence_counter += 1
                # If continuous silence duration exceeds 500ms, close out the current phrase segment bounds
                if silence_counter > silence_threshold_frames:
                    start_time = (start_frame * self.frame_duration_ms) / 1000.0
                    end_time = ((idx - silence_counter) * self.frame_duration_ms) / 1000.0

                    # Validate the finalized time delta block against the specified minimum duration limits
                    if end_time - start_time >= min_speech_duration: segments.append({"start": start_time, "end": min(end_time, duration)})

                    start_frame = None
                    silence_counter = 0

        # Handle edge-case boundaries where speech persists up to the end of the input track data buffer
        if start_frame is not None: segments.append({"start": (start_frame * self.frame_duration_ms) / 1000.0, "end": duration})
        return segments