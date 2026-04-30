import webrtcvad

import numpy as np

class VADProcessor:
    def __init__(
        self, 
        sensitivity_mode=3, 
        sample_rate=16000, 
        frame_duration_ms=30
    ):
        if sample_rate not in [8000, 16000, 32000, 48000]: raise ValueError
        if frame_duration_ms not in [10, 20, 30]: raise ValueError

        self.vad = webrtcvad.Vad(sensitivity_mode)
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * (frame_duration_ms / 1000.0))
        self.frame_duration_ms = frame_duration_ms

    def is_speech(self, audio_chunk):
        if audio_chunk.ndim > 1 and audio_chunk.shape[1] == 1: audio_chunk = audio_chunk.flatten()
        elif audio_chunk.ndim > 1: audio_chunk = np.mean(audio_chunk, axis=1)

        if np.max(np.abs(audio_chunk)) > 1.0: audio_chunk = np.clip(audio_chunk, -1.0, 1.0)

        audio_chunk = (audio_chunk * 32767).astype(np.int16)
        num_frames = len(audio_chunk) // self.frame_length

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
            for i in range(num_frames):
                start = i * self.frame_length

                if self.vad.is_speech(
                    audio_chunk[start:start + self.frame_length].tobytes(), 
                    self.sample_rate
                ): 
                    return True
            
            return False
        except Exception:
            return False

    def get_speech(self, y, min_speech_duration=0.5):
        duration = len(y) / self.sample_rate
        is_speech_list, segments = [], []

        for i in range(0, len(y), self.frame_length):
            chunk = y[i:i + self.frame_length]
            if len(chunk) < self.frame_length: chunk = np.pad(chunk, (0, self.frame_length - len(chunk)))
            is_speech_list.append(self.vad.is_speech((chunk * 32767).astype(np.int16).tobytes(), self.sample_rate))

        start_frame = None
        silence_threshold_frames = int(500 / self.frame_duration_ms)
        silence_counter = 0

        for idx, is_speech in enumerate(is_speech_list):
            if is_speech:
                if start_frame is None: start_frame = idx
                silence_counter = 0
            elif start_frame is not None:
                silence_counter += 1

                if silence_counter > silence_threshold_frames:
                    start_time = (start_frame * self.frame_duration_ms) / 1000.0
                    end_time = ((idx - silence_counter) * self.frame_duration_ms) / 1000.0

                    if end_time - start_time >= min_speech_duration: segments.append({"start": start_time, "end": min(end_time, duration)})

                    start_frame = None
                    silence_counter = 0

        if start_frame is not None: segments.append({"start": (start_frame * self.frame_duration_ms) / 1000.0, "end": duration})
        return segments