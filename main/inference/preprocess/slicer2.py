import numpy as np

class Slicer:
    """
    Slices audio waveforms based on Root Mean Square (RMS) energy thresholds to separate 
    voiced segments from silences.
    """

    def __init__(
        self, 
        sr, 
        threshold = -40.0, 
        min_length = 5000, 
        min_interval = 300, 
        hop_size = 20, 
        max_sil_kept = 5000
    ):
        """
        Initializes the Slicer with explicit temporal and amplitude configurations.

        Args:
            sr (int): Sampling rate of the audio signals.
            threshold (float): Threshold in dB below which audio is treated as silence. Defaults to -40.0.
            min_length (int): Minimum length of a valid audio chunk in milliseconds. Defaults to 5000.
            min_interval (int): Minimum duration of silence needed to trigger a cut in milliseconds. Defaults to 300.
            hop_size (int): Step size for frame feature extraction in milliseconds. Defaults to 20.
            max_sil_kept (int): Maximum silence buffer retained at chunk boundaries in milliseconds. Defaults to 5000.
        """

        # Convert absolute milliseconds to sample counts and frame numbers
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.0) # Convert dB to linear amplitude scale
        self.hop_size = round(sr * hop_size / 1000)
        # Window size is bounded by the minimum requested interval or 4x the hop size to preserve local context
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        # Scale time parameters into unit values matching the frame tracking array (divided by hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        """
        Extracts a segment from the waveform mapping frame boundaries back to time-domain sample indexes.

        Args:
            waveform (np.ndarray): Target audio array of shape (samples,) or (channels, samples).
            begin (int): Starting frame index.
            end (int): Terminating frame index.

        Returns:
            np.ndarray: Sliced subsection of the original audio waveform.
        """

        start_idx = begin * self.hop_size
        # Check dimensionality to handle mono arrays vs multichannel matrices correctly
        return waveform[
            :, start_idx:min(waveform.shape[1], end * self.hop_size)
        ] if len(waveform.shape) > 1 else waveform[
            start_idx:min(waveform.shape[0], end * self.hop_size)
        ]

    def slice(self, waveform):
        """
        Slices a waveform into a collection of variable-length audio chunks without returning indices.

        Args:
            waveform (np.ndarray): Audio track array profile to parse.

        Returns:
            List[np.ndarray]: List of split non-silent audio array fragments.
        """

        # Downmix audio to single channel mean representations if multidimensional input is given
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform
        # Immediately return the complete original file if its frame depth falls short of minimum split thresholds
        if samples.shape[0] <= self.min_length: return [waveform]

        # Extract continuous frame root-mean-square amplitude envelopes
        rms_list = get_rms(
            y=samples, 
            frame_length=self.win_size, 
            hop_length=self.hop_size
        ).squeeze(0)

        sil_tags = []
        silence_start, clip_start = None, 0
        # Scan across RMS frame elements to track noise gates
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None: silence_start = i
                continue

            if silence_start is None: continue
            # Identify edge cases such as long trailing silence buffers or leading gaps
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept

            need_slice_middle = (
                i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept: # Case A: Silence sequence fits nicely inside the max retained padding limits
                pos = rms_list[silence_start : i + 1].argmin() + silence_start

                sil_tags.append(
                    (0, pos) if silence_start == 0 else (pos, pos)
                )   

                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2: # Case B: Silence sequence extends past safety margins but stays below double padding limits
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()

                pos += i - self.max_sil_kept
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((
                        min((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos), 
                        max(pos_r, pos)
                    ))

                    clip_start = max(pos_r, pos)
            else: # Case C: Extended deep silence block spanning beyond standard tracking capacities
                pos_r = (rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept)

                sil_tags.append(
                    (
                        0, 
                        pos_r
                    ) if silence_start == 0 else (
                        (rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), 
                        pos_r
                    )
                )

                clip_start = pos_r

            silence_start = None

        total_frames = rms_list.shape[0]
        # Handle hanging fragments residing at the end of tracking timelines
        if (
            silence_start is not None and 
            total_frames - silence_start >= self.min_interval
        ): 
            sil_tags.append((
                rms_list[silence_start : min(total_frames, silence_start + self.max_sil_kept) + 1].argmin() + silence_start, 
                total_frames + 1
            ))

        # Return original track wrapper if zero slicing configurations matched
        if not sil_tags: return [waveform]
        else:
            # Assemble extracted audio snippets based on tag timestamps
            chunks = []
            if sil_tags[0][0] > 0: chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

            for i in range(len(sil_tags) - 1):
                chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))

            if sil_tags[-1][1] < total_frames: chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))
            return chunks

class Slicer2(Slicer):
    """
    An extended implementation of Slicer that returns both the audio chunks 
    and their exact sample boundaries (start and end indexes).
    """

    def slice2(self, waveform):
        """
        Slices a waveform into chunks, returning each segment along with its sample timestamps.

        Args:
            waveform (np.ndarray): Audio track array profile to parse.

        Returns:
            List[Tuple[np.ndarray, int, int]]: List of tuples containing (sliced_waveform, start_sample, end_sample).
        """

        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform

        if samples.shape[0] <= self.min_length: 
            return [(
                waveform, 
                0, 
                samples.shape[0]
            )]

        rms_list = get_rms(
            y=samples, 
            frame_length=self.win_size, 
            hop_length=self.hop_size
        ).squeeze(0)

        sil_tags = []
        silence_start, clip_start = None, 0
        # Scan across RMS frame elements to track noise gates
        for i, rms in enumerate(rms_list):
            if rms < self.threshold:
                if silence_start is None: silence_start = i
                continue

            if silence_start is None: continue

            is_leading_silence = silence_start == 0 and i > self.max_sil_kept

            need_slice_middle = (
                i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            )

            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                sil_tags.append((0, pos) if silence_start == 0 else (pos, pos))   

                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[
                    i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                ].argmin()

                pos += i - self.max_sil_kept
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept

                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((
                        min((rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), pos), 
                        max(pos_r, pos)
                    ))

                    clip_start = max(pos_r, pos)
            else:
                pos_r = rms_list[i - self.max_sil_kept : i + 1].argmin() + i - self.max_sil_kept  

                sil_tags.append(
                    (
                        0, 
                        pos_r
                    ) if silence_start == 0 else (
                        (rms_list[silence_start : silence_start + self.max_sil_kept + 1].argmin() + silence_start), 
                        pos_r
                    )
                )

                clip_start = pos_r

            silence_start = None

        total_frames = rms_list.shape[0]
        if (silence_start is not None and total_frames - silence_start >= self.min_interval): 
            sil_tags.append((
                rms_list[silence_start : min(total_frames, silence_start + self.max_sil_kept) + 1].argmin() + silence_start, 
                total_frames + 1
            ))

        if not sil_tags: 
            return [(
                waveform, 
                0, 
                samples.shape[-1]
            )]
        else:
            chunks = []
            # Package snippets with absolute sample-index indicators (frame index multiplied by hop_size)
            if sil_tags[0][0] > 0: 
                chunks.append((
                    self._apply_slice(waveform, 0, sil_tags[0][0]), 
                    0, 
                    sil_tags[0][0] * self.hop_size
                ))

            for i in range(len(sil_tags) - 1):
                chunks.append((
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]), 
                    sil_tags[i][1] * self.hop_size, 
                    sil_tags[i + 1][0] * self.hop_size
                ))

            if sil_tags[-1][1] < total_frames: 
                chunks.append((
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames), 
                    sil_tags[-1][1] * self.hop_size, 
                    samples.shape[-1]
                ))

            return chunks
            
def get_rms(y, frame_length=2048, hop_length=512, pad_mode="constant"):
    """
    Calculates Root Mean Square (RMS) energy envelopes using high-speed NumPy sliding strides.

    Args:
        y (np.ndarray): 1D single-channel floating-point audio data.
        frame_length (int): Length of each window frame loop. Defaults to 2048.
        hop_length (int): Distance in samples between adjacent windows. Defaults to 512.
        pad_mode (str): Padding strategy for array margin safety. Defaults to "constant".

    Returns:
        np.ndarray: Computed RMS energy contours array.
    """

    # Symmetric margin padding guarantees tracking snapshots remain centered across target metrics
    y = np.pad(y, (int(frame_length // 2), int(frame_length // 2)), mode=pad_mode)
    axis = -1

    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    # Extract high-efficiency internal pointers via sliding strides to prevent large data re-allocations
    xw = np.moveaxis(
        np.lib.stride_tricks.as_strided(
            y, 
            shape=tuple(x_shape_trimmed) + tuple([frame_length]), 
            strides=y.strides + tuple([y.strides[axis]])
        ), 
        -1, 
        axis - 1 if axis < 0 else axis + 1
    )
    
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    # Square values, compute local mean cross-sections, and extract root profiles
    return np.sqrt(
        np.mean(
            np.abs(xw[tuple(slices)]) ** 2, 
            axis=-2, 
            keepdims=True
        )
    )