import numpy as np

from numpy.lib.stride_tricks import sliding_window_view

def istft(frames, framesize, hopsize):
    """
    Inverse Short-Time Fourier Transform (ISTFT) with overlap-add method.

    Parameters:
        frames (ndarray): The STFT frames of shape (num_frames, num_bins).
        framesize (int or list/tuple): The size of the analysis and synthesis windows.
        hopsize (int): The number of samples between successive frames.

    Returns:
        ndarray: The reconstructed 1D time-domain signal.
    """

    # Ensure the input frames are a 2D numpy array
    frames = np.atleast_2d(frames)
    assert frames.ndim == 2

    # Extract analysis and synthesis window sizes (handles single integer or array-like inputs)
    analysis_window_size = np.ravel(framesize)[0]
    synthesis_window_size = np.ravel(framesize)[-1]
    # Analysis window must be larger than or equal to synthesis window size
    assert analysis_window_size >= synthesis_window_size

    # Select appropriate windows based on whether they are symmetric or asymmetric
    A = asymmetric_analysis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(analysis_window_size)
    S = asymmetric_synthesis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(synthesis_window_size)
    # Compute synthesis window scale factor for amplitude normalization
    W = S * hopsize / np.sum(A * S)
    # Calculate the total length of the target output signal
    N = frames.shape[0] * hopsize + analysis_window_size

    # Initialize the output time-domain signal with zeros
    y = np.zeros((N), float)

    # Zero out the first and last frequency bins (often required for constraints)
    frames[:,  0] = 0
    frames[:, -1] = 0
    # Create a writable sliding window view overlapping the output array
    frames0 = sliding_window_view(y, analysis_window_size, writeable=True)[::hopsize]
    # Perform Inverse Real FFT and scale with the window factor
    frames1 = np.fft.irfft(frames, axis=-1, norm='forward') * W

    # Perform overlap-add reconstruction
    for i in range(min(len(frames0), len(frames1))):
        frames0[i] += frames1[i]

    return y

def asymmetric_synthesis_window(analysis_window_size, synthesis_window_size):
    """
    Generates an asymmetric synthesis window based on specified window sizes.

    Parameters:
        analysis_window_size (int): Size of the analysis window.
        synthesis_window_size (int): Size of the synthesis window.

    Returns:
        ndarray: The generated asymmetric synthesis window.
    """

    n = analysis_window_size
    m = synthesis_window_size // 2

    # Get a base symmetric window for the right half
    right = symmetric_window(2 * m)
    window = np.zeros(n)
    # Construct the asymmetric parts using squared symmetric windows normalized by overlapping segments
    window[n-m-m:n-m] = np.square(right[:m]) / symmetric_window(2 * n - 2 * m)[n-m-m:n-m]
    window[-m:] = right[-m:]

    return window

def asymmetric_analysis_window(analysis_window_size, synthesis_window_size):
    """
    Generates an asymmetric analysis window based on specified window sizes.

    Parameters:
        analysis_window_size (int): Size of the analysis window.
        synthesis_window_size (int): Size of the synthesis window.

    Returns:
        ndarray: The generated asymmetric analysis window.
    """

    n = analysis_window_size
    m = synthesis_window_size // 2

    window = np.zeros(n)
    # Merge portions of symmetric windows of different lengths
    window[:n-m] = symmetric_window(2 * n - 2 * m)[:n-m]
    window[-m:] = symmetric_window(2 * m)[-m:]

    return window

def symmetric_window(symmetric_window_size):
    """
    Generates a standard symmetric Hann window.

    Parameters:
        symmetric_window_size (int): Total size of the window.

    Returns:
        ndarray: The symmetric Hann window.
    """
    n = symmetric_window_size
    # Compute the Hann window mathematical equation
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)

    return window

def stft(x, framesize, hopsize):
    """
    Short-Time Fourier Transform (STFT) using a sliding window view.

    Parameters:
        x (ndarray): The 1D input time-domain signal.
        framesize (int or list/tuple): The size of the analysis and synthesis windows.
        hopsize (int): The number of samples between successive frames.

    Returns:
        ndarray: The complex STFT spectrogram of shape (num_frames, num_bins).
    """

    # Ensure input signal is 1D array
    x = np.atleast_1d(x)
    assert x.ndim == 1

    # Extract window sizes
    analysis_window_size = np.ravel(framesize)[0]
    synthesis_window_size = np.ravel(framesize)[-1]

    assert analysis_window_size >= synthesis_window_size
    # Choose analysis window function
    W = asymmetric_analysis_window(analysis_window_size, synthesis_window_size) if analysis_window_size != synthesis_window_size else symmetric_window(analysis_window_size)

    # Create overlapping temporal frames from the input signal
    frames0 = sliding_window_view(x, analysis_window_size, writeable=False)[::hopsize]
    # Multiply by the window function and compute Real FFT along the last axis
    frames1 = np.fft.rfft(frames0 * W, axis=-1, norm='forward')

    return frames1

def normalize(frames, frames0):
    """
    Normalizes the energy of frames to match the original frames0 energy.

    Parameters:
        frames (ndarray): Modified frames.
        frames0 (ndarray): Original reference frames.

    Returns:
        ndarray: Energy-normalized frames.
    """

    for i in range(len(frames)):
        a = np.real(frames0[i])
        b = np.real(frames[i])
        # Compute dot product (total energy/squared norm)
        a = np.dot(a, a)
        b = np.dot(b, b)

        # Avoid division by zero if target frame energy is 0
        if b == 0: continue
        # Normalize real components based on energy ratio while preserving imaginary components
        frames[i] = np.real(frames[i]) * np.sqrt(a / b) + 1j * np.imag(frames[i])

    return frames

def lowpass(cepstrum, quefrency):
    """
    Applies a basic lowpass filter on a liftered/cepstral sequence.

    Parameters:
        cepstrum (ndarray): The input cepstrum sequence.
        quefrency (int): The cutoff quefrency index.

    Returns:
        ndarray: The filtered cepstrum.
    """

    # Boost lower quefrencies and discard frequencies above the cutoff
    cepstrum[1:quefrency] *= 2
    cepstrum[quefrency+1:] = 0

    return cepstrum

def lifter(frames, quefrency):
    """
    Extracts the spectral envelope using liftering (cepstral smoothing).

    Parameters:
        frames (ndarray): Input complex spectrogram frames.
        quefrency (int): Quefrency cutoff for spectral smoothing.

    Returns:
        ndarray: Spectral envelopes matching the shape of input frames.
    """

    envelopes = np.zeros(frames.shape)

    for i, frame in enumerate(frames):
        # Mute divide/invalid warnings when computing log10 of 0 or negative real components
        with np.errstate(divide='ignore', invalid='ignore'):
            spectrum = np.log10(np.real(frame)) 

        # Transform to cepstral domain (IFFT), apply lowpass filter, transform back (FFT), and undo log
        envelopes[i] = np.power(10, np.real(np.fft.rfft(lowpass(np.fft.irfft(spectrum, norm='forward'), quefrency), norm='forward')))

    return envelopes

def resample(x, factor):
    """
    Resamples a 1D array using basic linear interpolation.

    Parameters:
        x (ndarray): 1D input array.
        factor (float): Resampling scale factor.

    Returns:
        ndarray: Resampled array.
    """

    if factor == 1: return x.copy()
    y = np.zeros(x.shape, dtype=x.dtype)
    
    n = len(x)
    m = int(n * factor)
    # Compute new index mappings
    i = np.arange(min(n, m))
    k = i * (n / m)
    # Truncate to find lower index bounding
    j = np.trunc(k).astype(int)
    k = k - j
    # Keep indices strictly inside bounds
    ok = (0 <= j) & (j < n - 1)
    # Perform linear interpolation
    y[i[ok]] = k[ok] * x[j[ok] + 1] + (1 - k[ok]) * x[j[ok]]

    return y

def shiftpitch(frames, factors, samplerate):
    """
    Shifts the pitch of encoded phase-vocoder frames.

    Parameters:
        frames (ndarray): Encoded phase-vocoder frames (Magnitudes + 1j * Frequencies).
        factors (ndarray/list): Pitch scaling factors.
        samplerate (int): The sample rate of the audio.

    Returns:
        ndarray: Pitch-shifted frames.
    """

    for i in range(len(frames)):
        # Resample magnitude spectrum according to factors
        magnitudes = np.vstack([resample(np.real(frames[i]), factor) for factor in factors])
        # Resample and scale instantaneous frequencies
        frequencies = np.vstack([resample(np.imag(frames[i]), factor) * factor for factor in factors])
        # Zero out frequencies that fall out of the valid Nyquist range
        magnitudes[(frequencies <= 0) | (frequencies >= samplerate / 2)] = 0
        # Determine peak dominant tracking pitch indices
        mask = np.argmax(magnitudes, axis=0)
        # Pick the values matching peak tracking
        magnitudes = np.take_along_axis(magnitudes, mask[None,:], axis=0)
        frequencies = np.take_along_axis(frequencies, mask[None,:], axis=0)
        # Recombine into magnitude and frequency pairs
        frames[i] = magnitudes + 1j * frequencies

    return frames

def wrap(x):
    """
    Wraps phase angles to the range [-pi, pi].

    Parameters:
        x (ndarray): Phase angles in radians.

    Returns:
        ndarray: Wrapped phase angles.
    """

    return (x + np.pi) % (2 * np.pi) - np.pi

def encode(frames, framesize, hopsize, samplerate):
    """
    Encodes STFT spectrogram frames into magnitude and instantaneous frequency pairs.

    Parameters:
        frames (ndarray): Complex spectrogram frames.
        framesize (int or list/tuple): The size of the analysis window.
        hopsize (int): Hop size in samples.
        samplerate (int): Sample rate in Hz.

    Returns:
        ndarray: Encoded phase-vocoder frames (Magnitude + 1j * Instantaneous Frequency).
    """

    M, N = frames.shape
    analysis_framesize = np.ravel(framesize)[0]

    # Calculate frequency bin step and theoretical phase advancement step
    freqinc = samplerate / analysis_framesize
    phaseinc = 2 * np.pi * hopsize / analysis_framesize

    buffer = np.zeros(N)
    data = np.zeros((M, N), complex)

    for m, frame in enumerate(frames):
        arg = np.angle(frame)
        delta = arg - buffer # Phase difference between successive frames
        buffer = arg # Store current phase for the next frame iteration

        i = np.arange(N)
        # Convert phase deviations to absolute instantaneous frequencies
        data[m] = np.abs(frame) + 1j * ((i + (wrap(delta - i * phaseinc) / phaseinc)) * freqinc)

    return data

def decode(frames, framesize, hopsize, samplerate):
    """
    Decodes magnitude and instantaneous frequency pairs back into complex STFT frames.

    Parameters:
        frames (ndarray): Encoded frames (Magnitude + 1j * Instantaneous Frequency).
        framesize (int or list/tuple): Windows size settings.
        hopsize (int): Hop size in samples.
        samplerate (int): Sample rate in Hz.

    Returns:
        ndarray: Decoded complex STFT frames.
    """

    M, N = frames.shape
    analysis_framesize = np.ravel(framesize)[0]
    synthesis_framesize = np.ravel(framesize)[-1]
    # Compute frequency bin step and phase advancement step
    freqinc = samplerate / analysis_framesize
    phaseinc = 2 * np.pi * hopsize / analysis_framesize
    # Set time shift compensation factor if analysis and synthesis windows differ
    timeshift = 2 * np.pi * synthesis_framesize * np.arange(N) / N if synthesis_framesize != analysis_framesize else 0

    buffer = np.zeros(N)
    data = np.zeros((M, N), complex)

    for m, frame in enumerate(frames):
        i = np.arange(N)
        # Extract phase difference from instantaneous frequencies
        delta = (i + ((np.imag(frame) - i * freqinc) / freqinc)) * phaseinc
        buffer += delta # Accumulate phase over frames
        arg = buffer.copy()
        arg -= timeshift 
        # Reconstruct complex values from magnitudes and accumulated phases
        data[m] = np.real(frame) * np.exp(1j * arg)

    return data

class StftPitchShift:
    """
    A class wrapper to orchestrate high-quality pitch-shifting using STFT,
    phase-vocoder encoding/decoding, and formant preservation via liftering.
    """

    def __init__(self, framesize, hopsize, samplerate):
        """
        Initializes the StftPitchShift pipeline context.

        Parameters:
            framesize (int or list/tuple): Resolution/size configuration of audio windowing.
            hopsize (int): Hop interval step between sliding segments.
            samplerate (int): Processing sampling rate indicator.
        """

        self.framesize = framesize
        self.hopsize = hopsize
        self.samplerate = samplerate

    def shiftpitch(self, input, factors = 1, quefrency = 0, distortion = 1, normalization = False):
        """
        Applies pitch-shifting on a given 1D audio signal array.

        Parameters:
            input (ndarray): Input time-domain signal.
            factors (float or list/ndarray): Pitch scaling ratios (e.g., 2.0 shifts up one octave).
            quefrency (float): Formant preservation threshold coefficient (in seconds).
            distortion (float): Spectral envelope/formant stretching distortion factor.
            normalization (bool): True to enable energy normalization after processing.

        Returns:
            ndarray: The pitch-shifted time-domain audio signal.
        """

        # Ensure array properties match processing structure
        input = np.atleast_1d(input)
        dtype = input.dtype
        shape = input.shape

        input = np.squeeze(input)
        if input.ndim != 1: raise ValueError(f'Invalid input shape {shape}, expected a one-dimensional array!')

        # If data is in integer form, normalize range scale to floats [-1.0, 1.0]
        if np.issubdtype(dtype, np.integer):
            a, b = np.iinfo(dtype).min, np.iinfo(dtype).max

            input = ((input.astype(float) - a) / (b - a)) * 2 - 1
        elif not np.issubdtype(dtype, np.floating): raise TypeError(f'Invalid input data type {dtype}, expected {np.floating} or {np.integer}!')

        # Inline utility helper to flag bad matrix numbers (Inf, NaN, or underflow subnormals)
        def isnotnormal(x):
            return (np.isinf(x)) | (np.isnan(x)) | (abs(x) < np.finfo(x.dtype).tiny)

        framesize = self.framesize
        hopsize = self.hopsize
        samplerate = self.samplerate
        # Format parsing parameters
        factors = np.asarray(factors).flatten()
        quefrency = int(quefrency * samplerate)
        # Step 1: Compute STFT and encode phases into magnitudes and frequencies
        frames = encode(
            stft(
                input, 
                framesize, 
                hopsize
            ), 
            framesize, 
            hopsize, 
            samplerate
        )
        
        # Cache reference frames if normalization is requested
        if normalization: frames0 = frames.copy()

        # Step 2: Extract, modify, and process formants if a quefrency is set
        if quefrency:
            envelopes = lifter(frames, quefrency)
            mask = isnotnormal(envelopes)
            # Flatten/whiten the spectrum magnitude by stripping the spectral envelope
            frames.real /= envelopes
            frames.real[mask] = 0

            if distortion != 1: # Distort formants separately if distortion factor is requested
                envelopes[mask] = 0

                for i in range(len(envelopes)):
                    envelopes[i] = resample(envelopes[i], distortion)

                mask = isnotnormal(envelopes)

            # Perform the core pitch shifting process
            frames = shiftpitch(
                frames, 
                factors, 
                samplerate
            )
            # Re-apply the spectral envelope to restore formants
            frames.real *= envelopes
            frames.real[mask] = 0
        else: 
            # Perform basic pitch shifting without formant tracking
            frames = shiftpitch(frames, factors, samplerate) 

        # Step 3: Run optional frame energy restoration normalization
        if normalization: 
            frames = normalize(
                frames, 
                frames0
            )

        # Step 4: Decode tracking phases and reconstruct time domain signal using ISTFT
        output = istft(decode(frames, framesize, hopsize, samplerate), framesize, hopsize)
        output.resize(shape, refcheck=False)

        # Step 5: Convert back to original data type if input was integer-based
        if np.issubdtype(dtype, np.integer):
            a, b = np.iinfo(dtype).min, np.iinfo(dtype).max

            output = (((output + 1) / 2) * (b - a) + a).clip(a, b).astype(dtype)
        elif output.dtype != dtype: output = output.astype(dtype)

        # Final structural validity assurance safety assertions
        assert output.dtype == dtype
        assert output.shape == shape

        return output