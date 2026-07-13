import os
import pickle
import ctypes
import platform

import numpy as np

class DioOption(ctypes.Structure):
    """
    C-struct binding for DIO (Distributed Isolation Optimizer) configuration options.
    Matches the original struct definition in the WORLD vocoder C++ library.
    """

    _fields_ = [
        ("F0Floor", ctypes.c_double), # Minimum F0 search boundary (Hz) 
        ("F0Ceil", ctypes.c_double), # Maximum F0 search boundary (Hz)
        ("ChannelsInOctave", ctypes.c_double), # Number of channels per octave band
        ("FramePeriod", ctypes.c_double), # Window shift period (ms)
        ("Speed", ctypes.c_int), # Downsampling/Speed factor (1 to 12)
        ("AllowedRange", ctypes.c_double) # Allowed deviation parameter for post-processing
    ]

class HarvestOption(ctypes.Structure):
    """
    C-struct binding for Harvest F0 estimation configuration options.
    Matches the original struct definition in the WORLD vocoder C++ library.
    """

    _fields_ = [
        ("F0Floor", ctypes.c_double), # Minimum F0 search boundary (Hz)
        ("F0Ceil", ctypes.c_double), # Maximum F0 search boundary (Hz)
        ("FramePeriod", ctypes.c_double) # Window shift period (ms)
    ]

class PYWORLD:
    """
    Python wrapper for the WORLD speech synthesis vocoder C library using ctypes.
    Handles cross-platform dynamic library extraction, loading, and memory orchestration 
    for Harvest, DIO, and StoneMask F0 processing routines.
    """

    def __init__(self, world_path, model_path, harvest = True):
        """
        Initializes the PYWORLD wrapper, extracts the system-specific binary dependency 
        from a pickled package, and binds default pitch estimation interfaces.

        Parameters:
        -----------
        world_path : str
            Directory path where the extracted dynamic library (.dll/.so/.dylib) will be written.
        model_path : str
            File path to the serialized pickle file containing binary objects of the WORLD library.
        harvest : bool, optional
            If True, assigns the `harvest` function as the default baseline estimator. 
            If False, falls back to `dio` (default: True).
        """

        self.world_path = world_path
        os.makedirs(self.world_path, exist_ok=True)

        # Detect operating system and architecture to identify target binary type
        # 'world_android64', 'world_android86', 'world_arm_eabi', 'world_arm64', 'world_linux', 'world_mac', 'world_64', 'world_86'
        if platform.system() == "Windows":
            model_type, suffix = (
                ("world_64" if platform.architecture()[0] == "64bit" else "world_86"), ".dll"
            ) 
        elif platform.system() == "Linux":
            model_type, suffix = (
                "world_linux", ".so"
            )
        elif platform.system() == "Darwin":
            model_type, suffix = (
                "world_mac", ".dylib"
            )
        else:
            raise ValueError(f"Unsupported operating system architecture: {platform.system()}")

        self.world_file_path = os.path.join(self.world_path, f"{model_type}{suffix}")
        # Extract compiled binary file from pickle payload if it does not exist locally
        if not os.path.exists(self.world_file_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            with open(self.world_file_path, "wb") as w:
                w.write(model[model_type])

        self.idx_dio = 0
        self.idx_harvest = 0
        self.idx_stonemask = 0
        self.option_dio = None
        self.option_harvest = None
        # Load the dynamic library via ctypes
        self.world_dll = ctypes.CDLL(self.world_file_path)
        # Bind the primary inference entry point alias
        self.infer = self.harvest if harvest else self.dio

    def harvest(
        self, 
        x, 
        fs, 
        f0_floor=50, 
        f0_ceil=1100, 
        frame_period=10
    ):
        """
        Estimates the fundamental frequency (F0) contour using the Harvest algorithm.
        Harvest offers high-accuracy pitch contour tracking via fundamental component candidates.

        Parameters:
        -----------
        x : numpy.ndarray or list
            Input 1D waveform data array.
        fs : int
            Sampling frequency of the audio (Hz).
        f0_floor : float, optional
            Lower bound constraint for pitch detection (default: 50).
        f0_ceil : float, optional
            Upper bound constraint for pitch detection (default: 1100).
        frame_period : float, optional
            Time interval window step between frames in milliseconds (default: 10).

        Returns:
        --------
        f0 : numpy.ndarray
            Estimated fundamental frequency track (Hz) per frame.
        tpos : numpy.ndarray
            Time markers (seconds) corresponding to each frame center.
        """

        if self.idx_harvest == 0:
            # Configure ctypes signature profiles for underlying C functions
            self.world_dll.Harvest.argtypes = [
                ctypes.POINTER(ctypes.c_double), # input waveform array
                ctypes.c_int, # x length
                ctypes.c_int, # sampling frequency
                ctypes.POINTER(HarvestOption), # algorithm configuration parameters
                ctypes.POINTER(ctypes.c_double), # output time positions array
                ctypes.POINTER(ctypes.c_double) # output f0 frequency array
            ]

            self.world_dll.Harvest.restype = None 
            self.world_dll.InitializeHarvestOption.argtypes = [ctypes.POINTER(HarvestOption)]
            self.world_dll.InitializeHarvestOption.restype = None
            self.world_dll.GetSamplesForHarvest.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
            self.world_dll.GetSamplesForHarvest.restype = ctypes.c_int
            # Instantiate option structure and apply defaults from the C source template
            self.option_harvest = HarvestOption()
            self.world_dll.InitializeHarvestOption(ctypes.byref(self.option_harvest))

            # Override defaults with runtime configurations
            self.option_harvest.F0Floor = f0_floor
            self.option_harvest.F0Ceil = f0_ceil
            self.option_harvest.FramePeriod = frame_period
            self.idx_harvest += 1

        # Determine target buffer size allocations needed for output arrays
        f0_length = self.world_dll.GetSamplesForHarvest(fs, len(x), self.option_harvest.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()

        # Execute C backend processing (casting input to flat continuous double precision array)
        self.world_dll.Harvest((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(self.option_harvest), tpos, f0)
        return np.array(f0, dtype=np.float32), np.array(tpos, dtype=np.float32)

    def dio(
        self, 
        x, 
        fs, 
        f0_floor=50, 
        f0_ceil=1100, 
        channels_in_octave=2, 
        frame_period=10, 
        speed=1, 
        allowed_range=0.1
    ):
        """
        Estimates the fundamental frequency (F0) contour using the DIO algorithm.
        DIO is computationally faster and utilizes multiple band-pass filters.

        Parameters:
        -----------
        x : numpy.ndarray or list
            Input 1D waveform data array.
        fs : int
            Sampling frequency of the audio (Hz).
        f0_floor : float, optional
            Lower bound constraint for pitch detection (default: 50).
        f0_ceil : float, optional
            Upper bound constraint for pitch detection (default: 1100).
        channels_in_octave : float, optional
            Resolution of filter banks distributed per octave (default: 2).
        frame_period : float, optional
            Time interval window step between frames in milliseconds (default: 10).
        speed : int, optional
            Downsampling execution factor acceleration parameter [1, 12] (default: 1).
        allowed_range : float, optional
            Threshold parameter controlling voice tracking deviations (default: 0.1).

        Returns:
        --------
        f0 : numpy.ndarray
            Estimated baseline fundamental frequency track (Hz).
        tpos : numpy.ndarray
            Time markers (seconds) corresponding to each frame center.
        """

        if self.idx_dio == 0:
            # Configure ctypes signature profiles for underlying DIO functions
            self.world_dll.Dio.argtypes = [
                ctypes.POINTER(ctypes.c_double), 
                ctypes.c_int, 
                ctypes.c_int, 
                ctypes.POINTER(DioOption), 
                ctypes.POINTER(ctypes.c_double), 
                ctypes.POINTER(ctypes.c_double)
            ]

            self.world_dll.Dio.restype = None  
            self.world_dll.InitializeDioOption.argtypes = [ctypes.POINTER(DioOption)]
            self.world_dll.InitializeDioOption.restype = None
            self.world_dll.GetSamplesForDIO.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
            self.world_dll.GetSamplesForDIO.restype = ctypes.c_int

            # Instantiate structure and pull internal C library initialization variables
            self.option_dio = DioOption()
            self.world_dll.InitializeDioOption(ctypes.byref(self.option_dio))
            # Assign user parameters to struct memory layout
            self.option_dio.F0Floor = f0_floor
            self.option_dio.F0Ceil = f0_ceil
            self.option_dio.ChannelsInOctave = channels_in_octave
            self.option_dio.FramePeriod = frame_period
            self.option_dio.Speed = speed
            self.option_dio.AllowedRange = allowed_range
            self.idx_dio += 1

        # Allocate double array storage matching required output structures
        f0_length = self.world_dll.GetSamplesForDIO(fs, len(x), self.option_dio.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()

        # Execute baseline execution sequence
        self.world_dll.Dio((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(self.option_dio), tpos, f0)
        return np.array(f0, dtype=np.float32), np.array(tpos, dtype=np.float32)

    def stonemask(
        self, 
        x, 
        fs, 
        tpos, 
        f0
    ):
        """
        Refines a coarse F0 trajectory estimate using the StoneMask refinement protocol.
        Typically executed as a corrective refinement pass over raw DIO trajectories.

        Parameters:
        -----------
        x : numpy.ndarray or list
            Original input audio timeline sequence.
        fs : int
            Sampling frequency of the audio (Hz).
        tpos : numpy.ndarray or list
            Target analysis frame timestamp index markers (retrieved from DIO/Harvest).
        f0 : numpy.ndarray or list
            Coarse initial F0 frequency tracking array values to polish.

        Returns:
        --------
        out_f0 : numpy.ndarray
            Refined high-precision F0 estimation trajectory map (Hz).
        """

        if self.idx_stonemask == 0:
            # Define function parameters mapping onto the exact C source definition
            self.world_dll.StoneMask.argtypes = [
                ctypes.POINTER(ctypes.c_double), # input waveform reference
                ctypes.c_int, # x total length
                ctypes.c_int, # sampling frequency
                ctypes.POINTER(ctypes.c_double), # timestamps tracking grid list
                ctypes.POINTER(ctypes.c_double), # target initial coarse f0 values array
                ctypes.c_int, # length of f0 vector elements
                ctypes.POINTER(ctypes.c_double) # output array pointer for corrected f0 data
            ]

            self.world_dll.StoneMask.restype = None 
            self.idx_stonemask += 1

        # Allocate memory space for receiving refined data contents
        out_f0 = (ctypes.c_double * len(f0))()
        # Cast elements and invoke the dynamic modification pipeline layer
        self.world_dll.StoneMask(
            (ctypes.c_double * len(x))(*x), 
            len(x), 
            fs, 
            (ctypes.c_double * len(tpos))(*tpos), 
            (ctypes.c_double * len(f0))(*f0), 
            len(f0), 
            out_f0
        )

        return np.array(out_f0, dtype=np.float32)