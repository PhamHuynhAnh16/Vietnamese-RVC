import os
import torch
import shutil
import ctypes
import platform

import numpy as np
import tempfile as tf

class DioOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("ChannelsInOctave", ctypes.c_double), ("FramePeriod", ctypes.c_double), ("Speed", ctypes.c_int), ("AllowedRange", ctypes.c_double)]

class HarvestOption(ctypes.Structure):
    _fields_ = [("F0Floor", ctypes.c_double), ("F0Ceil", ctypes.c_double), ("FramePeriod", ctypes.c_double)]

class PYWORLD:
    def __init__(self):
        model = torch.load(os.path.join("assets", "models", "predictors", "world.pth"), map_location="cpu")

        model_type, suffix = (("world_64" if platform.architecture()[0] == "64bit" else "world_86"), ".dll") if platform.system() == "Windows" else ("world_linux", ".so")

        self.temp_folder = os.path.join("assets", "models", "predictors", "temp")
        os.makedirs(self.temp_folder, exist_ok=True)

        with tf.NamedTemporaryFile(delete=False, suffix=suffix, dir=self.temp_folder) as temp_file:
            temp_file.write(model[model_type])
            temp_path = temp_file.name

        self.world_dll = ctypes.CDLL(temp_path)

    def harvest(self, x, fs, f0_floor=50, f0_ceil=1100, frame_period=10):
        self.world_dll.Harvest.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(HarvestOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.world_dll.Harvest.restype = None 

        self.world_dll.InitializeHarvestOption.argtypes = [ctypes.POINTER(HarvestOption)]
        self.world_dll.InitializeHarvestOption.restype = None

        self.world_dll.GetSamplesForHarvest.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.world_dll.GetSamplesForHarvest.restype = ctypes.c_int

        option = HarvestOption()
        self.world_dll.InitializeHarvestOption(ctypes.byref(option))

        option.F0Floor = f0_floor
        option.F0Ceil = f0_ceil
        option.FramePeriod = frame_period

        f0_length = self.world_dll.GetSamplesForHarvest(fs, len(x), option.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()

        self.world_dll.Harvest((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
        return np.array(f0, dtype=np.float32), np.array(tpos, dtype=np.float32)

    def dio(self, x, fs, f0_floor=50, f0_ceil=1100, channels_in_octave=2, frame_period=10, speed=1, allowed_range=0.1):
        self.world_dll.Dio.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(DioOption), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        self.world_dll.Dio.restype = None  

        self.world_dll.InitializeDioOption.argtypes = [ctypes.POINTER(DioOption)]
        self.world_dll.InitializeDioOption.restype = None

        self.world_dll.GetSamplesForDIO.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        self.world_dll.GetSamplesForDIO.restype = ctypes.c_int

        option = DioOption()
        self.world_dll.InitializeDioOption(ctypes.byref(option))

        option.F0Floor = f0_floor
        option.F0Ceil = f0_ceil
        option.ChannelsInOctave = channels_in_octave
        option.FramePeriod = frame_period
        option.Speed = speed
        option.AllowedRange = allowed_range

        f0_length = self.world_dll.GetSamplesForDIO(fs, len(x), option.FramePeriod)
        f0 = (ctypes.c_double * f0_length)()
        tpos = (ctypes.c_double * f0_length)()

        self.world_dll.Dio((ctypes.c_double * len(x))(*x), len(x), fs, ctypes.byref(option), tpos, f0)
        return np.array(f0, dtype=np.float32), np.array(tpos, dtype=np.float32)

    def stonemask(self, x, fs, tpos, f0):
        self.world_dll.StoneMask.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        self.world_dll.StoneMask.restype = None 

        out_f0 = (ctypes.c_double * len(f0))()
        self.world_dll.StoneMask((ctypes.c_double * len(x))(*x), len(x), fs, (ctypes.c_double * len(tpos))(*tpos), (ctypes.c_double * len(f0))(*f0), len(f0), out_f0)

        if os.path.exists(self.temp_folder): shutil.rmtree(self.temp_folder, ignore_errors=True)
        return np.array(out_f0, dtype=np.float32)