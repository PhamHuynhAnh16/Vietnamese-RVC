import os
import sys
import pickle
import locale
import atexit
import platform
import threading
import contextlib
import _cffi_backend

import numpy as np

from numbers import Integral
from ctypes.util import find_library

sys.path.append(os.getcwd())

from main.app.variables import configs, translations

portaudiolib = None

# Initialize FFI for PortAudio binding
ffi = _cffi_backend.FFI('portaudio',
    _version = 0x2601,
    _types = b'\x00\x00\x76\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x79\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x1C\x0D\x00\x00\x8D\x03\x00\x00\x00\x0F\x00\x00\x7B\x0D\x00\x00\x00\x0F\x00\x00\x80\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x01\x01\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x00\x0F\x00\x00\x21\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x01\x0B\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x82\x03\x00\x00\x1F\x11\x00\x00\x0E\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x03\x00\x00\x1F\x11\x00\x00\x1F\x11\x00\x00\x0E\x01\x00\x00\x0A\x01\x00\x00\x0A\x01\x00\x00\x52\x03\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x2E\x11\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x0A\x01\x00\x00\x0E\x01\x00\x00\x0A\x01\x00\x00\x34\x11\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x07\x11\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x8D\x03\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x6B\x03\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x4B\x11\x00\x00\x07\x11\x00\x00\x0A\x01\x00\x00\x7F\x03\x00\x00\x0A\x01\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x00\x0F\x00\x00\x69\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x7D\x03\x00\x00\x8B\x03\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x60\x11\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x09\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x07\x11\x00\x00\x09\x01\x00\x00\x07\x11\x00\x00\x09\x01\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x09\x00\x00\x77\x03\x00\x00\x02\x09\x00\x00\x00\x0B\x00\x00\x7A\x03\x00\x00\x03\x09\x00\x00\x7C\x03\x00\x00\x04\x09\x00\x00\x00\x09\x00\x00\x02\x0B\x00\x00\x05\x09\x00\x00\x81\x03\x00\x00\x06\x09\x00\x00\x07\x09\x00\x00\x03\x0B\x00\x00\x04\x0B\x00\x00\x08\x09\x00\x00\x05\x0B\x00\x00\x06\x0B\x00\x00\x89\x03\x00\x00\x02\x01\x00\x00\x01\x03\x00\x00\x15\x01\x00\x00\x6E\x03\x00\x00\x00\x01',
    _globals = (b'\x00\x00\x11\x23PaMacCore_GetChannelName',0,b'\x00\x00\x5F\x23PaMacCore_SetupChannelMap',0,b'\x00\x00\x64\x23PaMacCore_SetupStreamInfo',0,b'\x00\x00\x23\x23PaWasapi_IsLoopback',0,b'\x00\x00\x5A\x23PaWasapi_UpdateDeviceList',0,b'\x00\x00\x41\x23Pa_AbortStream',0,b'\x00\x00\x41\x23Pa_CloseStream',0,b'\x00\x00\x5A\x23Pa_GetDefaultHostApi',0,b'\x00\x00\x5A\x23Pa_GetDefaultInputDevice',0,b'\x00\x00\x5A\x23Pa_GetDefaultOutputDevice',0,b'\x00\x00\x5A\x23Pa_GetDeviceCount',0,b'\x00\x00\x00\x23Pa_GetDeviceInfo',0,b'\x00\x00\x0E\x23Pa_GetErrorText',0,b'\x00\x00\x5A\x23Pa_GetHostApiCount',0,b'\x00\x00\x03\x23Pa_GetHostApiInfo',0,b'\x00\x00\x09\x23Pa_GetLastHostErrorInfo',0,b'\x00\x00\x2A\x23Pa_GetSampleSize',0,b'\x00\x00\x18\x23Pa_GetStreamCpuLoad',0,b'\x00\x00\x06\x23Pa_GetStreamHostApiType',0,b'\x00\x00\x0B\x23Pa_GetStreamInfo',0,b'\x00\x00\x5C\x23Pa_GetStreamReadAvailable',0,b'\x00\x00\x18\x23Pa_GetStreamTime',0,b'\x00\x00\x5C\x23Pa_GetStreamWriteAvailable',0,b'\x00\x00\x5A\x23Pa_GetVersion',0,b'\x00\x00\x16\x23Pa_GetVersionText',0,b'\x00\x00\x26\x23Pa_HostApiDeviceIndexToDeviceIndex',0,b'\x00\x00\x1B\x23Pa_HostApiTypeIdToHostApiIndex',0,b'\x00\x00\x5A\x23Pa_Initialize',0,b'\x00\x00\x1E\x23Pa_IsFormatSupported',0,b'\x00\x00\x41\x23Pa_IsStreamActive',0,b'\x00\x00\x41\x23Pa_IsStreamStopped',0,b'\x00\x00\x37\x23Pa_OpenDefaultStream',0,b'\x00\x00\x2D\x23Pa_OpenStream',0,b'\x00\x00\x44\x23Pa_ReadStream',0,b'\x00\x00\x4E\x23Pa_SetStreamFinishedCallback',0,b'\x00\x00\x68\x23Pa_Sleep',0,b'\x00\x00\x41\x23Pa_StartStream',0,b'\x00\x00\x41\x23Pa_StopStream',0,b'\x00\x00\x5A\x23Pa_Terminate',0,b'\x00\x00\x49\x23Pa_WriteStream',0,b'\xFF\xFF\xFF\x0BeAudioCategoryAlerts',4,b'\xFF\xFF\xFF\x0BeAudioCategoryCommunications',3,b'\xFF\xFF\xFF\x0BeAudioCategoryGameChat',8,b'\xFF\xFF\xFF\x0BeAudioCategoryGameEffects',6,b'\xFF\xFF\xFF\x0BeAudioCategoryGameMedia',7,b'\xFF\xFF\xFF\x0BeAudioCategoryMedia',11,b'\xFF\xFF\xFF\x0BeAudioCategoryMovie',10,b'\xFF\xFF\xFF\x0BeAudioCategoryOther',0,b'\xFF\xFF\xFF\x0BeAudioCategorySoundEffects',5,b'\xFF\xFF\xFF\x0BeAudioCategorySpeech',9,b'\xFF\xFF\xFF\x0BeStreamOptionMatchFormat',2,b'\xFF\xFF\xFF\x0BeStreamOptionNone',0,b'\xFF\xFF\xFF\x0BeStreamOptionRaw',1,b'\xFF\xFF\xFF\x0BeThreadPriorityAudio',1,b'\xFF\xFF\xFF\x0BeThreadPriorityCapture',2,b'\xFF\xFF\xFF\x0BeThreadPriorityDistribution',3,b'\xFF\xFF\xFF\x0BeThreadPriorityGames',4,b'\xFF\xFF\xFF\x0BeThreadPriorityNone',0,b'\xFF\xFF\xFF\x0BeThreadPriorityPlayback',5,b'\xFF\xFF\xFF\x0BeThreadPriorityProAudio',6,b'\xFF\xFF\xFF\x0BeThreadPriorityWindowManager',7,b'\xFF\xFF\xFF\x0BpaAL',9,b'\xFF\xFF\xFF\x0BpaALSA',8,b'\xFF\xFF\xFF\x0BpaASIO',3,b'\xFF\xFF\xFF\x0BpaAbort',2,b'\xFF\xFF\xFF\x1FpaAsioUseChannelSelectors',1,b'\xFF\xFF\xFF\x0BpaAudioScienceHPI',14,b'\xFF\xFF\xFF\x0BpaBadBufferPtr',-9972,b'\xFF\xFF\xFF\x0BpaBadIODeviceCombination',-9993,b'\xFF\xFF\xFF\x0BpaBadStreamPtr',-9988,b'\xFF\xFF\xFF\x0BpaBeOS',10,b'\xFF\xFF\xFF\x0BpaBufferTooBig',-9991,b'\xFF\xFF\xFF\x0BpaBufferTooSmall',-9990,b'\xFF\xFF\xFF\x0BpaCanNotReadFromACallbackStream',-9977,b'\xFF\xFF\xFF\x0BpaCanNotReadFromAnOutputOnlyStream',-9975,b'\xFF\xFF\xFF\x0BpaCanNotWriteToACallbackStream',-9976,b'\xFF\xFF\xFF\x0BpaCanNotWriteToAnInputOnlyStream',-9974,b'\xFF\xFF\xFF\x1FpaClipOff',1,b'\xFF\xFF\xFF\x0BpaComplete',1,b'\xFF\xFF\xFF\x0BpaContinue',0,b'\xFF\xFF\xFF\x0BpaCoreAudio',5,b'\xFF\xFF\xFF\x1FpaCustomFormat',65536,b'\xFF\xFF\xFF\x0BpaDeviceUnavailable',-9985,b'\xFF\xFF\xFF\x0BpaDirectSound',1,b'\xFF\xFF\xFF\x1FpaDitherOff',2,b'\xFF\xFF\xFF\x1FpaFloat32',1,b'\xFF\xFF\xFF\x1FpaFormatIsSupported',0,b'\xFF\xFF\xFF\x1FpaFramesPerBufferUnspecified',0,b'\xFF\xFF\xFF\x0BpaHostApiNotFound',-9979,b'\xFF\xFF\xFF\x0BpaInDevelopment',0,b'\xFF\xFF\xFF\x0BpaIncompatibleHostApiSpecificStreamInfo',-9984,b'\xFF\xFF\xFF\x0BpaIncompatibleStreamHostApi',-9973,b'\xFF\xFF\xFF\x1FpaInputOverflow',2,b'\xFF\xFF\xFF\x0BpaInputOverflowed',-9981,b'\xFF\xFF\xFF\x1FpaInputUnderflow',1,b'\xFF\xFF\xFF\x0BpaInsufficientMemory',-9992,b'\xFF\xFF\xFF\x1FpaInt16',8,b'\xFF\xFF\xFF\x1FpaInt24',4,b'\xFF\xFF\xFF\x1FpaInt32',2,b'\xFF\xFF\xFF\x1FpaInt8',16,b'\xFF\xFF\xFF\x0BpaInternalError',-9986,b'\xFF\xFF\xFF\x0BpaInvalidChannelCount',-9998,b'\xFF\xFF\xFF\x0BpaInvalidDevice',-9996,b'\xFF\xFF\xFF\x0BpaInvalidFlag',-9995,b'\xFF\xFF\xFF\x0BpaInvalidHostApi',-9978,b'\xFF\xFF\xFF\x0BpaInvalidSampleRate',-9997,b'\xFF\xFF\xFF\x0BpaJACK',12,b'\xFF\xFF\xFF\x0BpaMME',2,b'\xFF\xFF\xFF\x1FpaMacCoreChangeDeviceParameters',1,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityHigh',1024,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityLow',768,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMax',0,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMedium',512,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMin',256,b'\xFF\xFF\xFF\x1FpaMacCoreFailIfConversionRequired',2,b'\xFF\xFF\xFF\x1FpaMacCoreMinimizeCPU',257,b'\xFF\xFF\xFF\x1FpaMacCoreMinimizeCPUButPlayNice',256,b'\xFF\xFF\xFF\x1FpaMacCorePlayNice',0,b'\xFF\xFF\xFF\x1FpaMacCorePro',1,b'\xFF\xFF\xFF\x1FpaNeverDropInput',4,b'\xFF\xFF\xFF\x1FpaNoDevice',-1,b'\xFF\xFF\xFF\x0BpaNoError',0,b'\xFF\xFF\xFF\x1FpaNoFlag',0,b'\xFF\xFF\xFF\x1FpaNonInterleaved',2147483648,b'\xFF\xFF\xFF\x0BpaNotInitialized',-10000,b'\xFF\xFF\xFF\x0BpaNullCallback',-9989,b'\xFF\xFF\xFF\x0BpaOSS',7,b'\xFF\xFF\xFF\x1FpaOutputOverflow',8,b'\xFF\xFF\xFF\x1FpaOutputUnderflow',4,b'\xFF\xFF\xFF\x0BpaOutputUnderflowed',-9980,b'\xFF\xFF\xFF\x1FpaPlatformSpecificFlags',4294901760,b'\xFF\xFF\xFF\x1FpaPrimeOutputBuffersUsingStreamCallback',8,b'\xFF\xFF\xFF\x1FpaPrimingOutput',16,b'\xFF\xFF\xFF\x0BpaSampleFormatNotSupported',-9994,b'\xFF\xFF\xFF\x0BpaSoundManager',4,b'\xFF\xFF\xFF\x0BpaStreamIsNotStopped',-9982,b'\xFF\xFF\xFF\x0BpaStreamIsStopped',-9983,b'\xFF\xFF\xFF\x0BpaTimedOut',-9987,b'\xFF\xFF\xFF\x1FpaUInt8',32,b'\xFF\xFF\xFF\x0BpaUnanticipatedHostError',-9999,b'\xFF\xFF\xFF\x1FpaUseHostApiSpecificDeviceSpecification',-2,b'\xFF\xFF\xFF\x0BpaWASAPI',13,b'\xFF\xFF\xFF\x0BpaWDMKS',11,b'\xFF\xFF\xFF\x0BpaWinWasapiAutoConvert',64,b'\xFF\xFF\xFF\x0BpaWinWasapiExclusive',1,b'\xFF\xFF\xFF\x0BpaWinWasapiExplicitSampleFormat',32,b'\xFF\xFF\xFF\x0BpaWinWasapiPolling',8,b'\xFF\xFF\xFF\x0BpaWinWasapiRedirectHostProcessor',2,b'\xFF\xFF\xFF\x0BpaWinWasapiThreadPriority',16,b'\xFF\xFF\xFF\x0BpaWinWasapiUseChannelMask',4),
    _struct_unions = ((b'\x00\x00\x00\x7D\x00\x00\x00\x02$PaMacCoreStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x61\x11channelMap',b'\x00\x00\x2B\x11channelMapSize'),(b'\x00\x00\x00\x75\x00\x00\x00\x02PaAsioStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x8A\x11channelSelectors'),(b'\x00\x00\x00\x77\x00\x00\x00\x02PaDeviceInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x88\x11name',b'\x00\x00\x01\x11hostApi',b'\x00\x00\x01\x11maxInputChannels',b'\x00\x00\x01\x11maxOutputChannels',b'\x00\x00\x21\x11defaultLowInputLatency',b'\x00\x00\x21\x11defaultLowOutputLatency',b'\x00\x00\x21\x11defaultHighInputLatency',b'\x00\x00\x21\x11defaultHighOutputLatency',b'\x00\x00\x21\x11defaultSampleRate'),(b'\x00\x00\x00\x7A\x00\x00\x00\x02PaHostApiInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x1C\x11type',b'\x00\x00\x88\x11name',b'\x00\x00\x01\x11deviceCount',b'\x00\x00\x01\x11defaultInputDevice',b'\x00\x00\x01\x11defaultOutputDevice'),(b'\x00\x00\x00\x7C\x00\x00\x00\x02PaHostErrorInfo',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x69\x11errorCode',b'\x00\x00\x88\x11errorText'),(b'\x00\x00\x00\x7F\x00\x00\x00\x02PaStreamCallbackTimeInfo',b'\x00\x00\x21\x11inputBufferAdcTime',b'\x00\x00\x21\x11currentTime',b'\x00\x00\x21\x11outputBufferDacTime'),(b'\x00\x00\x00\x81\x00\x00\x00\x02PaStreamInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x21\x11inputLatency',b'\x00\x00\x21\x11outputLatency',b'\x00\x00\x21\x11sampleRate'),(b'\x00\x00\x00\x82\x00\x00\x00\x02PaStreamParameters',b'\x00\x00\x01\x11device',b'\x00\x00\x01\x11channelCount',b'\x00\x00\x2B\x11sampleFormat',b'\x00\x00\x21\x11suggestedLatency',b'\x00\x00\x07\x11hostApiSpecificStreamInfo'),(b'\x00\x00\x00\x85\x00\x00\x00\x02PaWasapiStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x2B\x11channelMask',b'\x00\x00\x8C\x11hostProcessorOutput',b'\x00\x00\x8C\x11hostProcessorInput',b'\x00\x00\x87\x11threadPriority',b'\x00\x00\x84\x11streamCategory',b'\x00\x00\x86\x11streamOption')),
    _enums = (b'\x00\x00\x00\x78\x00\x00\x00\x15PaErrorCode\x00paNoError,paNotInitialized,paUnanticipatedHostError,paInvalidChannelCount,paInvalidSampleRate,paInvalidDevice,paInvalidFlag,paSampleFormatNotSupported,paBadIODeviceCombination,paInsufficientMemory,paBufferTooBig,paBufferTooSmall,paNullCallback,paBadStreamPtr,paTimedOut,paInternalError,paDeviceUnavailable,paIncompatibleHostApiSpecificStreamInfo,paStreamIsStopped,paStreamIsNotStopped,paInputOverflowed,paOutputUnderflowed,paHostApiNotFound,paInvalidHostApi,paCanNotReadFromACallbackStream,paCanNotWriteToACallbackStream,paCanNotReadFromAnOutputOnlyStream,paCanNotWriteToAnInputOnlyStream,paIncompatibleStreamHostApi,paBadBufferPtr',b'\x00\x00\x00\x1C\x00\x00\x00\x16PaHostApiTypeId\x00paInDevelopment,paDirectSound,paMME,paASIO,paSoundManager,paCoreAudio,paOSS,paALSA,paAL,paBeOS,paWDMKS,paJACK,paWASAPI,paAudioScienceHPI',b'\x00\x00\x00\x7E\x00\x00\x00\x16PaStreamCallbackResult\x00paContinue,paComplete,paAbort',b'\x00\x00\x00\x83\x00\x00\x00\x16PaWasapiFlags\x00paWinWasapiExclusive,paWinWasapiRedirectHostProcessor,paWinWasapiUseChannelMask,paWinWasapiPolling,paWinWasapiThreadPriority,paWinWasapiExplicitSampleFormat,paWinWasapiAutoConvert',b'\x00\x00\x00\x84\x00\x00\x00\x16PaWasapiStreamCategory\x00eAudioCategoryOther,eAudioCategoryCommunications,eAudioCategoryAlerts,eAudioCategorySoundEffects,eAudioCategoryGameEffects,eAudioCategoryGameMedia,eAudioCategoryGameChat,eAudioCategorySpeech,eAudioCategoryMovie,eAudioCategoryMedia',b'\x00\x00\x00\x86\x00\x00\x00\x16PaWasapiStreamOption\x00eStreamOptionNone,eStreamOptionRaw,eStreamOptionMatchFormat',b'\x00\x00\x00\x87\x00\x00\x00\x16PaWasapiThreadPriority\x00eThreadPriorityNone,eThreadPriorityAudio,eThreadPriorityCapture,eThreadPriorityDistribution,eThreadPriorityGames,eThreadPriorityPlayback,eThreadPriorityProAudio,eThreadPriorityWindowManager'),
    _typenames = (b'\x00\x00\x00\x75PaAsioStreamInfo',b'\x00\x00\x00\x01PaDeviceIndex',b'\x00\x00\x00\x77PaDeviceInfo',b'\x00\x00\x00\x01PaError',b'\x00\x00\x00\x78PaErrorCode',b'\x00\x00\x00\x01PaHostApiIndex',b'\x00\x00\x00\x7APaHostApiInfo',b'\x00\x00\x00\x1CPaHostApiTypeId',b'\x00\x00\x00\x7CPaHostErrorInfo',b'\x00\x00\x00\x7DPaMacCoreStreamInfo',b'\x00\x00\x00\x2BPaSampleFormat',b'\x00\x00\x00\x8DPaStream',b'\x00\x00\x00\x52PaStreamCallback',b'\x00\x00\x00\x2BPaStreamCallbackFlags',b'\x00\x00\x00\x7EPaStreamCallbackResult',b'\x00\x00\x00\x7FPaStreamCallbackTimeInfo',b'\x00\x00\x00\x6BPaStreamFinishedCallback',b'\x00\x00\x00\x2BPaStreamFlags',b'\x00\x00\x00\x81PaStreamInfo',b'\x00\x00\x00\x82PaStreamParameters',b'\x00\x00\x00\x21PaTime',b'\x00\x00\x00\x83PaWasapiFlags',b'\x00\x00\x00\x8CPaWasapiHostProcessorCallback',b'\x00\x00\x00\x84PaWasapiStreamCategory',b'\x00\x00\x00\x85PaWasapiStreamInfo',b'\x00\x00\x00\x86PaWasapiStreamOption',b'\x00\x00\x00\x87PaWasapiThreadPriority',b'\x00\x00\x00\x2BPaWinWaveFormatChannelMask',b'\x00\x00\x00\x8BSInt32'),
)

# Load PortAudio library depending on the OS platform
if portaudiolib is None:
    try:
        # Try finding standard PortAudio installations
        for libname in ('portaudio', 'bin\\libportaudio-2.dll', 'lib/libportaudio.dylib'):
            libname = find_library(libname)
            if libname is not None: break
        else: raise OSError(translations['portaudio_not_found'])

        portaudiolib = ffi.dlopen(libname)
    except OSError:
        # Fallback to local binaries if not found in system paths
        if platform.system() == 'Darwin': libname = 'libportaudio.dylib'
        elif platform.system() == 'Windows':
            if platform.machine().lower() in ('arm64', 'aarch64'): platform_suffix = 'arm64'
            else: platform_suffix = platform.architecture()[0]

            libname = 'libportaudio' + platform_suffix + ('-asio.dll' if 'SD_ENABLE_ASIO' in os.environ else '.dll')
        else: raise

        libdir = configs.get("portaudiolib", os.path.join("assets", "binary"))
        libpath = os.path.join(libdir, libname)

        # Unpack binary from a pre-packaged pickle file if it doesn't exist
        if not os.path.exists(libpath):
            with open(os.path.join(libdir, "portaudiolib.bin"), "rb") as f:
                model = pickle.load(f)

            with open(libpath, "wb") as w:
                w.write(model[libname])

        portaudiolib = ffi.dlopen(os.path.abspath(libpath))

initialized = 0
last_callback = None
sampleformats = {'float32': portaudiolib.paFloat32, 'int32': portaudiolib.paInt32, 'int24': portaudiolib.paInt24, 'int16': portaudiolib.paInt16, 'int8': portaudiolib.paInt8, 'uint8': portaudiolib.paUInt8}

def play(data, samplerate=None, mapping=None, blocking=False, loop=False, **kwargs):
    """
    Play audio data using an output stream.

    Args:
        data (array-like): The audio data to play.
        samplerate (int, optional): The sampling frequency. Defaults to None.
        mapping (list/tuple, optional): Channel mapping configuration. Defaults to None.
        blocking (bool): If True, wait until playback finishes. Defaults to False.
        loop (bool): If True, loop the audio data indefinitely. Defaults to False.
        **kwargs: Additional stream parameters.
    """

    ctx = CallbackContext(loop=loop)
    ctx.frames = ctx.check_data(data, mapping, kwargs.get('device'))

    def callback(outdata, frames, time, status):
        assert len(outdata) == frames
        ctx.callback_enter(status, outdata)
        ctx.write_outdata(outdata)
        ctx.callback_exit()

    ctx.start_stream(OutputStream, samplerate, ctx.output_channels, ctx.output_dtype, callback, blocking, prime_output_buffers_using_stream_callback=False, **kwargs)

def rec(frames=None, samplerate=None, channels=None, dtype=None, out=None, mapping=None, blocking=False, **kwargs):
    """
    Record audio data using an input stream.

    Args:
        frames (int, optional): Number of frames to record. Defaults to None.
        samplerate (int, optional): The sampling frequency. Defaults to None.
        channels (int, optional): Number of input channels. Defaults to None.
        dtype (str/type, optional): Data type of the recorded audio. Defaults to None.
        out (array-like, optional): Pre-allocated buffer to store recorded audio. Defaults to None.
        mapping (list/tuple, optional): Channel mapping configuration. Defaults to None.
        blocking (bool): If True, wait until recording finishes. Defaults to False.
        **kwargs: Additional stream parameters.

    Returns:
        array-like: The array containing the recorded audio data.
    """

    ctx = CallbackContext()
    out, ctx.frames = ctx.check_out(out, frames, channels, dtype, mapping)

    def callback(indata, frames, time, status):
        assert len(indata) == frames
        ctx.callback_enter(status, indata)
        ctx.read_indata(indata)
        ctx.callback_exit()

    ctx.start_stream(InputStream, samplerate, ctx.input_channels,
                     ctx.input_dtype, callback, blocking, **kwargs)
    return out

def playrec(data, samplerate=None, channels=None, dtype=None, out=None, input_mapping=None, output_mapping=None, blocking=False, **kwargs):
    """
    Simultaneously play and record audio data using a duplex stream.

    Args:
        data (array-like): Audio data for playback.
        samplerate (int, optional): The sampling frequency. Defaults to None.
        channels (int, optional): Number of input channels. Defaults to None.
        dtype (str/type, optional): Data type for recording. Defaults to None.
        out (array-like, optional): Pre-allocated buffer for recorded data. Defaults to None.
        input_mapping (list/tuple, optional): Input channel mapping. Defaults to None.
        output_mapping (list/tuple, optional): Output channel mapping. Defaults to None.
        blocking (bool): If True, wait until the operation completes. Defaults to False.
        **kwargs: Additional stream parameters.

    Raises:
        ValueError: If input and output frame counts do not match.

    Returns:
        array-like: The array containing the recorded audio data.
    """

    ctx = CallbackContext()
    output_frames = ctx.check_data(data, output_mapping, kwargs.get('device'))

    if dtype is None: dtype = ctx.data.dtype
    out, input_frames = ctx.check_out(out, output_frames, channels, dtype, input_mapping)

    if input_frames != output_frames: raise ValueError("Input and output frame sizes must match for playrec operation.")
    ctx.frames = input_frames

    def callback(indata, outdata, frames, time, status):
        assert len(indata) == len(outdata) == frames
        ctx.callback_enter(status, indata)
        ctx.read_indata(indata)
        ctx.write_outdata(outdata)
        ctx.callback_exit()

    ctx.start_stream(Stream, samplerate, (ctx.input_channels, ctx.output_channels), (ctx.input_dtype, ctx.output_dtype), callback, blocking, prime_output_buffers_using_stream_callback=False, **kwargs)
    return out

def wait(ignore_errors=True):
    """
    Wait for the active playback/recording stream to finish.

    Args:
        ignore_errors (bool): If True, ignore stream processing errors during wait. Defaults to True.
    """

    if last_callback: return last_callback.wait(ignore_errors)

def stop(ignore_errors=True):
    """
    Stop and close the currently active stream.

    Args:
        ignore_errors (bool): If True, suppress errors during shutdown. Defaults to True.
    """

    if last_callback:
        last_callback.stream.stop(ignore_errors)
        last_callback.stream.close(ignore_errors)

def get_status():
    """
    Get the status flags of the last active stream callback.

    Raises:
        RuntimeError: If no active stream or callback history exists.

    Returns:
        PaCallbackFlags: The status flags of the stream.
    """

    if last_callback: return last_callback.status
    else: raise RuntimeError("No active stream or callback status found.")

def get_stream():
    """
    Retrieve the last active stream object.

    Raises:
        RuntimeError: If no stream has been opened or initialized yet.

    Returns:
        StreamBase: The active stream instance.
    """

    if last_callback: return last_callback.stream
    else: raise RuntimeError("No active stream found.")

def query_devices(device=None, kind=None):
    """
    Query information about host audio devices.

    Args:
        device (int/str, optional): Device index or name pattern. Defaults to None.
        kind (str, optional): Filter devices by 'input' or 'output'. Defaults to None.

    Raises:
        ValueError: If 'kind' is invalid, or selected device does not support specified 'kind'.
        PortAudioError: If PortAudio fails to fetch information for the device index.

    Returns:
        dict/DeviceList: Information dictionary of the specified device, or a list of all devices.
    """

    if kind not in ('input', 'output', None): raise ValueError(f"Invalid device kind: {kind!r}. Expected 'input', 'output', or None.")
    if device is None and kind is None: return DeviceList(query_devices(i) for i in range(check(portaudiolib.Pa_GetDeviceCount())))

    device = get_device_id(device, kind, raise_on_error=True)
    info = portaudiolib.Pa_GetDeviceInfo(device)

    if not info: raise PortAudioError(f"Failed to query device information for index: {device}")
    assert info.structVersion == 2
    name_bytes = ffi_string(info.name)

    # Handle OS-specific string encodings for device names
    try:
        name = name_bytes.decode('utf-8')
    except UnicodeDecodeError:
        api_idx = portaudiolib.Pa_HostApiTypeIdToHostApiIndex

        if info.hostApi in (api_idx(portaudiolib.paDirectSound), api_idx(portaudiolib.paMME)): name = name_bytes.decode('mbcs')
        elif info.hostApi == api_idx(portaudiolib.paASIO): name = name_bytes.decode(locale.getpreferredencoding())
        else: raise

    device_dict = {
        'name': name,
        'index': device,
        'hostapi': info.hostApi,
        'max_input_channels': info.maxInputChannels,
        'max_output_channels': info.maxOutputChannels,
        'default_low_input_latency': info.defaultLowInputLatency,
        'default_low_output_latency': info.defaultLowOutputLatency,
        'default_high_input_latency': info.defaultHighInputLatency,
        'default_high_output_latency': info.defaultHighOutputLatency,
        'default_samplerate': info.defaultSampleRate,
    }

    if kind and device_dict['max_' + kind + '_channels'] < 1: raise ValueError(f"Device '{device_dict['name']}' does not support {kind} channels.")
    return device_dict

def query_hostapis(index=None):
    """
    Query information about Host APIs supported by the system.

    Args:
        index (int, optional): The index of the Host API. Defaults to None.

    Raises:
        PortAudioError: If details cannot be retrieved for the given Host API index.

    Returns:
        dict/tuple: Dictionary info of the API, or a tuple containing info of all host APIs.
    """

    if index is None: return tuple(query_hostapis(i) for i in range(check(portaudiolib.Pa_GetHostApiCount())))
    info = portaudiolib.Pa_GetHostApiInfo(index)
    if not info: raise PortAudioError(f"Failed to fetch host API info for index: {index}")

    assert info.structVersion == 1
    return {
        'name': ffi_string(info.name).decode(),
        'devices': [portaudiolib.Pa_HostApiDeviceIndexToDeviceIndex(index, i) for i in range(info.deviceCount)],
        'default_input_device': info.defaultInputDevice,
        'default_output_device': info.defaultOutputDevice,
    }

def check_input_settings(device=None, channels=None, dtype=None, extra_settings=None, samplerate=None):
    """
    Validate if the provided input configuration is natively supported by PortAudio.

    Args:
        device (int/str, optional): Target device.
        channels (int, optional): Requested number of input channels.
        dtype (str/type, optional): Desired sample format.
        extra_settings (cdata, optional): Host API specific configurations.
        samplerate (int, optional): Target sampling frequency.
    """

    parameters, dtype, _, samplerate = get_stream_parameters('input', device=device, channels=channels, dtype=dtype, latency=None, extra_settings=extra_settings, samplerate=samplerate)
    check(portaudiolib.Pa_IsFormatSupported(parameters, ffi.NULL, samplerate))

def check_output_settings(device=None, channels=None, dtype=None, extra_settings=None, samplerate=None):
    """
    Validate if the provided output configuration is natively supported by PortAudio.

    Args:
        device (int/str, optional): Target device.
        channels (int, optional): Requested number of output channels.
        dtype (str/type, optional): Desired sample format.
        extra_settings (cdata, optional): Host API specific configurations.
        samplerate (int, optional): Target sampling frequency.
    """

    parameters, dtype, _, samplerate = get_stream_parameters('output', device=device, channels=channels, dtype=dtype, latency=None, extra_settings=extra_settings, samplerate=samplerate)
    check(portaudiolib.Pa_IsFormatSupported(ffi.NULL, parameters, samplerate))

def sleep(msec):
    """
    Put the current executing thread to sleep for a specified duration using PortAudio.

    Args:
        msec (int): Duration to sleep in milliseconds.
    """

    portaudiolib.Pa_Sleep(msec)

def get_portaudio_version():
    """
    Get the version identifier and textual release details of PortAudio.

    Returns:
        tuple: An (int, str) containing the version code and description text.
    """

    return portaudiolib.Pa_GetVersion(), ffi_string(portaudiolib.Pa_GetVersionText()).decode()

class StreamBase:
    def __init__(self, kind, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None, userdata=None, wrap_callback=None):
        """
        Initialize the base stream with relevant parameters.

        Raises:
            ValueError: If sample rates between input and output configurations differ in a duplex setup.
            PortAudioError: If metadata cannot be acquired for the opened stream.
        """

        assert kind in ('input', 'output', 'duplex')
        assert wrap_callback in ('array', 'buffer', None)
        stream_flags = portaudiolib.paNoFlag

        # Fallback to defaults if specific parameters aren't configured
        if blocksize is None: blocksize = default.blocksize
        if clip_off is None: clip_off = default.clip_off
        if dither_off is None: dither_off = default.dither_off
        if never_drop_input is None: never_drop_input = default.never_drop_input
        if prime_output_buffers_using_stream_callback is None: prime_output_buffers_using_stream_callback = default.prime_output_buffers_using_stream_callback
        if clip_off: stream_flags |= portaudiolib.paClipOff
        if dither_off: stream_flags |= portaudiolib.paDitherOff
        if never_drop_input: stream_flags |= portaudiolib.paNeverDropInput
        if prime_output_buffers_using_stream_callback: stream_flags |= portaudiolib.paPrimeOutputBuffersUsingStreamCallback

        # Handle stream configuration parameter retrieval depending on stream nature
        if kind == 'duplex':
            idevice, odevice = split(device)
            ichannels, ochannels = split(channels)
            idtype, odtype = split(dtype)
            ilatency, olatency = split(latency)
            iextra, oextra = split(extra_settings)

            iparameters, idtype, isize, isamplerate = get_stream_parameters('input', idevice, ichannels, idtype, ilatency, iextra, samplerate)
            oparameters, odtype, osize, osamplerate = get_stream_parameters('output', odevice, ochannels, odtype, olatency, oextra, samplerate)

            self._dtype = idtype, odtype
            self._device = iparameters.device, oparameters.device
            self._channels = iparameters.channelCount, oparameters.channelCount
            self._samplesize = isize, osize

            if isamplerate != osamplerate: raise ValueError
            else: samplerate = isamplerate
        else:
            parameters, self._dtype, self._samplesize, samplerate = get_stream_parameters(kind, device, channels, dtype, latency, extra_settings, samplerate)
            self._device = parameters.device
            self._channels = parameters.channelCount
            iparameters = ffi.NULL
            oparameters = ffi.NULL

            if kind == 'input': iparameters = parameters
            elif kind == 'output': oparameters = parameters

        ffi_callback = ffi.callback('PaStreamCallback', error=portaudiolib.paAbort)

        # Map dynamic callback functions based on the data formatting requirements ('array' or 'buffer')
        if callback is None: callback_ptr = ffi.NULL
        elif kind == 'input' and wrap_callback == 'buffer':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                data = mbuffer(iptr, frames, self._channels, self._samplesize)
                return _wrap_callback(callback, data, frames, time, status)
        elif kind == 'input' and wrap_callback == 'array':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                data = array(mbuffer(iptr, frames, self._channels, self._samplesize), self._channels, self._dtype)
                return _wrap_callback(callback, data, frames, time, status)
        elif kind == 'output' and wrap_callback == 'buffer':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                data = mbuffer(optr, frames, self._channels, self._samplesize)
                return _wrap_callback(callback, data, frames, time, status)
        elif kind == 'output' and wrap_callback == 'array':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                data = array(mbuffer(optr, frames, self._channels, self._samplesize), self._channels, self._dtype)
                return _wrap_callback(callback, data, frames, time, status)
        elif kind == 'duplex' and wrap_callback == 'buffer':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                idata, odata = mbuffer(iptr, frames, self._channels[0], self._samplesize[0]), mbuffer(optr, frames, self._channels[1], self._samplesize[1])
                return _wrap_callback(callback, idata, odata, frames, time, status)
        elif kind == 'duplex' and wrap_callback == 'array':
            @ffi_callback
            def callback_ptr(iptr, optr, frames, time, status, _):
                idata, odata = array(mbuffer(iptr, frames, self._channels[0], self._samplesize[0]), self._channels[0], self._dtype[0]), array(mbuffer(optr, frames, self._channels[1], self._samplesize[1]), self._channels[1], self._dtype[1])
                return _wrap_callback(callback, idata, odata, frames, time, status)
        else: callback_ptr = ffi.cast('PaStreamCallback*', callback)

        self._callback = callback_ptr
        if userdata is None: userdata = ffi.NULL
    
        self._ptr = ffi.new('PaStream**')
        check(portaudiolib.Pa_OpenStream(self._ptr, iparameters, oparameters, samplerate, blocksize, stream_flags, callback_ptr, userdata), self.__class__.__name__)

        self._ptr = self._ptr[0]
        self._blocksize = blocksize

        info = portaudiolib.Pa_GetStreamInfo(self._ptr)
        if not info: raise PortAudioError("Could not retrieve info from opened PortAudio stream.")
        self._samplerate = info.sampleRate

        if not oparameters: self._latency = info.inputLatency
        elif not iparameters: self._latency = info.outputLatency
        else: self._latency = info.inputLatency, info.outputLatency

        if finished_callback:
            if isinstance(finished_callback, ffi.CData): self._finished_callback = finished_callback
            else:
                def finished_callback_wrapper(_):
                    return finished_callback()

                self._finished_callback = ffi.callback('PaStreamFinishedCallback', finished_callback_wrapper)

            check(portaudiolib.Pa_SetStreamFinishedCallback(self._ptr, self._finished_callback))

    _ptr = ffi.NULL

    @property
    def samplerate(self):
        return self._samplerate

    @property
    def blocksize(self):
        return self._blocksize

    @property
    def device(self):
        return self._device

    @property
    def channels(self):
        return self._channels

    @property
    def dtype(self):
        return self._dtype

    @property
    def samplesize(self):
        return self._samplesize

    @property
    def latency(self):
        return self._latency

    @property
    def active(self):
        if self.closed: return False
        return check(portaudiolib.Pa_IsStreamActive(self._ptr)) == 1

    @property
    def stopped(self):
        if self.closed: return True
        return check(portaudiolib.Pa_IsStreamStopped(self._ptr)) == 1

    @property
    def closed(self):
        return self._ptr == ffi.NULL

    @property
    def time(self):
        time = portaudiolib.Pa_GetStreamTime(self._ptr)
        if not time: raise PortAudioError("Failed to fetch current stream timeline position.")
        return time

    @property
    def cpu_load(self):
        return portaudiolib.Pa_GetStreamCpuLoad(self._ptr)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
        self.close()

    def start(self):
        """Start the processing on the active stream."""

        err = portaudiolib.Pa_StartStream(self._ptr)
        if err != portaudiolib.paStreamIsNotStopped: check(err)

    def stop(self, ignore_errors=True):
        """Stop processing audio on the stream."""

        err = portaudiolib.Pa_StopStream(self._ptr)
        if not ignore_errors: check(err)

    def abort(self, ignore_errors=True):
        """Abort audio processing immediately on the stream."""

        err = portaudiolib.Pa_AbortStream(self._ptr)
        if not ignore_errors: check(err)

    def close(self, ignore_errors=True):
        """Close the stream and release internal processing allocations."""

        err = portaudiolib.Pa_CloseStream(self._ptr)
        self._ptr = ffi.NULL
        if not ignore_errors: check(err)

class InputStreamBase(StreamBase):
    """Stream subclass specific to processing input captures."""

    @property
    def read_available(self):
        return check(portaudiolib.Pa_GetStreamReadAvailable(self._ptr))

    def _raw_read(self, frames):
        """
        Read synchronous buffer samples directly out from an input pipeline.

        Args:
            frames (int): Target frame chunk size.

        Returns:
            tuple: (buffer, bool) buffer data chunk and input overflow flag status.
        """

        channels, _ = split(self._channels)
        samplesize, _ = split(self._samplesize)

        data = ffi.new('signed char[]', channels * samplesize * frames)
        err = portaudiolib.Pa_ReadStream(self._ptr, data, frames)

        if err == portaudiolib.paInputOverflowed: overflowed = True
        else:
            check(err)
            overflowed = False

        return ffi.mbuffer(data), overflowed

class RawInputStream(InputStreamBase):
    """Raw byte-oriented variation of an Input Stream structure."""

    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='input', wrap_callback='buffer', **remove_self(locals()))

    read = InputStreamBase._raw_read

class OutputStreamBase(StreamBase):
    @property
    def write_available(self):
        return check(portaudiolib.Pa_GetStreamWriteAvailable(self._ptr))

    def _raw_write(self, data):
        """
        Write synchronous sample buffer objects onto an output pipeline.

        Args:
            data (bufferable): Buffer byte structure containing frame samples.

        Raises:
            ValueError: If sample data chunk dimensions are malformed.

        Returns:
            bool: True if output underflow occurred, False otherwise.
        """

        try:
            data = ffi.from_buffer(data)
        except AttributeError:
            pass
        except TypeError:
            pass

        _, samplesize = split(self._samplesize)
        _, channels = split(self._channels)

        samples, remainder = divmod(len(data), samplesize)
        if remainder: raise ValueError("Buffer data size is not a multiple of the sample size.")

        frames, remainder = divmod(samples, channels)
        if remainder: raise ValueError("Total sample count is not a multiple of the channel count.")

        err = portaudiolib.Pa_WriteStream(self._ptr, data, frames)

        if err == portaudiolib.paOutputUnderflowed: underflowed = True
        else:
            check(err)
            underflowed = False

        return underflowed

class RawOutputStream(OutputStreamBase):
    """Raw byte-oriented variation of an Output Stream structure."""

    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='output', wrap_callback='buffer', **remove_self(locals()))

    write = OutputStreamBase._raw_write

class RawStream(RawInputStream, RawOutputStream):
    """A full-duplex raw byte stream providing concurrent IO pipelines."""

    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='duplex', wrap_callback='buffer', **remove_self(locals()))

class InputStream(InputStreamBase):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        """
        Initialize an input stream.

        Args:
            samplerate (float): The desired sampling rate in Hz.
            blocksize (int): The number of frames passed to the stream callback.
            device (int or str): The device index or name.
            channels (int): The number of channels.
            dtype (str or numpy.dtype): The data type of the audio buffer.
            latency (float or str): The desired latency in seconds or 'low'/'high'.
            extra_settings (object): API-specific settings object.
            callback (callable): The callback function for audio processing.
            finished_callback (callable): Callback called when stream finishes.
            clip_off (bool): Disable clipping of out-of-range samples.
            dither_off (bool): Disable dithering.
            never_drop_input (bool): Overflow management strategy.
            prime_output_buffers_using_stream_callback (bool): Output priming control.
        """

        # Call the base class initializer for the input stream
        StreamBase.__init__(self, kind='input', wrap_callback='array', **remove_self(locals()))

    def read(self, frames):
        """
        Read data from the input stream.

        Args:
            frames (int): The number of frames to read.

        Returns:
            tuple: A tuple containing the audio data (numpy.ndarray) and an overflow indicator (bool).
        """

        dtype, _ = split(self._dtype)
        channels, _ = split(self._channels)

        data, overflowed = InputStreamBase._raw_read(self, frames)
        data = array(data, channels, dtype)

        return data, overflowed

class OutputStream(OutputStreamBase):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        """
        Initialize an output stream.

        Args:
            samplerate (float): The desired sampling rate in Hz.
            blocksize (int): The number of frames passed to the stream callback.
            device (int or str): The device index or name.
            channels (int): The number of channels.
            dtype (str or numpy.dtype): The data type of the audio buffer.
            latency (float or str): The desired latency in seconds or 'low'/'high'.
            extra_settings (object): API-specific settings object.
            callback (callable): The callback function for audio processing.
            finished_callback (callable): Callback called when stream finishes.
            clip_off (bool): Disable clipping of out-of-range samples.
            dither_off (bool): Disable dithering.
            never_drop_input (bool): Overflow management strategy.
            prime_output_buffers_using_stream_callback (bool): Output priming control.
        """

        # Call the base class initializer for the output stream
        StreamBase.__init__(self, kind='output', wrap_callback='array', **remove_self(locals()))

    def write(self, data):
        """
        Write data to the output stream.

        Args:
            data (array_like): The audio data to be played back.

        Returns:
            bool: An underflow indicator.
        """

        data = np.asarray(data)
        _, dtype = split(self._dtype)
        _, channels = split(self._channels)

        # Enforce the audio data to be 2-dimensional (frames, channels)
        if data.ndim < 2: data = data.reshape(-1, 1)
        elif data.ndim > 2: raise ValueError("Audio data must be 1-dimensional or 2-dimensional.")

        # Validate that data dimensions match configured channel counts
        if data.shape[1] != channels: raise ValueError(f"Channel mismatch: data has {data.shape[1]} channels, stream expects {channels}.")
        # Ensure the underlying data type matches the hardware requirements
        if data.dtype != dtype: raise TypeError(f"Data type mismatch: data is {data.dtype}, stream expects {dtype}.")
        # C-contiguous layout is mandatory for direct memory access via CFFI
        if not data.flags.c_contiguous:  raise TypeError("Audio data buffer must be C-contiguous.")

        return OutputStreamBase._raw_write(self, data)

class Stream(InputStream, OutputStream):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        """Initialize a duplex stream.

        Args:
            samplerate (float): The desired sampling rate in Hz.
            blocksize (int): The number of frames passed to the stream callback.
            device (int, str or tuple): The device index/name, or a tuple of (input, output).
            channels (int or tuple): Number of channels, or a tuple of (input, output).
            dtype (str, numpy.dtype or tuple): Data type, or a tuple of (input, output).
            latency (float, str or tuple): Latency setting, or a tuple of (input, output).
            extra_settings (object or tuple): Settings object, or a tuple of (input, output).
            callback (callable): The callback function for audio processing.
            finished_callback (callable): Callback called when stream finishes.
            clip_off (bool): Disable clipping of out-of-range samples.
            dither_off (bool): Disable dithering.
            never_drop_input (bool): Overflow management strategy.
            prime_output_buffers_using_stream_callback (bool): Output priming control.
        """

        # Call the base class initializer for a duplex stream configuration
        StreamBase.__init__(self, kind='duplex', wrap_callback='array', **remove_self(locals()))

class DeviceList(tuple):
    """A list of available audio devices."""
    __slots__ = ()

    def __repr__(self):
        """Return a formatted string representing all available devices."""

        # Get default input/output device indexes from PortAudio
        idev = get_device_id(default.device['input'], 'input')
        odev = get_device_id(default.device['output'], 'output')
        # Calculate padding size for clean string formatting alignment
        digits = len(str(portaudiolib.Pa_GetDeviceCount() - 1))
        hostapi_names = [hostapi['name'] for hostapi in query_hostapis()]

        def get_mark(idx):
            # Helper to assign visual flags: '>' for input, '<' for output, '*' for duplex
            return (' ', '>', '<', '*')[(idx == idev) + 2 * (idx == odev)]

        # Generate a multiline string for every audio device found in the structure
        text = '\n'.join('{mark} {idx:{dig}} {name}, {ha} ({ins} in, {outs} out)'.format(mark=get_mark(info['index']), idx=info['index'], dig=digits, name=info['name'], ha=hostapi_names[info['hostapi']], ins=info['max_input_channels'], outs=info['max_output_channels']) for info in self)
        return text

class CallbackFlags:
    """Flags passed to the stream callback function."""
    __slots__ = '_flags'

    def __init__(self, flags=0x0):
        """
        Initialize callback flags.

        Args:
            flags (int): Integer bitmask containing PortAudio status flags.
        """

        self._flags = flags

    def __repr__(self):
        flags = str(self)
        return f'<sounddevice.CallbackFlags: {flags}>'

    def __str__(self):
        # Format names cleanly by swapping underscores with spaces
        return ', '.join(name.replace('_', ' ') for name in dir(self) if not name.startswith('_') and getattr(self, name))

    def __bool__(self):
        return bool(self._flags)

    def __ior__(self, other):
        # Implement bitwise in-place OR assignments (e.g., flags |= other)
        if not isinstance(other, CallbackFlags): return NotImplemented
        self._flags |= other._flags
        return self

    @property
    def input_underflow(self):
        return self._hasflag(portaudiolib.paInputUnderflow)

    @input_underflow.setter
    def input_underflow(self, value):
        self._updateflag(portaudiolib.paInputUnderflow, value)

    @property
    def input_overflow(self):
        return self._hasflag(portaudiolib.paInputOverflow)

    @input_overflow.setter
    def input_overflow(self, value):
        self._updateflag(portaudiolib.paInputOverflow, value)

    @property
    def output_underflow(self):
        return self._hasflag(portaudiolib.paOutputUnderflow)

    @output_underflow.setter
    def output_underflow(self, value):
        self._updateflag(portaudiolib.paOutputUnderflow, value)

    @property
    def output_overflow(self):
        return self._hasflag(portaudiolib.paOutputOverflow)

    @output_overflow.setter
    def output_overflow(self, value):
        self._updateflag(portaudiolib.paOutputOverflow, value)

    @property
    def priming_output(self):
        return self._hasflag(portaudiolib.paPrimingOutput)

    def _hasflag(self, flag):
        # Perform binary bit-mask validation
        return bool(self._flags & flag)

    def _updateflag(self, flag, value):
        # Flip bits depending on target boolean flag value
        if value: self._flags |= flag
        else: self._flags &= ~flag

class InputOutputPair:
    """Helper class to store configurations for both input and output settings."""
    _indexmapping = {'input': 0, 'output': 1}

    def __init__(self, parent, default_attr):
        """
        Initialize an input-output pair.

        Args:
            parent (object): Reference to the owner container.
            default_attr (str): Name of the fallback attribute string.
        """

        self._pair = [None, None]
        self._parent = parent
        self._default_attr = default_attr

    def __getitem__(self, index):
        # Map indices 'input'/'output' or integers to internal list lookup
        index = self._indexmapping.get(index, index)
        value = self._pair[index]
        # Return fallback configuration if local context value is undefined
        if value is None: value = getattr(self._parent, self._default_attr)[index]
        return value

    def __setitem__(self, index, value):
        index = self._indexmapping.get(index, index)
        self._pair[index] = value

    def __repr__(self):
        return '[{0[0]!r}, {0[1]!r}]'.format(self)

class default:
    """Global default settings container."""

    _pairs = 'device', 'channels', 'dtype', 'latency', 'extra_settings'
    device = (None, None)
    _default_channels = (None, None)
    channels = _default_channels
    _default_dtype = 'float32', 'float32'
    dtype = _default_dtype
    _default_latency = 'high', 'high'
    latency = _default_latency
    _default_extra_settings = (None, None)
    extra_settings = _default_extra_settings
    samplerate = None
    blocksize = portaudiolib.paFramesPerBufferUnspecified
    clip_off = False
    dither_off = False
    never_drop_input = False
    prime_output_buffers_using_stream_callback = False

    def __init__(self):
        # Wrap configurations in InputOutputPairs dynamically
        for attr in self._pairs:
            vars(self)[attr] = InputOutputPair(self, '_default_' + attr)

    def __setattr__(self, name, value):
        # Route attribute writing to target proxy items or internal dictionary storage
        if name in self._pairs: getattr(self, name)._pair[:] = split(value)
        elif name in dir(self) and name != 'reset': object.__setattr__(self, name, value)
        else: raise AttributeError(f"'{type(self).__name__}' object has no attribute '{repr(name)}'")

    @property
    def _default_device(self):
        # Dynamically retrieve host OS default interfaces
        return (portaudiolib.Pa_GetDefaultInputDevice(), portaudiolib.Pa_GetDefaultOutputDevice())

    @property
    def hostapi(self):
        return check(portaudiolib.Pa_GetDefaultHostApi())

    def reset(self):
        vars(self).clear()
        self.__init__()

# Instantiate singleton unless running inside a fake test execution ecosystem
if not hasattr(ffi, 'I_AM_FAKE'):
    _default_instance = default()
    default = _default_instance

class PortAudioError(Exception):
    """Exception raised for errors returned from PortAudio API operations."""
    def __init__(self, *args):
        super().__init__(*args)

class CallbackStop(Exception):
    """Exception to cleanly stop audio stream loop inside python callbacks."""
    def __init__(self, *args):
        super().__init__(*args)

class CallbackAbort(Exception):
    """Exception to immediately terminate audio stream processing queues."""
    def __init__(self, *args):
        super().__init__(*args)

class AsioSettings:
    """ASIO driver-specific host settings."""

    def __init__(self, channel_selectors):
        if isinstance(channel_selectors, int): raise TypeError("channel_selectors must be an iterable/list of integers, not a single integer.")

        # Instantiate continuous C memory space variables via FFI
        self._selectors = ffi.new('int[]', channel_selectors)
        self._streaminfo = ffi.new('PaAsioStreamInfo*', dict(size=ffi.sizeof('PaAsioStreamInfo'), hostApiType=portaudiolib.paASIO, version=1, flags=portaudiolib.paAsioUseChannelSelectors, channelSelectors=self._selectors))

class CoreAudioSettings:
    """CoreAudio-specific host settings (macOS)."""

    def __init__(self, channel_map=None, change_device_parameters=False, fail_if_conversion_required=False, conversion_quality='max'):
        """
        Initialize CoreAudio settings.

        Args:
            channel_map (list of int): Custom hardware channel remapping sequence.
            change_device_parameters (bool): Permission flag to mutate OS hardware parameters.
            fail_if_conversion_required (bool): Assert strict match without backend sampling resampling.
            conversion_quality (str): Quality configuration mapping ('min' to 'max').
        """

        conversion_dict = {
            'min': portaudiolib.paMacCoreConversionQualityMin,
            'low': portaudiolib.paMacCoreConversionQualityLow,
            'medium': portaudiolib.paMacCoreConversionQualityMedium,
            'high': portaudiolib.paMacCoreConversionQualityHigh,
            'max': portaudiolib.paMacCoreConversionQualityMax,
        }

        if isinstance(channel_map, int): raise TypeError("channel_map must be an iterable/list of integers, not a single integer.")

        try:
            self._flags = conversion_dict[conversion_quality.lower()]
        except (KeyError, AttributeError):
            raise ValueError(f"Invalid conversion quality. Must be one of: {repr(list(conversion_dict))}")

        if change_device_parameters: self._flags |= portaudiolib.paMacCoreChangeDeviceParameters
        if fail_if_conversion_required: self._flags |= portaudiolib.paMacCoreFailIfConversionRequired

        self._streaminfo = ffi.new('PaMacCoreStreamInfo*')
        portaudiolib.PaMacCore_SetupStreamInfo(self._streaminfo, self._flags)

        if channel_map is not None:
            self._channel_map = ffi.new('SInt32[]', channel_map)
            if len(self._channel_map) == 0: raise TypeError
            portaudiolib.PaMacCore_SetupChannelMap(self._streaminfo, self._channel_map, len(self._channel_map))

class WasapiSettings:
    """WASAPI-specific host settings (Windows)."""

    def __init__(self, exclusive=False, auto_convert=False, explicit_sample_format=False):
        """
        Initialize WASAPI configurations.

        Args:
            exclusive (bool): Request exclusive hardware audio access panel locks.
            auto_convert (bool): Allow auto fallback formatting standardizations.
            explicit_sample_format (bool): Skip automatic system channel conversion pipelines.
        """

        flags = 0x0
        if exclusive: flags |= portaudiolib.paWinWasapiExclusive
        if auto_convert: flags |= portaudiolib.paWinWasapiAutoConvert
        if explicit_sample_format: flags |= portaudiolib.paWinWasapiExplicitSampleFormat

        self._streaminfo = ffi.new('PaWasapiStreamInfo*', dict(size=ffi.sizeof('PaWasapiStreamInfo'), hostApiType=portaudiolib.paWASAPI, version=1, flags=flags,))

class CallbackContext:
    """Context manager and data processor for handling audio stream callbacks."""

    frame = 0
    frames = 0
    input_channels = output_channels = None
    input_dtype = output_dtype = None
    silent_channels = None

    def __init__(self, loop=False):
        """
        Initialize the callback context.

        Args:
            loop (bool): Whether to loop the playback data continuously.
        """

        self.loop = loop
        self.event = threading.Event()
        self.status = CallbackFlags()

    def check_data(self, data, mapping, device):
        """
        Validate and prepare output playback data, channels, and custom mapping.

        Args:
            data (array_like): The input audio data buffer.
            mapping (list of int): Custom output channel indices mapping.
            device (int or str): Target output audio device identifier.

        Returns:
            int: The total number of frames in the provided audio data.
        """

        data = np.asarray(data)

        # Enforce audio data matrix constraints
        if data.ndim < 2: data = data.reshape(-1, 1)
        elif data.ndim > 2: raise ValueError("Audio data matrix must be 1D or 2D.")

        frames, channels = data.shape
        dtype = check_dtype(data.dtype)

        mapping_is_explicit = mapping is not None
        mapping, channels = check_mapping(mapping, channels)

        # Verify that explicit matrix column count matches target mappings
        if data.shape[1] == 1: pass
        elif data.shape[1] != len(mapping): raise ValueError(f"Data column count ({data.shape[1]}) mismatch with mapping size ({len(mapping)}).")

        # Fallback handling for default mono array matching stereo devices
        if (mapping_is_explicit and np.array_equal(mapping, [0]) and query_devices(device, 'output')['max_output_channels'] >= 2): channels = 2

        # Extract channels that should be padded with silent waveforms
        silent_channels = np.setdiff1d(np.arange(channels), mapping)
        if len(mapping) + len(silent_channels) != channels: raise ValueError("Total active and silent channels do not match stream configuration.")

        self.data = data
        self.output_channels = channels
        self.output_dtype = dtype
        self.output_mapping = mapping
        self.silent_channels = silent_channels

        return frames

    def check_out(self, out, frames, channels, dtype, mapping):
        """
        Validate or dynamically pre-allocate memory buffers for recording audio.

        Args:
            out (numpy.ndarray or None): Destination output array or None to auto-allocate.
            frames (int or None): Expected total recording frames.
            channels (int or None): Desired input channels.
            dtype (str or numpy.dtype): Target audio structure type.
            mapping (list of int): Custom input channel mappings.

        Returns:
            tuple: A tuple containing the destination array (numpy.ndarray) and total frames (int).
        """

        if out is None:
            if frames is None: raise TypeError("The 'frames' parameter must be provided if 'out' is None.")
            if channels is None: channels = default.channels['input']

            if channels is None:
                if mapping is None: raise TypeError("Either 'channels' or 'mapping' must be provided to determine the layout.")
                else: channels = len(np.atleast_1d(mapping))

            if dtype is None: dtype = default.dtype['input']

            try:
                # Pre-allocate continuous contiguous C-buffer memory array blocks
                out = np.empty((frames, channels), dtype, order='C')
            except TypeError as e:
                if not isinstance(frames, Integral): raise TypeError("The 'frames' parameter must be an integer.")
                if not isinstance(channels, Integral): raise TypeError("The 'channels' parameter must be an integer.")
                raise e
        else:
            frames, channels = out.shape
            dtype = out.dtype

        dtype = check_dtype(dtype)
        mapping, channels = check_mapping(mapping, channels)
        if out.shape[1] != len(mapping): raise ValueError(f"Output array column count ({out.shape[1]}) does not match mapping layout size ({len(mapping)}).")

        self.out = out
        self.input_channels = channels
        self.input_dtype = dtype
        self.input_mapping = mapping

        return out, frames

    def callback_enter(self, status, data):
        """
        Trigger entry procedures upon receiving stream callback events.

        Args:
            status (CallbackFlags): Hardware status flags.
            data (buffer): The raw buffer payload context.
        """

        self.status |= status
        # Prevent accessing more index blocks than available remaining sizes
        self.blocksize = min(self.frames - self.frame, len(data))

    def read_indata(self, indata):
        """
        Map incoming hardware recording frames directly into the target output array.

        Args:
            indata (numpy.ndarray): Audio matrix captured from the hardware.
        """

        for target, source in enumerate(self.input_mapping):
            self.out[self.frame:self.frame + self.blocksize, target] = indata[:self.blocksize, source]

    def write_outdata(self, outdata):
        """
        Write user audio context structures into physical hardware output data frames.

        Args:
            outdata (numpy.ndarray): Target output sound array to fill.
        """

        outdata[:self.blocksize, self.output_mapping] = self.data[self.frame:self.frame + self.blocksize]
        outdata[:self.blocksize, self.silent_channels] = 0

        # Recursively loop back the play block pointer if file looping is toggled
        if self.loop and self.blocksize < len(outdata):
            self.frame = 0
            outdata = outdata[self.blocksize:]

            self.blocksize = min(self.frames, len(outdata))
            self.write_outdata(outdata)
        else:
            outdata[self.blocksize:] = 0

    def callback_exit(self):
        """Manage loop state iterations on exit; terminates execution on zero blocks."""

        if not self.blocksize: raise CallbackAbort("Stream processing loop complete; no further frames remaining.")
        self.frame += self.blocksize

    def finished_callback(self):
        """Release streaming lock contexts safely upon execution termination."""

        self.event.set()

        # Safely clean up local references to clear out memory scopes
        with contextlib.suppress(AttributeError):
            del self.data

        with contextlib.suppress(AttributeError):
            del self.out

        self.stream._callback = None
        self.stream._finished_callback = None

    def start_stream(self, StreamClass, samplerate, channels, dtype, callback, blocking, **kwargs):
        """
        Instantiate and execute a target hardware streaming class pipeline.

        Args:
            StreamClass (type): Class instance type used for stream instantiation.
            samplerate (float): System sampling rate target.
            channels (int): Channel count constraints.
            dtype (str or numpy.dtype): Target audio structure type.
            callback (callable): Runtime event processing block function.
            blocking (bool): Blocks thread operations until playback has finished.
        """

        stop()
        self.stream = StreamClass(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback, finished_callback=self.finished_callback, **kwargs)
        self.stream.start()

        global last_callback
        last_callback = self
        if blocking: self.wait()

    def wait(self, ignore_errors=True):
        """
        Block thread operations waiting for the signal notification lock.

        Args:
            ignore_errors (bool): Prevents raising errors on stream teardown.

        Returns:
            CallbackFlags or None: Hardware status log updates or None.
        """
        try:
            self.event.wait()
        finally:
            self.stream.close(ignore_errors)

        return self.status if self.status else None

def ffi_string(cdata):
    """Safely cast raw C data strings via FFI translation scopes."""

    return ffi.string(cdata)

def remove_self(d):
    """
    Generate a clean shallow copy dictionary without local 'self' reference keys.

    Args:
        d (dict): Local variable dictionaries via `locals()`.

    Returns:
        dict: A mutated variable block structure map.
    """

    d = d.copy()
    del d['self']

    return d

def check_mapping(mapping, channels):
    """
    Validate and align target device routing map index arrays.

    Args:
        mapping (list, array or None): Input/Output channel layout structure arrays.
        channels (int): System base reference channel context size.

    Returns:
        tuple: A tuple containing the parsed mapping array (numpy.ndarray) and channel count (int).
    """

    if mapping is None: mapping = np.arange(channels)
    else:
        mapping = np.array(mapping, copy=True)
        mapping = np.atleast_1d(mapping)

        # 1-based index validation check for safety
        if mapping.min() < 1: raise ValueError("Channel mapping indices must be 1-based (greater than or equal to 1).")

        channels = mapping.max()
        mapping -= 1 # Convert to 0-based index for internal numpy access

    return mapping, channels

def check_dtype(dtype):
    """
    Verify system-specific formatting target matches supported sound device ranges.

    Args:
        dtype (str or numpy.dtype): Data type to check.

    Returns:
        str: Validated numpy data type name string.
    """

    dtype = np.dtype(dtype).name

    if dtype in sampleformats: pass
    elif dtype == 'float64': dtype = np.dtype(np.float32).name # Cast double-precision to standard floats
    else: raise TypeError(f"Unsupported audio data type structure: {dtype}")

    return dtype

def get_stream_parameters(kind, device, channels, dtype, latency, extra_settings, samplerate):
    """Generate structured CFFI parameter pointers matching specific hardware pipeline options.

    Args:
        kind (str): Stream type direction ('input' or 'output').
        device (int or str or None): Audio target index identifier.
        channels (int or None): Channel allocation maps.
        dtype (str or None): Targeted layout types.
        latency (float or str or None): System hardware speed presets.
        extra_settings (object or None): Custom API specific stream contexts.
        samplerate (float or None): Baseline clock speed targets.

    Returns:
        tuple: (PaStreamParameters pointer, datatype name, sample size, sample rate)
    """

    assert kind in ('input', 'output')
    # Resolve default fallbacks if properties match None state parameters
    if device is None: device = default.device
    device = get_device_id(device, kind, raise_on_error=True)

    if channels is None: channels = default.channels
    channels = select_input_or_output(channels, kind)

    if dtype is None: dtype = default.dtype
    dtype = select_input_or_output(dtype, kind)

    if latency is None: latency = default.latency
    latency = select_input_or_output(latency, kind)

    if extra_settings is None: extra_settings = default.extra_settings
    extra_settings = select_input_or_output(extra_settings, kind)

    if samplerate is None: samplerate = default.samplerate
    info = query_devices(device)

    if channels is None: channels = info['max_' + kind + '_channels']

    try:
        dtype = sys.modules['numpy'].dtype(dtype).name
    except Exception:
        pass

    try:
        sampleformat = sampleformats[dtype]
    except KeyError as e:
        raise ValueError(f"The specified data type format '{dtype}' is invalid or unmapped.")

    samplesize = check(portaudiolib.Pa_GetSampleSize(sampleformat))

    if latency in ('low', 'high'): latency = info['default_' + latency + '_' + kind + '_latency']
    if samplerate is None: samplerate = info['default_samplerate']

    parameters = ffi.new('PaStreamParameters*', (device, channels, sampleformat, latency, extra_settings._streaminfo if extra_settings else ffi.NULL))
    return parameters, dtype, samplesize, samplerate

def _wrap_callback(callback, *args):
    """Internal proxy callback to process wrapper conversions and map Python returns into C codes."""

    args = args[:-1] + (CallbackFlags(args[-1]),)

    try:
        callback(*args)
    except CallbackStop:
        return portaudiolib.paComplete
    except CallbackAbort:
        return portaudiolib.paAbort

    return portaudiolib.paContinue

def mbuffer(ptr, frames, channels, samplesize):
    """Directly wrap explicit pointers safely into an exposed memory viewing panel."""

    return ffi.buffer(ptr, frames * channels * samplesize)

def array(buffer, channels, dtype):
    """Convert flat raw buffer data structures back into formatted 2D multidimensional matrices."""

    data = np.frombuffer(buffer, dtype=dtype)
    data.shape = -1, channels
    return data

def split(value):
    """
    Split unified pair parameters down evenly into directional component parameters.

    Args:
        value (object): Single target configuration entry or a sequence list pair block.

    Returns:
        tuple: Formatted pair values representing (input_value, output_value).
    """

    if isinstance(value, (str, bytes)): return value, value

    try:
        invalue, outvalue = value
    except TypeError:
        invalue = outvalue = value
    except ValueError:
        raise ValueError("Sequence unpacking failed. Target value must contain exactly 2 components.")

    return invalue, outvalue

def check(err, msg=''):
    """Assert PortAudio result code statuses, tracking errors dynamically to throw custom Exceptions.

    Args:
        err (int): Error return code flag generated by the dynamic C library pipeline.
        msg (str): Append error context traces.

    Returns:
        int: The validated error code if greater than or equal to zero.
    """

    if err >= 0: return err

    errormsg = ffi_string(portaudiolib.Pa_GetErrorText(err)).decode()
    if msg: errormsg = f'{msg}: {errormsg}'

    # Check for underlying unanticipated driver or OS layer exceptions
    if err == portaudiolib.paUnanticipatedHostError:
        info = portaudiolib.Pa_GetLastHostErrorInfo()
        host_api = portaudiolib.Pa_HostApiTypeIdToHostApiIndex(info.hostApiType)

        hosterror_text = ffi_string(info.errorText).decode()
        hosterror_info = host_api, info.errorCode, hosterror_text

        raise PortAudioError(errormsg, err, hosterror_info)
    raise PortAudioError(errormsg, err)

def select_input_or_output(value_or_pair, kind):
    """Pick correct routing fields from combined value/pair sets based on device path directions."""

    ivalue, ovalue = split(value_or_pair)

    if kind == 'input': return ivalue
    elif kind == 'output': return ovalue

    assert False

def get_device_id(id_or_query_string, kind, raise_on_error=False):
    """
    Locate unique hardware target IDs dynamically parsing indices or string search query patterns.

    Args:
        id_or_query_string (int or str): Device index identifier or partial search string name match.
        kind (str or None): Targeted pathway validation format filter ('input' or 'output').
        raise_on_error (bool): Flag forcing explicit lookup tracking to crash on failure states.

    Returns:
        int: The resolved integer identifier index for the targeted physical hardware device.
    """

    assert kind in ('input', 'output', None)

    if id_or_query_string is None: id_or_query_string = default.device
    idev, odev = split(id_or_query_string)

    if kind == 'input': id_or_query_string = idev
    elif kind == 'output': id_or_query_string = odev
    else:
        if idev == odev: id_or_query_string = idev
        else: raise ValueError(id_or_query_string)

    if isinstance(id_or_query_string, int): return id_or_query_string
    device_list, matches, exact_device_matches = [], [], []

    # Map down a clean list of candidate sound card systems running on the host OS
    for id, info in enumerate(query_devices()):
        if not kind or info['max_' + kind + '_channels'] > 0:
            hostapi_info = query_hostapis(info['hostapi'])
            device_list.append((id, info['name'], hostapi_info['name']))

    query_string = id_or_query_string.lower()
    substrings = query_string.split()

    # Search for matching text blocks across device and host API names
    for id, device_string, hostapi_string in device_list:
        full_string = device_string + ', ' + hostapi_string
        pos = 0

        for substring in substrings:
            pos = full_string.lower().find(substring, pos)
            if pos < 0: break
            pos += len(substring)
        else:
            matches.append((id, full_string))
            if query_string in [device_string.lower(), full_string.lower()]: exact_device_matches.append(id)

    if kind is None: kind = 'input/output'
    if not matches:
        if raise_on_error: raise ValueError(f"No matching {kind} device found for query string: {repr(id_or_query_string)}")
        else: return -1

    if len(matches) > 1:
        if len(exact_device_matches) == 1: return exact_device_matches[0]

        if raise_on_error: raise ValueError(f"Multiple {kind} device matches found for '{id_or_query_string}':\n" +'\n'.join(f'[{id}] {name}' for id, name in matches))
        else: return -1

    return matches[0][0]

def initialize():
    """Boot standard runtime dependencies inside PortAudio environments safely silencing error tracks."""

    old_stderr = None

    try:
        # Duplicate standard error file descriptors to silence noisy default hardware terminal warning texts
        old_stderr = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
    except OSError:
        pass

    try:
        check(portaudiolib.Pa_Initialize())
        global initialized
        initialized += 1
    finally:
        if old_stderr is not None:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)

def terminate():
    """Unload internal runtime modules shutting down backend audio driver hooks cleanly."""

    global initialized

    check(portaudiolib.Pa_Terminate())
    initialized -= 1


def exit_handler():
    """Global system process callback handling script exit hooks to close active streaming tracks."""

    assert initialized >= 0

    if last_callback:
        last_callback.stream.stop()
        last_callback.stream.close()

    while initialized:
        terminate()

# Register cleanup routines to run at program termination
atexit.register(exit_handler)
initialize()