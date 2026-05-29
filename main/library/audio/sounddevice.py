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

ffi = _cffi_backend.FFI('portaudio',
    _version = 0x2601,
    _types = b'\x00\x00\x76\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x79\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x1C\x0D\x00\x00\x8D\x03\x00\x00\x00\x0F\x00\x00\x7B\x0D\x00\x00\x00\x0F\x00\x00\x80\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x01\x01\x00\x00\x00\x0F\x00\x00\x88\x0D\x00\x00\x00\x0F\x00\x00\x21\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x01\x0B\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x82\x03\x00\x00\x1F\x11\x00\x00\x0E\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x03\x00\x00\x1F\x11\x00\x00\x1F\x11\x00\x00\x0E\x01\x00\x00\x0A\x01\x00\x00\x0A\x01\x00\x00\x52\x03\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x2E\x11\x00\x00\x07\x01\x00\x00\x07\x01\x00\x00\x0A\x01\x00\x00\x0E\x01\x00\x00\x0A\x01\x00\x00\x34\x11\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x07\x11\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x8D\x03\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x07\x11\x00\x00\x6B\x03\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x4B\x11\x00\x00\x07\x11\x00\x00\x0A\x01\x00\x00\x7F\x03\x00\x00\x0A\x01\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x0D\x00\x00\x00\x0F\x00\x00\x69\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x7D\x03\x00\x00\x8B\x03\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x60\x11\x00\x00\x0A\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x09\x01\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x8D\x0D\x00\x00\x07\x11\x00\x00\x09\x01\x00\x00\x07\x11\x00\x00\x09\x01\x00\x00\x07\x11\x00\x00\x00\x0F\x00\x00\x01\x09\x00\x00\x77\x03\x00\x00\x02\x09\x00\x00\x00\x0B\x00\x00\x7A\x03\x00\x00\x03\x09\x00\x00\x7C\x03\x00\x00\x04\x09\x00\x00\x00\x09\x00\x00\x02\x0B\x00\x00\x05\x09\x00\x00\x81\x03\x00\x00\x06\x09\x00\x00\x07\x09\x00\x00\x03\x0B\x00\x00\x04\x0B\x00\x00\x08\x09\x00\x00\x05\x0B\x00\x00\x06\x0B\x00\x00\x89\x03\x00\x00\x02\x01\x00\x00\x01\x03\x00\x00\x15\x01\x00\x00\x6E\x03\x00\x00\x00\x01',
    _globals = (b'\x00\x00\x11\x23PaMacCore_GetChannelName',0,b'\x00\x00\x5F\x23PaMacCore_SetupChannelMap',0,b'\x00\x00\x64\x23PaMacCore_SetupStreamInfo',0,b'\x00\x00\x23\x23PaWasapi_IsLoopback',0,b'\x00\x00\x5A\x23PaWasapi_UpdateDeviceList',0,b'\x00\x00\x41\x23Pa_AbortStream',0,b'\x00\x00\x41\x23Pa_CloseStream',0,b'\x00\x00\x5A\x23Pa_GetDefaultHostApi',0,b'\x00\x00\x5A\x23Pa_GetDefaultInputDevice',0,b'\x00\x00\x5A\x23Pa_GetDefaultOutputDevice',0,b'\x00\x00\x5A\x23Pa_GetDeviceCount',0,b'\x00\x00\x00\x23Pa_GetDeviceInfo',0,b'\x00\x00\x0E\x23Pa_GetErrorText',0,b'\x00\x00\x5A\x23Pa_GetHostApiCount',0,b'\x00\x00\x03\x23Pa_GetHostApiInfo',0,b'\x00\x00\x09\x23Pa_GetLastHostErrorInfo',0,b'\x00\x00\x2A\x23Pa_GetSampleSize',0,b'\x00\x00\x18\x23Pa_GetStreamCpuLoad',0,b'\x00\x00\x06\x23Pa_GetStreamHostApiType',0,b'\x00\x00\x0B\x23Pa_GetStreamInfo',0,b'\x00\x00\x5C\x23Pa_GetStreamReadAvailable',0,b'\x00\x00\x18\x23Pa_GetStreamTime',0,b'\x00\x00\x5C\x23Pa_GetStreamWriteAvailable',0,b'\x00\x00\x5A\x23Pa_GetVersion',0,b'\x00\x00\x16\x23Pa_GetVersionText',0,b'\x00\x00\x26\x23Pa_HostApiDeviceIndexToDeviceIndex',0,b'\x00\x00\x1B\x23Pa_HostApiTypeIdToHostApiIndex',0,b'\x00\x00\x5A\x23Pa_Initialize',0,b'\x00\x00\x1E\x23Pa_IsFormatSupported',0,b'\x00\x00\x41\x23Pa_IsStreamActive',0,b'\x00\x00\x41\x23Pa_IsStreamStopped',0,b'\x00\x00\x37\x23Pa_OpenDefaultStream',0,b'\x00\x00\x2D\x23Pa_OpenStream',0,b'\x00\x00\x44\x23Pa_ReadStream',0,b'\x00\x00\x4E\x23Pa_SetStreamFinishedCallback',0,b'\x00\x00\x68\x23Pa_Sleep',0,b'\x00\x00\x41\x23Pa_StartStream',0,b'\x00\x00\x41\x23Pa_StopStream',0,b'\x00\x00\x5A\x23Pa_Terminate',0,b'\x00\x00\x49\x23Pa_WriteStream',0,b'\xFF\xFF\xFF\x0BeAudioCategoryAlerts',4,b'\xFF\xFF\xFF\x0BeAudioCategoryCommunications',3,b'\xFF\xFF\xFF\x0BeAudioCategoryGameChat',8,b'\xFF\xFF\xFF\x0BeAudioCategoryGameEffects',6,b'\xFF\xFF\xFF\x0BeAudioCategoryGameMedia',7,b'\xFF\xFF\xFF\x0BeAudioCategoryMedia',11,b'\xFF\xFF\xFF\x0BeAudioCategoryMovie',10,b'\xFF\xFF\xFF\x0BeAudioCategoryOther',0,b'\xFF\xFF\xFF\x0BeAudioCategorySoundEffects',5,b'\xFF\xFF\xFF\x0BeAudioCategorySpeech',9,b'\xFF\xFF\xFF\x0BeStreamOptionMatchFormat',2,b'\xFF\xFF\xFF\x0BeStreamOptionNone',0,b'\xFF\xFF\xFF\x0BeStreamOptionRaw',1,b'\xFF\xFF\xFF\x0BeThreadPriorityAudio',1,b'\xFF\xFF\xFF\x0BeThreadPriorityCapture',2,b'\xFF\xFF\xFF\x0BeThreadPriorityDistribution',3,b'\xFF\xFF\xFF\x0BeThreadPriorityGames',4,b'\xFF\xFF\xFF\x0BeThreadPriorityNone',0,b'\xFF\xFF\xFF\x0BeThreadPriorityPlayback',5,b'\xFF\xFF\xFF\x0BeThreadPriorityProAudio',6,b'\xFF\xFF\xFF\x0BeThreadPriorityWindowManager',7,b'\xFF\xFF\xFF\x0BpaAL',9,b'\xFF\xFF\xFF\x0BpaALSA',8,b'\xFF\xFF\xFF\x0BpaASIO',3,b'\xFF\xFF\xFF\x0BpaAbort',2,b'\xFF\xFF\xFF\x1FpaAsioUseChannelSelectors',1,b'\xFF\xFF\xFF\x0BpaAudioScienceHPI',14,b'\xFF\xFF\xFF\x0BpaBadBufferPtr',-9972,b'\xFF\xFF\xFF\x0BpaBadIODeviceCombination',-9993,b'\xFF\xFF\xFF\x0BpaBadStreamPtr',-9988,b'\xFF\xFF\xFF\x0BpaBeOS',10,b'\xFF\xFF\xFF\x0BpaBufferTooBig',-9991,b'\xFF\xFF\xFF\x0BpaBufferTooSmall',-9990,b'\xFF\xFF\xFF\x0BpaCanNotReadFromACallbackStream',-9977,b'\xFF\xFF\xFF\x0BpaCanNotReadFromAnOutputOnlyStream',-9975,b'\xFF\xFF\xFF\x0BpaCanNotWriteToACallbackStream',-9976,b'\xFF\xFF\xFF\x0BpaCanNotWriteToAnInputOnlyStream',-9974,b'\xFF\xFF\xFF\x1FpaClipOff',1,b'\xFF\xFF\xFF\x0BpaComplete',1,b'\xFF\xFF\xFF\x0BpaContinue',0,b'\xFF\xFF\xFF\x0BpaCoreAudio',5,b'\xFF\xFF\xFF\x1FpaCustomFormat',65536,b'\xFF\xFF\xFF\x0BpaDeviceUnavailable',-9985,b'\xFF\xFF\xFF\x0BpaDirectSound',1,b'\xFF\xFF\xFF\x1FpaDitherOff',2,b'\xFF\xFF\xFF\x1FpaFloat32',1,b'\xFF\xFF\xFF\x1FpaFormatIsSupported',0,b'\xFF\xFF\xFF\x1FpaFramesPerBufferUnspecified',0,b'\xFF\xFF\xFF\x0BpaHostApiNotFound',-9979,b'\xFF\xFF\xFF\x0BpaInDevelopment',0,b'\xFF\xFF\xFF\x0BpaIncompatibleHostApiSpecificStreamInfo',-9984,b'\xFF\xFF\xFF\x0BpaIncompatibleStreamHostApi',-9973,b'\xFF\xFF\xFF\x1FpaInputOverflow',2,b'\xFF\xFF\xFF\x0BpaInputOverflowed',-9981,b'\xFF\xFF\xFF\x1FpaInputUnderflow',1,b'\xFF\xFF\xFF\x0BpaInsufficientMemory',-9992,b'\xFF\xFF\xFF\x1FpaInt16',8,b'\xFF\xFF\xFF\x1FpaInt24',4,b'\xFF\xFF\xFF\x1FpaInt32',2,b'\xFF\xFF\xFF\x1FpaInt8',16,b'\xFF\xFF\xFF\x0BpaInternalError',-9986,b'\xFF\xFF\xFF\x0BpaInvalidChannelCount',-9998,b'\xFF\xFF\xFF\x0BpaInvalidDevice',-9996,b'\xFF\xFF\xFF\x0BpaInvalidFlag',-9995,b'\xFF\xFF\xFF\x0BpaInvalidHostApi',-9978,b'\xFF\xFF\xFF\x0BpaInvalidSampleRate',-9997,b'\xFF\xFF\xFF\x0BpaJACK',12,b'\xFF\xFF\xFF\x0BpaMME',2,b'\xFF\xFF\xFF\x1FpaMacCoreChangeDeviceParameters',1,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityHigh',1024,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityLow',768,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMax',0,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMedium',512,b'\xFF\xFF\xFF\x1FpaMacCoreConversionQualityMin',256,b'\xFF\xFF\xFF\x1FpaMacCoreFailIfConversionRequired',2,b'\xFF\xFF\xFF\x1FpaMacCoreMinimizeCPU',257,b'\xFF\xFF\xFF\x1FpaMacCoreMinimizeCPUButPlayNice',256,b'\xFF\xFF\xFF\x1FpaMacCorePlayNice',0,b'\xFF\xFF\xFF\x1FpaMacCorePro',1,b'\xFF\xFF\xFF\x1FpaNeverDropInput',4,b'\xFF\xFF\xFF\x1FpaNoDevice',-1,b'\xFF\xFF\xFF\x0BpaNoError',0,b'\xFF\xFF\xFF\x1FpaNoFlag',0,b'\xFF\xFF\xFF\x1FpaNonInterleaved',2147483648,b'\xFF\xFF\xFF\x0BpaNotInitialized',-10000,b'\xFF\xFF\xFF\x0BpaNullCallback',-9989,b'\xFF\xFF\xFF\x0BpaOSS',7,b'\xFF\xFF\xFF\x1FpaOutputOverflow',8,b'\xFF\xFF\xFF\x1FpaOutputUnderflow',4,b'\xFF\xFF\xFF\x0BpaOutputUnderflowed',-9980,b'\xFF\xFF\xFF\x1FpaPlatformSpecificFlags',4294901760,b'\xFF\xFF\xFF\x1FpaPrimeOutputBuffersUsingStreamCallback',8,b'\xFF\xFF\xFF\x1FpaPrimingOutput',16,b'\xFF\xFF\xFF\x0BpaSampleFormatNotSupported',-9994,b'\xFF\xFF\xFF\x0BpaSoundManager',4,b'\xFF\xFF\xFF\x0BpaStreamIsNotStopped',-9982,b'\xFF\xFF\xFF\x0BpaStreamIsStopped',-9983,b'\xFF\xFF\xFF\x0BpaTimedOut',-9987,b'\xFF\xFF\xFF\x1FpaUInt8',32,b'\xFF\xFF\xFF\x0BpaUnanticipatedHostError',-9999,b'\xFF\xFF\xFF\x1FpaUseHostApiSpecificDeviceSpecification',-2,b'\xFF\xFF\xFF\x0BpaWASAPI',13,b'\xFF\xFF\xFF\x0BpaWDMKS',11,b'\xFF\xFF\xFF\x0BpaWinWasapiAutoConvert',64,b'\xFF\xFF\xFF\x0BpaWinWasapiExclusive',1,b'\xFF\xFF\xFF\x0BpaWinWasapiExplicitSampleFormat',32,b'\xFF\xFF\xFF\x0BpaWinWasapiPolling',8,b'\xFF\xFF\xFF\x0BpaWinWasapiRedirectHostProcessor',2,b'\xFF\xFF\xFF\x0BpaWinWasapiThreadPriority',16,b'\xFF\xFF\xFF\x0BpaWinWasapiUseChannelMask',4),
    _struct_unions = ((b'\x00\x00\x00\x7D\x00\x00\x00\x02$PaMacCoreStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x61\x11channelMap',b'\x00\x00\x2B\x11channelMapSize'),(b'\x00\x00\x00\x75\x00\x00\x00\x02PaAsioStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x8A\x11channelSelectors'),(b'\x00\x00\x00\x77\x00\x00\x00\x02PaDeviceInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x88\x11name',b'\x00\x00\x01\x11hostApi',b'\x00\x00\x01\x11maxInputChannels',b'\x00\x00\x01\x11maxOutputChannels',b'\x00\x00\x21\x11defaultLowInputLatency',b'\x00\x00\x21\x11defaultLowOutputLatency',b'\x00\x00\x21\x11defaultHighInputLatency',b'\x00\x00\x21\x11defaultHighOutputLatency',b'\x00\x00\x21\x11defaultSampleRate'),(b'\x00\x00\x00\x7A\x00\x00\x00\x02PaHostApiInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x1C\x11type',b'\x00\x00\x88\x11name',b'\x00\x00\x01\x11deviceCount',b'\x00\x00\x01\x11defaultInputDevice',b'\x00\x00\x01\x11defaultOutputDevice'),(b'\x00\x00\x00\x7C\x00\x00\x00\x02PaHostErrorInfo',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x69\x11errorCode',b'\x00\x00\x88\x11errorText'),(b'\x00\x00\x00\x7F\x00\x00\x00\x02PaStreamCallbackTimeInfo',b'\x00\x00\x21\x11inputBufferAdcTime',b'\x00\x00\x21\x11currentTime',b'\x00\x00\x21\x11outputBufferDacTime'),(b'\x00\x00\x00\x81\x00\x00\x00\x02PaStreamInfo',b'\x00\x00\x01\x11structVersion',b'\x00\x00\x21\x11inputLatency',b'\x00\x00\x21\x11outputLatency',b'\x00\x00\x21\x11sampleRate'),(b'\x00\x00\x00\x82\x00\x00\x00\x02PaStreamParameters',b'\x00\x00\x01\x11device',b'\x00\x00\x01\x11channelCount',b'\x00\x00\x2B\x11sampleFormat',b'\x00\x00\x21\x11suggestedLatency',b'\x00\x00\x07\x11hostApiSpecificStreamInfo'),(b'\x00\x00\x00\x85\x00\x00\x00\x02PaWasapiStreamInfo',b'\x00\x00\x2B\x11size',b'\x00\x00\x1C\x11hostApiType',b'\x00\x00\x2B\x11version',b'\x00\x00\x2B\x11flags',b'\x00\x00\x2B\x11channelMask',b'\x00\x00\x8C\x11hostProcessorOutput',b'\x00\x00\x8C\x11hostProcessorInput',b'\x00\x00\x87\x11threadPriority',b'\x00\x00\x84\x11streamCategory',b'\x00\x00\x86\x11streamOption')),
    _enums = (b'\x00\x00\x00\x78\x00\x00\x00\x15PaErrorCode\x00paNoError,paNotInitialized,paUnanticipatedHostError,paInvalidChannelCount,paInvalidSampleRate,paInvalidDevice,paInvalidFlag,paSampleFormatNotSupported,paBadIODeviceCombination,paInsufficientMemory,paBufferTooBig,paBufferTooSmall,paNullCallback,paBadStreamPtr,paTimedOut,paInternalError,paDeviceUnavailable,paIncompatibleHostApiSpecificStreamInfo,paStreamIsStopped,paStreamIsNotStopped,paInputOverflowed,paOutputUnderflowed,paHostApiNotFound,paInvalidHostApi,paCanNotReadFromACallbackStream,paCanNotWriteToACallbackStream,paCanNotReadFromAnOutputOnlyStream,paCanNotWriteToAnInputOnlyStream,paIncompatibleStreamHostApi,paBadBufferPtr',b'\x00\x00\x00\x1C\x00\x00\x00\x16PaHostApiTypeId\x00paInDevelopment,paDirectSound,paMME,paASIO,paSoundManager,paCoreAudio,paOSS,paALSA,paAL,paBeOS,paWDMKS,paJACK,paWASAPI,paAudioScienceHPI',b'\x00\x00\x00\x7E\x00\x00\x00\x16PaStreamCallbackResult\x00paContinue,paComplete,paAbort',b'\x00\x00\x00\x83\x00\x00\x00\x16PaWasapiFlags\x00paWinWasapiExclusive,paWinWasapiRedirectHostProcessor,paWinWasapiUseChannelMask,paWinWasapiPolling,paWinWasapiThreadPriority,paWinWasapiExplicitSampleFormat,paWinWasapiAutoConvert',b'\x00\x00\x00\x84\x00\x00\x00\x16PaWasapiStreamCategory\x00eAudioCategoryOther,eAudioCategoryCommunications,eAudioCategoryAlerts,eAudioCategorySoundEffects,eAudioCategoryGameEffects,eAudioCategoryGameMedia,eAudioCategoryGameChat,eAudioCategorySpeech,eAudioCategoryMovie,eAudioCategoryMedia',b'\x00\x00\x00\x86\x00\x00\x00\x16PaWasapiStreamOption\x00eStreamOptionNone,eStreamOptionRaw,eStreamOptionMatchFormat',b'\x00\x00\x00\x87\x00\x00\x00\x16PaWasapiThreadPriority\x00eThreadPriorityNone,eThreadPriorityAudio,eThreadPriorityCapture,eThreadPriorityDistribution,eThreadPriorityGames,eThreadPriorityPlayback,eThreadPriorityProAudio,eThreadPriorityWindowManager'),
    _typenames = (b'\x00\x00\x00\x75PaAsioStreamInfo',b'\x00\x00\x00\x01PaDeviceIndex',b'\x00\x00\x00\x77PaDeviceInfo',b'\x00\x00\x00\x01PaError',b'\x00\x00\x00\x78PaErrorCode',b'\x00\x00\x00\x01PaHostApiIndex',b'\x00\x00\x00\x7APaHostApiInfo',b'\x00\x00\x00\x1CPaHostApiTypeId',b'\x00\x00\x00\x7CPaHostErrorInfo',b'\x00\x00\x00\x7DPaMacCoreStreamInfo',b'\x00\x00\x00\x2BPaSampleFormat',b'\x00\x00\x00\x8DPaStream',b'\x00\x00\x00\x52PaStreamCallback',b'\x00\x00\x00\x2BPaStreamCallbackFlags',b'\x00\x00\x00\x7EPaStreamCallbackResult',b'\x00\x00\x00\x7FPaStreamCallbackTimeInfo',b'\x00\x00\x00\x6BPaStreamFinishedCallback',b'\x00\x00\x00\x2BPaStreamFlags',b'\x00\x00\x00\x81PaStreamInfo',b'\x00\x00\x00\x82PaStreamParameters',b'\x00\x00\x00\x21PaTime',b'\x00\x00\x00\x83PaWasapiFlags',b'\x00\x00\x00\x8CPaWasapiHostProcessorCallback',b'\x00\x00\x00\x84PaWasapiStreamCategory',b'\x00\x00\x00\x85PaWasapiStreamInfo',b'\x00\x00\x00\x86PaWasapiStreamOption',b'\x00\x00\x00\x87PaWasapiThreadPriority',b'\x00\x00\x00\x2BPaWinWaveFormatChannelMask',b'\x00\x00\x00\x8BSInt32'),
)

try:
    for libname in ('portaudio', 'bin\\libportaudio-2.dll', 'lib/libportaudio.dylib'):
        libname = find_library(libname)
        if libname is not None: break
    else: raise OSError(translations['portaudio_not_found'])

    portaudiolib = ffi.dlopen(libname)
except OSError:
    if platform.system() == 'Darwin': libname = 'libportaudio.dylib'
    elif platform.system() == 'Windows':
        if platform.machine().lower() in ('arm64', 'aarch64'): platform_suffix = 'arm64'
        else: platform_suffix = platform.architecture()[0]

        libname = 'libportaudio' + platform_suffix + ('-asio.dll' if 'SD_ENABLE_ASIO' in os.environ else '.dll')
    else: raise

    libdir = configs.get("portaudiolib", os.path.join("assets", "binary"))
    libpath = os.path.join(libdir, libname)

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
    ctx = CallbackContext(loop=loop)
    ctx.frames = ctx.check_data(data, mapping, kwargs.get('device'))

    def callback(outdata, frames, time, status):
        assert len(outdata) == frames
        ctx.callback_enter(status, outdata)
        ctx.write_outdata(outdata)
        ctx.callback_exit()

    ctx.start_stream(OutputStream, samplerate, ctx.output_channels, ctx.output_dtype, callback, blocking, prime_output_buffers_using_stream_callback=False, **kwargs)

def rec(frames=None, samplerate=None, channels=None, dtype=None, out=None, mapping=None, blocking=False, **kwargs):
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
    ctx = CallbackContext()
    output_frames = ctx.check_data(data, output_mapping, kwargs.get('device'))

    if dtype is None: dtype = ctx.data.dtype
    out, input_frames = ctx.check_out(out, output_frames, channels, dtype, input_mapping)

    if input_frames != output_frames: raise ValueError
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
    if last_callback: return last_callback.wait(ignore_errors)

def stop(ignore_errors=True):
    if last_callback:
        last_callback.stream.stop(ignore_errors)
        last_callback.stream.close(ignore_errors)

def get_status():
    if last_callback: return last_callback.status
    else: raise RuntimeError

def get_stream():
    if last_callback: return last_callback.stream
    else: raise RuntimeError

def query_devices(device=None, kind=None):
    if kind not in ('input', 'output', None): raise ValueError(f'{kind!r}')
    if device is None and kind is None: return DeviceList(query_devices(i) for i in range(check(portaudiolib.Pa_GetDeviceCount())))

    device = get_device_id(device, kind, raise_on_error=True)
    info = portaudiolib.Pa_GetDeviceInfo(device)

    if not info: raise PortAudioError(device)
    assert info.structVersion == 2
    name_bytes = ffi_string(info.name)

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

    if kind and device_dict['max_' + kind + '_channels'] < 1: raise ValueError(f"{kind} {device_dict['name']}")
    return device_dict

def query_hostapis(index=None):
    if index is None: return tuple(query_hostapis(i) for i in range(check(portaudiolib.Pa_GetHostApiCount())))
    info = portaudiolib.Pa_GetHostApiInfo(index)
    if not info: raise PortAudioError(index)

    assert info.structVersion == 1
    return {
        'name': ffi_string(info.name).decode(),
        'devices': [portaudiolib.Pa_HostApiDeviceIndexToDeviceIndex(index, i) for i in range(info.deviceCount)],
        'default_input_device': info.defaultInputDevice,
        'default_output_device': info.defaultOutputDevice,
    }

def check_input_settings(device=None, channels=None, dtype=None, extra_settings=None, samplerate=None):
    parameters, dtype, _, samplerate = get_stream_parameters('input', device=device, channels=channels, dtype=dtype, latency=None, extra_settings=extra_settings, samplerate=samplerate)
    check(portaudiolib.Pa_IsFormatSupported(parameters, ffi.NULL, samplerate))

def check_output_settings(device=None, channels=None, dtype=None, extra_settings=None, samplerate=None):
    parameters, dtype, _, samplerate = get_stream_parameters('output', device=device, channels=channels, dtype=dtype, latency=None, extra_settings=extra_settings, samplerate=samplerate)
    check(portaudiolib.Pa_IsFormatSupported(ffi.NULL, parameters, samplerate))

def sleep(msec):
    portaudiolib.Pa_Sleep(msec)

def get_portaudio_version():
    return portaudiolib.Pa_GetVersion(), ffi_string(portaudiolib.Pa_GetVersionText()).decode()

class StreamBase:
    def __init__(self, kind, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None, userdata=None, wrap_callback=None):
        assert kind in ('input', 'output', 'duplex')
        assert wrap_callback in ('array', 'buffer', None)
        stream_flags = portaudiolib.paNoFlag

        if blocksize is None: blocksize = default.blocksize
        if clip_off is None: clip_off = default.clip_off
        if dither_off is None: dither_off = default.dither_off
        if never_drop_input is None: never_drop_input = default.never_drop_input
        if prime_output_buffers_using_stream_callback is None: prime_output_buffers_using_stream_callback = default.prime_output_buffers_using_stream_callback
        if clip_off: stream_flags |= portaudiolib.paClipOff
        if dither_off: stream_flags |= portaudiolib.paDitherOff
        if never_drop_input: stream_flags |= portaudiolib.paNeverDropInput
        if prime_output_buffers_using_stream_callback: stream_flags |= portaudiolib.paPrimeOutputBuffersUsingStreamCallback

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
        if not info: raise PortAudioError
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
        if not time: raise PortAudioError
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
        err = portaudiolib.Pa_StartStream(self._ptr)
        if err != portaudiolib.paStreamIsNotStopped: check(err)

    def stop(self, ignore_errors=True):
        err = portaudiolib.Pa_StopStream(self._ptr)
        if not ignore_errors: check(err)

    def abort(self, ignore_errors=True):
        err = portaudiolib.Pa_AbortStream(self._ptr)
        if not ignore_errors: check(err)

    def close(self, ignore_errors=True):
        err = portaudiolib.Pa_CloseStream(self._ptr)
        self._ptr = ffi.NULL
        if not ignore_errors: check(err)

class InputStreamBase(StreamBase):
    @property
    def read_available(self):
        return check(portaudiolib.Pa_GetStreamReadAvailable(self._ptr))

    def _raw_read(self, frames):
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
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='input', wrap_callback='buffer', **remove_self(locals()))

    read = InputStreamBase._raw_read

class _OutputStreamBase(StreamBase):
    @property
    def write_available(self):
        return check(portaudiolib.Pa_GetStreamWriteAvailable(self._ptr))

    def _raw_write(self, data):
        try:
            data = ffi.from_buffer(data)
        except AttributeError:
            pass
        except TypeError:
            pass

        _, samplesize = split(self._samplesize)
        _, channels = split(self._channels)

        samples, remainder = divmod(len(data), samplesize)
        if remainder: raise ValueError

        frames, remainder = divmod(samples, channels)
        if remainder: raise ValueError

        err = portaudiolib.Pa_WriteStream(self._ptr, data, frames)

        if err == portaudiolib.paOutputUnderflowed: underflowed = True
        else:
            check(err)
            underflowed = False

        return underflowed

class RawOutputStream(_OutputStreamBase):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='output', wrap_callback='buffer', **remove_self(locals()))

    write = _OutputStreamBase._raw_write

class RawStream(RawInputStream, RawOutputStream):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='duplex', wrap_callback='buffer', **remove_self(locals()))

class InputStream(InputStreamBase):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='input', wrap_callback='array', **remove_self(locals()))

    def read(self, frames):
        dtype, _ = split(self._dtype)
        channels, _ = split(self._channels)

        data, overflowed = InputStreamBase._raw_read(self, frames)
        data = array(data, channels, dtype)

        return data, overflowed

class OutputStream(_OutputStreamBase):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='output', wrap_callback='array', **remove_self(locals()))

    def write(self, data):
        data = np.asarray(data)
        _, dtype = split(self._dtype)
        _, channels = split(self._channels)

        if data.ndim < 2: data = data.reshape(-1, 1)
        elif data.ndim > 2: raise ValueError

        if data.shape[1] != channels: raise ValueError
        if data.dtype != dtype: raise TypeError
        if not data.flags.c_contiguous: raise TypeError

        return _OutputStreamBase._raw_write(self, data)

class Stream(InputStream, OutputStream):
    def __init__(self, samplerate=None, blocksize=None, device=None, channels=None, dtype=None, latency=None, extra_settings=None, callback=None, finished_callback=None, clip_off=None, dither_off=None, never_drop_input=None, prime_output_buffers_using_stream_callback=None):
        StreamBase.__init__(self, kind='duplex', wrap_callback='array', **remove_self(locals()))

class DeviceList(tuple):
    __slots__ = ()

    def __repr__(self):
        idev = get_device_id(default.device['input'], 'input')
        odev = get_device_id(default.device['output'], 'output')
        digits = len(str(portaudiolib.Pa_GetDeviceCount() - 1))
        hostapi_names = [hostapi['name'] for hostapi in query_hostapis()]

        def get_mark(idx):
            return (' ', '>', '<', '*')[(idx == idev) + 2 * (idx == odev)]

        text = '\n'.join('{mark} {idx:{dig}} {name}, {ha} ({ins} in, {outs} out)'.format(mark=get_mark(info['index']), idx=info['index'], dig=digits, name=info['name'], ha=hostapi_names[info['hostapi']], ins=info['max_input_channels'], outs=info['max_output_channels']) for info in self)
        return text

class CallbackFlags:
    __slots__ = '_flags'

    def __init__(self, flags=0x0):
        self._flags = flags

    def __repr__(self):
        flags = str(self)
        return f'<sounddevice.CallbackFlags: {flags}>'

    def __str__(self):
        return ', '.join(name.replace('_', ' ') for name in dir(self) if not name.startswith('_') and getattr(self, name))

    def __bool__(self):
        return bool(self._flags)

    def __ior__(self, other):
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
        return bool(self._flags & flag)

    def _updateflag(self, flag, value):
        if value: self._flags |= flag
        else: self._flags &= ~flag

class InputOutputPair:
    _indexmapping = {'input': 0, 'output': 1}

    def __init__(self, parent, default_attr):
        self._pair = [None, None]
        self._parent = parent
        self._default_attr = default_attr

    def __getitem__(self, index):
        index = self._indexmapping.get(index, index)
        value = self._pair[index]
        if value is None: value = getattr(self._parent, self._default_attr)[index]
        return value

    def __setitem__(self, index, value):
        index = self._indexmapping.get(index, index)
        self._pair[index] = value

    def __repr__(self):
        return '[{0[0]!r}, {0[1]!r}]'.format(self)

class default:
    _pairs = 'device', 'channels', 'dtype', 'latency', 'extra_settings'
    device = (None, None)
    _default_channels = None, None
    channels = _default_channels
    _default_dtype = 'float32', 'float32'
    dtype = _default_dtype
    _default_latency = 'high', 'high'
    latency = _default_latency
    _default_extra_settings = None, None
    extra_settings = _default_extra_settings
    samplerate = None
    blocksize = portaudiolib.paFramesPerBufferUnspecified
    clip_off = False
    dither_off = False
    never_drop_input = False
    prime_output_buffers_using_stream_callback = False

    def __init__(self):
        for attr in self._pairs:
            vars(self)[attr] = InputOutputPair(self, '_default_' + attr)

    def __setattr__(self, name, value):
        if name in self._pairs: getattr(self, name)._pair[:] = split(value)
        elif name in dir(self) and name != 'reset': object.__setattr__(self, name, value)
        else: raise AttributeError(repr(name))

    @property
    def _default_device(self):
        return (portaudiolib.Pa_GetDefaultInputDevice(), portaudiolib.Pa_GetDefaultOutputDevice())

    @property
    def hostapi(self):
        return check(portaudiolib.Pa_GetDefaultHostApi())

    def reset(self):
        vars(self).clear()
        self.__init__()

if not hasattr(ffi, 'I_AM_FAKE'):
    _default_instance = default()
    default = _default_instance

class PortAudioError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class CallbackStop(Exception):
    def __init__(self, *args):
        super().__init__(*args)


class CallbackAbort(Exception):
    def __init__(self, *args):
        super().__init__(*args)

class AsioSettings:
    def __init__(self, channel_selectors):
        if isinstance(channel_selectors, int): raise TypeError
        self._selectors = ffi.new('int[]', channel_selectors)
        self._streaminfo = ffi.new('PaAsioStreamInfo*', dict(size=ffi.sizeof('PaAsioStreamInfo'), hostApiType=portaudiolib.paASIO, version=1, flags=portaudiolib.paAsioUseChannelSelectors, channelSelectors=self._selectors))

class CoreAudioSettings:
    def __init__(self, channel_map=None, change_device_parameters=False, fail_if_conversion_required=False, conversion_quality='max'):
        conversion_dict = {
            'min':    portaudiolib.paMacCoreConversionQualityMin,
            'low':    portaudiolib.paMacCoreConversionQualityLow,
            'medium': portaudiolib.paMacCoreConversionQualityMedium,
            'high':   portaudiolib.paMacCoreConversionQualityHigh,
            'max':    portaudiolib.paMacCoreConversionQualityMax,
        }

        if isinstance(channel_map, int): raise TypeError

        try:
            self._flags = conversion_dict[conversion_quality.lower()]
        except (KeyError, AttributeError):
            raise ValueError(repr(list(conversion_dict)))

        if change_device_parameters: self._flags |= portaudiolib.paMacCoreChangeDeviceParameters
        if fail_if_conversion_required: self._flags |= portaudiolib.paMacCoreFailIfConversionRequired

        self._streaminfo = ffi.new('PaMacCoreStreamInfo*')
        portaudiolib.PaMacCore_SetupStreamInfo(self._streaminfo, self._flags)

        if channel_map is not None:
            self._channel_map = ffi.new('SInt32[]', channel_map)
            if len(self._channel_map) == 0: raise TypeError
            portaudiolib.PaMacCore_SetupChannelMap(self._streaminfo, self._channel_map, len(self._channel_map))

class WasapiSettings:
    def __init__(self, exclusive=False, auto_convert=False, explicit_sample_format=False):
        flags = 0x0

        if exclusive: flags |= portaudiolib.paWinWasapiExclusive
        if auto_convert: flags |= portaudiolib.paWinWasapiAutoConvert
        if explicit_sample_format: flags |= portaudiolib.paWinWasapiExplicitSampleFormat

        self._streaminfo = ffi.new('PaWasapiStreamInfo*', dict(size=ffi.sizeof('PaWasapiStreamInfo'), hostApiType=portaudiolib.paWASAPI, version=1, flags=flags,))

class CallbackContext:
    frame = 0
    frames = 0
    input_channels = output_channels = None
    input_dtype = output_dtype = None
    silent_channels = None

    def __init__(self, loop=False):
        self.loop = loop
        self.event = threading.Event()
        self.status = CallbackFlags()

    def check_data(self, data, mapping, device):
        data = np.asarray(data)

        if data.ndim < 2: data = data.reshape(-1, 1)
        elif data.ndim > 2: raise ValueError

        frames, channels = data.shape
        dtype = check_dtype(data.dtype)

        mapping_is_explicit = mapping is not None
        mapping, channels = check_mapping(mapping, channels)

        if data.shape[1] == 1: pass
        elif data.shape[1] != len(mapping): raise ValueError

        if (mapping_is_explicit and np.array_equal(mapping, [0]) and query_devices(device, 'output')['max_output_channels'] >= 2): channels = 2

        silent_channels = np.setdiff1d(np.arange(channels), mapping)
        if len(mapping) + len(silent_channels) != channels: raise ValueError

        self.data = data
        self.output_channels = channels
        self.output_dtype = dtype
        self.output_mapping = mapping
        self.silent_channels = silent_channels

        return frames

    def check_out(self, out, frames, channels, dtype, mapping):
        if out is None:
            if frames is None: raise TypeError
            if channels is None: channels = default.channels['input']

            if channels is None:
                if mapping is None: raise TypeError
                else: channels = len(np.atleast_1d(mapping))

            if dtype is None: dtype = default.dtype['input']

            try:
                out = np.empty((frames, channels), dtype, order='C')
            except TypeError as e:
                if not isinstance(frames, Integral): raise TypeError
                if not isinstance(channels, Integral): raise TypeError
                raise e
        else:
            frames, channels = out.shape
            dtype = out.dtype

        dtype = check_dtype(dtype)
        mapping, channels = check_mapping(mapping, channels)
        if out.shape[1] != len(mapping): raise ValueError

        self.out = out
        self.input_channels = channels
        self.input_dtype = dtype
        self.input_mapping = mapping

        return out, frames

    def callback_enter(self, status, data):
        self.status |= status
        self.blocksize = min(self.frames - self.frame, len(data))

    def read_indata(self, indata):
        for target, source in enumerate(self.input_mapping):
            self.out[self.frame:self.frame + self.blocksize, target] = indata[:self.blocksize, source]

    def write_outdata(self, outdata):
        outdata[:self.blocksize, self.output_mapping] = self.data[self.frame:self.frame + self.blocksize]
        outdata[:self.blocksize, self.silent_channels] = 0

        if self.loop and self.blocksize < len(outdata):
            self.frame = 0
            outdata = outdata[self.blocksize:]

            self.blocksize = min(self.frames, len(outdata))
            self.write_outdata(outdata)
        else:
            outdata[self.blocksize:] = 0

    def callback_exit(self):
        if not self.blocksize: raise CallbackAbort
        self.frame += self.blocksize

    def finished_callback(self):
        self.event.set()

        with contextlib.suppress(AttributeError):
            del self.data

        with contextlib.suppress(AttributeError):
            del self.out

        self.stream._callback = None
        self.stream._finished_callback = None

    def start_stream(self, StreamClass, samplerate, channels, dtype, callback, blocking, **kwargs):
        stop()

        self.stream = StreamClass(samplerate=samplerate, channels=channels, dtype=dtype, callback=callback, finished_callback=self.finished_callback, **kwargs)
        self.stream.start()

        global last_callback
        last_callback = self
        if blocking: self.wait()

    def wait(self, ignore_errors=True):
        try:
            self.event.wait()
        finally:
            self.stream.close(ignore_errors)

        return self.status if self.status else None

def ffi_string(cdata):
    return ffi.string(cdata)

def remove_self(d):
    d = d.copy()
    del d['self']

    return d

def check_mapping(mapping, channels):
    if mapping is None: mapping = np.arange(channels)
    else:
        mapping = np.array(mapping, copy=True)
        mapping = np.atleast_1d(mapping)

        if mapping.min() < 1: raise ValueError

        channels = mapping.max()
        mapping -= 1

    return mapping, channels

def check_dtype(dtype):
    dtype = np.dtype(dtype).name

    if dtype in sampleformats: pass
    elif dtype == 'float64': dtype = np.dtype(np.float32).name
    else: raise TypeError

    return dtype

def get_stream_parameters(kind, device, channels, dtype, latency, extra_settings, samplerate):
    assert kind in ('input', 'output')

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
        raise ValueError

    samplesize = check(portaudiolib.Pa_GetSampleSize(sampleformat))

    if latency in ('low', 'high'): latency = info['default_' + latency + '_' + kind + '_latency']
    if samplerate is None: samplerate = info['default_samplerate']

    parameters = ffi.new('PaStreamParameters*', (device, channels, sampleformat, latency, extra_settings._streaminfo if extra_settings else ffi.NULL))
    return parameters, dtype, samplesize, samplerate

def _wrap_callback(callback, *args):
    args = args[:-1] + (CallbackFlags(args[-1]),)

    try:
        callback(*args)
    except CallbackStop:
        return portaudiolib.paComplete
    except CallbackAbort:
        return portaudiolib.paAbort

    return portaudiolib.paContinue

def mbuffer(ptr, frames, channels, samplesize):
    return ffi.buffer(ptr, frames * channels * samplesize)

def array(buffer, channels, dtype):
    data = np.frombuffer(buffer, dtype=dtype)
    data.shape = -1, channels
    return data

def split(value):
    if isinstance(value, (str, bytes)): return value, value

    try:
        invalue, outvalue = value
    except TypeError:
        invalue = outvalue = value
    except ValueError as e:
        raise ValueError

    return invalue, outvalue

def check(err, msg=''):
    if err >= 0: return err

    errormsg = ffi_string(portaudiolib.Pa_GetErrorText(err)).decode()
    if msg: errormsg = f'{msg}: {errormsg}'

    if err == portaudiolib.paUnanticipatedHostError:
        info = portaudiolib.Pa_GetLastHostErrorInfo()
        host_api = portaudiolib.Pa_HostApiTypeIdToHostApiIndex(info.hostApiType)

        hosterror_text = ffi_string(info.errorText).decode()
        hosterror_info = host_api, info.errorCode, hosterror_text

        raise PortAudioError(errormsg, err, hosterror_info)
    raise PortAudioError(errormsg, err)

def select_input_or_output(value_or_pair, kind):
    ivalue, ovalue = split(value_or_pair)

    if kind == 'input': return ivalue
    elif kind == 'output': return ovalue

    assert False

def get_device_id(id_or_query_string, kind, raise_on_error=False):
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

    for id, info in enumerate(query_devices()):
        if not kind or info['max_' + kind + '_channels'] > 0:
            hostapi_info = query_hostapis(info['hostapi'])
            device_list.append((id, info['name'], hostapi_info['name']))

    query_string = id_or_query_string.lower()
    substrings = query_string.split()

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
        if raise_on_error: raise ValueError(repr(id_or_query_string))
        else: return -1

    if len(matches) > 1:
        if len(exact_device_matches) == 1: return exact_device_matches[0]

        if raise_on_error: raise ValueError(kind + " " + repr(id_or_query_string) + ':\n' + '\n'.join(f'[{id}] {name}' for id, name in matches))
        else: return -1

    return matches[0][0]

def initialize():
    old_stderr = None

    try:
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
    global initialized

    check(portaudiolib.Pa_Terminate())
    initialized -= 1


def exit_handler():
    assert initialized >= 0

    if last_callback:
        last_callback.stream.stop()
        last_callback.stream.close()

    while initialized:
        terminate()

atexit.register(exit_handler)
initialize()