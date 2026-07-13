import os
import gc
import sys
import torch
import pickle
import ctypes

from ctypes import wintypes

sys.path.append(os.getcwd())

from main.library.embedders import fairseq
from main.library.backends.utils import GRU, DeviceProperties

# Use these backends only if the library is installed.
try:
    import torch_directml
except:
    torch_directml = None

# Global hardware configurations and hooks
devices = []
torch_available = torch_directml != None
setupapi, advapi32, empty_cache_ctypes = None, None, None
orig_conv_transpose1d = torch.nn.functional.conv_transpose1d
empty_cache_dll = os.path.join("assets", "binary", "empty_cache.dll")

class GUID_BYTE(ctypes.Structure):
    """
    Represents a Windows Globally Unique Identifier (GUID) structure.
    Uses standard byte array format for compatibility with SetupAPI calls.
    """

    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_byte * 8)]
    
class GUID_UBYTPE(ctypes.Structure):
    """
    Represents a Windows GUID structure using unsigned byte values.
    Typically required for COM and DXGI component registration interfaces.
    """

    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]

class SP_DEVINFO_DATA(ctypes.Structure):
    """
    Defines a Windows Device Information structure (`SP_DEVINFO_DATA`).
    Contains metadata referencing a specific device instance within a device information set.
    """

    _fields_ = [("cbSize", wintypes.DWORD), ("ClassGuid", GUID_BYTE), ("DevInst", wintypes.DWORD), ("Reserved", ctypes.c_void_p)]

class DXGI_ADAPTER_DESC1(ctypes.Structure):
    """
    Describes a Direct3D 11 / DXGI graphics adapter interface descriptor block.
    Extracts explicit VRAM capacities and low-level device descriptors.
    """

    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT), ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", ctypes.c_uint64), ("Flags", wintypes.UINT)]

def unpack_empty_cache():
    """
    Unpacks compressed binary dependency payloads (`.dll`, `.exp`, `.lib`, `.obj`) 
    from a pickled backup file (`empty_cache.bin`) and binds the target WinDLL interface.

    Raises:
        OSError: If binary assets cannot be created or loaded via ctypes.
    """

    global empty_cache_ctypes
    # Extract binaries if missing
    if not os.path.exists(empty_cache_dll):
        with open(os.path.join("assets", "binary", "empty_cache.bin"), "rb") as f:
            data = pickle.load(f)
        
        for i in ["dll", "exp", "lib", "obj"]:
            with open(empty_cache_dll.replace(".dll", "." + i), "wb") as f:
                f.write(data[i])
    
    # Load DLL and map parameter signatures
    try:
        empty_cache_ctypes = ctypes.WinDLL(empty_cache_dll)
        empty_cache_ctypes.empty_cache.argtypes = [ctypes.c_uint]
        empty_cache_ctypes.empty_cache.restype = ctypes.c_bool
    except Exception as e:
        raise OSError(f"Failed to load dynamic library interface from '{empty_cache_dll}': {e}")

def setup_api():
    """
    Initializes system-level ctypes mappings for Windows SetupAPI and Advapi32 DLLs.
    Configures strict static type declarations for external C functions.
    """

    global setupapi, advapi32

    try:
        setupapi = ctypes.windll.setupapi
        advapi32 = ctypes.windll.advapi32
    except Exception as e:
        raise OSError(f"Failed to initialize target Windows environment subsystem libraries: {e}")

    # SetupDiGetClassDevsW: Query device information sets based on GUID class
    setupapi.SetupDiGetClassDevsW.argtypes = [ctypes.POINTER(GUID_BYTE), ctypes.c_wchar_p, wintypes.HWND, wintypes.DWORD]
    setupapi.SetupDiGetClassDevsW.restype = ctypes.c_void_p

    # SetupDiEnumDeviceInfo: Enumerate elements within a device information set
    setupapi.SetupDiEnumDeviceInfo.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(SP_DEVINFO_DATA)]
    setupapi.SetupDiEnumDeviceInfo.restype = wintypes.BOOL

    # SetupDiGetDeviceRegistryPropertyW: Extract explicit registry keys representing hardware traits
    setupapi.SetupDiGetDeviceRegistryPropertyW.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA), wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    setupapi.SetupDiGetDeviceRegistryPropertyW.restype = wintypes.BOOL

    # SetupDiOpenDevRegKey: Open configuration registry scopes belonging to device drivers
    setupapi.SetupDiOpenDevRegKey.argtypes = [ctypes.c_void_p, ctypes.POINTER(SP_DEVINFO_DATA), wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
    setupapi.SetupDiOpenDevRegKey.restype = wintypes.HKEY

    # Bind Advapi32 functions to evaluate properties directly out of registry pipelines
    advapi32.RegQueryValueExW.argtypes = [wintypes.HKEY, ctypes.c_wchar_p, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    advapi32.RegQueryValueExW.restype = wintypes.LONG
    advapi32.RegCloseKey.argtypes = [wintypes.HKEY]
    
    # Dynamically extract and assign internal system clearing binaries
    unpack_empty_cache()

def get_directml_device():
    """
    Scans the local operating system configuration space to discover hardware units matching 
    DirectML structural categories, parsing absolute VRAM values from physical system nodes.
    """

    global devices

    if len(devices) >= 1: return
    if not torch_available: return
    if setupapi is None or advapi32 is None: setup_api()

    # GUID matching system Display Adapter profiles
    display_guid = GUID_BYTE(0x4D36E968, 0xE325, 0x11CE, (ctypes.c_byte * 8)(0xBF, 0xC1, 0x08, 0x00, 0x2B, 0xE1, 0x03, 0x18))
    h_dev_info = ctypes.c_void_p(setupapi.SetupDiGetClassDevsW(ctypes.byref(display_guid), None, None, 0x00000002))
    if h_dev_info.value is None or h_dev_info.value == int(0xFFFFFFFFFFFFFFFF): return
    
    dev_info_data = SP_DEVINFO_DATA()
    dev_info_data.cbSize = ctypes.sizeof(SP_DEVINFO_DATA)
    idx, gpu_idx = 0, 0

    # Iterative discovery loops across system hardware trees
    while setupapi.SetupDiEnumDeviceInfo(h_dev_info, idx, ctypes.byref(dev_info_data)):
        buf = ctypes.create_unicode_buffer(256)
        # Retrieve structural name descriptions
        if setupapi.SetupDiGetDeviceRegistryPropertyW(h_dev_info, ctypes.byref(dev_info_data), 0x00000000, None, ctypes.byref(buf), ctypes.sizeof(buf), ctypes.byref(wintypes.DWORD())):
            gpu_name = buf.value.replace("\x00", "").strip()
            name_lower = gpu_name.lower()

            # Exclude virtual visualization interfaces and mirror rendering targets
            if gpu_name and "virtual" not in name_lower and "mirror" not in name_lower:
                # Omit underpowered integrated legacy Intel lines unless they belong to modern high-performance series
                if "intel" in name_lower and "graphics" in name_lower:
                    if not ("arc" in name_lower or "iris" in name_lower or "ultra" in name_lower):
                        idx += 1
                        continue     
                
                vram_bytes = 0.0
                h_key = setupapi.SetupDiOpenDevRegKey(h_dev_info, ctypes.byref(dev_info_data), 0x00000001, 0, 0x00000002, 0x20019)
                # Extract allocation sizes directly out of active device hardware registry entries
                if h_key and h_key != int(0xFFFFFFFFFFFFFFFF):
                    size_buf_64 = ctypes.c_ulonglong()
                    size_len = wintypes.DWORD(ctypes.sizeof(size_buf_64))
                    type_buf = wintypes.DWORD()
                    # Attempt reading 64-bit size values
                    ret = advapi32.RegQueryValueExW(h_key, "HardwareInformation.qwMemorySize", None, ctypes.byref(type_buf), ctypes.byref(size_buf_64), ctypes.byref(size_len))
                    if ret == 0 and size_buf_64.value > 0: vram_bytes = size_buf_64.value
                    else:
                        # Fallback to standard 32-bit values if 64-bit property doesn't exist
                        size_buf_32 = wintypes.DWORD()
                        size_len = wintypes.DWORD(ctypes.sizeof(size_buf_32))

                        ret = advapi32.RegQueryValueExW(h_key, "HardwareInformation.MemorySize", None, ctypes.byref(type_buf), ctypes.byref(size_buf_32), ctypes.byref(size_len))
                        if ret == 0 and size_buf_32.value > 0: vram_bytes = size_buf_32.value
                    
                    advapi32.RegCloseKey(h_key)
                
                # Add the GPU descriptor to the device list.
                devices.append(DeviceProperties(gpu_idx, gpu_name, float(vram_bytes)))
                gpu_idx += 1
        idx += 1

def mapping_directml(device_id = 0):
    """
    Basically, torch_directml prioritizes device 0 as the most powerful GPU available on the system, whereas ONNX Runtime uses the system's actual device indices. 
    This function remaps the ONNX Runtime device indices to match the device ordering used by torch_directml.

    Args:
        device_id (int): Logical tracking identifier for execution. Defaults to 0.

    Returns:
        int: Real hardware hardware registry execution target context runtime ID offset index.
    
    Raises:
        IndexError: If device_id is completely out of bounds of scanned targets.
        OSError: If DXGI component initialization fails or factory instantiation falls through.
    """

    if device_id < 0 or device_id >= len(devices):
        raise IndexError(f"Provided device index '{device_id}' maps outside bounds of discovered system components.")

    IID_IDXGIFactory1 = GUID_UBYTPE(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))

    try:
        CreateDXGIFactory1 = ctypes.WinDLL("dxgi.dll").CreateDXGIFactory1
    except Exception as e:
        raise OSError(f"Failed to access backend dxgi.dll drivers: {e}")

    CreateDXGIFactory1.argtypes = [ctypes.POINTER(GUID_UBYTPE), ctypes.POINTER(ctypes.c_void_p)]
    CreateDXGIFactory1.restype = ctypes.c_long

    factory = ctypes.c_void_p()
    hr = CreateDXGIFactory1(ctypes.byref(IID_IDXGIFactory1), ctypes.byref(factory))
    if hr != 0: raise OSError(f"Failed to create DXGI factory instance block. ComHRESULT error token: {hex(hr & 0xffffffff)}")

    # Navigate virtual tables manually to call standard DXGI interfaces
    factory_vtbl = ctypes.cast(factory, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    EnumAdapters1 = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(factory_vtbl[12])
    idx = 0

    while 1:
        adapter = ctypes.c_void_p()
        hr = EnumAdapters1(factory, idx, ctypes.byref(adapter))
        if (hr & 0xffffffff) == 0x887A0002: break # DXGI_ERROR_NOT_FOUND signals completion

        desc = DXGI_ADAPTER_DESC1()
        ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(DXGI_ADAPTER_DESC1))(ctypes.cast(adapter, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents[10])(adapter, ctypes.byref(desc))
        # Confirm exact text matching to bind accurate identity references
        if desc.Description.lower() == devices[device_id].name.lower(): return idx
        idx += 1
    
    return device_id

def device_count():
    """
    Obtains the count of valid acceleration targets registered inside active execution environments.

    Returns:
        int: Number of hardware units discovered.
    """

    if (setupapi is None or advapi32 is None) and len(devices) == 0: get_directml_device()

    gpus = len(devices)
    return (torch_directml.device_count() if gpus == 0 else gpus) if torch_available else 0

def get_device_name(device_id = 0):
    """
    Queries descriptive naming configurations representing a given system index target.

    Args:
        device_id (int): Hardware reference ID. Defaults to 0.

    Returns:
        str: Discovered string representation of the targeted GPU device model name.
        
    Raises:
        RuntimeError: If execution context is triggered but no active targets were detected.
        ValueError: If requested device offset outranges total hardware inventory.
    """

    if (setupapi is None or advapi32 is None) and len(devices) == 0: get_directml_device()

    if len(devices) == 0:
        raise RuntimeError("No hardware acceleration devices available in the current execution environment.")

    if device_id >= 0 and device_id < device_count():
        try:
            return devices[device_id].name
        except:
            return torch_directml.device_name(device_id)
    else:
        raise ValueError(f"Requested hardware reference target index ({device_id}) out of bounds. Found total devices: {device_count()}")

def get_device_properties(device_id = 0):
    """
    Extracts the custom property profiles detailing capacities belonging to standard acceleration IDs.

    Args:
        device_id (int): Hardware target identifier tracking context references. Defaults to 0.

    Returns:
        DeviceProperties: Configuration object detailing tracking specifications.
        
    Raises:
        RuntimeError: If properties are requested but no hardware layer is tracked.
        ValueError: If requested index falls out of registered hardware profile ranges.
    """

    if (setupapi is None or advapi32 is None) and len(devices) == 0: get_directml_device()

    if len(devices) == 0:
        raise RuntimeError("No hardware acceleration devices available in the current execution environment.")

    if device_id >= 0 and device_id < device_count():
        try:
            return devices[device_id]
        except:
            return DeviceProperties(device_id, torch_directml.device_name(device_id), 0.0)
    else:
        raise ValueError(f"Requested hardware properties target index ({device_id}) out of bounds. Found total profiles: {device_count()}")

def is_available():
    """
    Checks if DirectML runtime dependencies are valid and accessible.

    Returns:
        bool: True if acceleration structures pass setup conditions successfully.
    """

    return torch_directml.is_available() if torch_available else False

def empty_cache():
    """
    Invokes external unmanaged native code logic to release unused VRAM/cache buffers, 
    then triggers Python's built-in garbage collector to reclaim system resources.

    C++ source code: ```
        #include <windows.h>

        #include <dxgi1_6.h>
        #include <d3d12.h>
        #include <wrl/client.h>

        #include <iostream>

        using Microsoft::WRL::ComPtr;

        extern "C" __declspec(dllexport)
        bool empty_cache(UINT adapterIndex)
        {
            HRESULT hr;

            ComPtr<IDXGIFactory6> factory;
            hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
            if (FAILED(hr)) return false;

            // std::wcout << L"========== GPU List ==========\n";

            // UINT i = 0;
            // ComPtr<IDXGIAdapter1> adapter;

            // while (factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND)
            // {
            //     DXGI_ADAPTER_DESC1 desc;
            //     adapter->GetDesc1(&desc);

            //     std::wcout
            //         << L"[" << i << L"] "
            //         << desc.Description
            //         << L" | Dedicated VRAM: "
            //         << desc.DedicatedVideoMemory / 1024 / 1024
            //         << L" MB"
            //         << std::endl;

            //     adapter.Reset();
            //     ++i;
            // }

            // std::wcout << L"==============================\n";

            ComPtr<IDXGIAdapter1> adapter1;

            hr = factory->EnumAdapters1(adapterIndex, &adapter1);
            if (FAILED(hr)) return false;

            DXGI_ADAPTER_DESC1 desc;
            adapter1->GetDesc1(&desc);

            // std::wcout << L"Using GPU: " << desc.Description << std::endl;

            ComPtr<IDXGIAdapter3> adapter3;
            adapter1.As(&adapter3);

            DXGI_QUERY_VIDEO_MEMORY_INFO before = {};
            DXGI_QUERY_VIDEO_MEMORY_INFO after = {};

            adapter3->QueryVideoMemoryInfo(
                0,
                DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                &before
            );

            ComPtr<ID3D12Device> device;

            hr = D3D12CreateDevice(
                adapter1.Get(),
                D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)
            );

            if (FAILED(hr)) return false;
            D3D12_COMMAND_QUEUE_DESC queueDesc = {};
            queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;

            ComPtr<ID3D12CommandQueue> queue;

            hr = device->CreateCommandQueue(
                &queueDesc,
                IID_PPV_ARGS(&queue)
            );

            if (FAILED(hr)) return false;
            ComPtr<ID3D12Fence> fence;

            hr = device->CreateFence(
                0,
                D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&fence));

            if (FAILED(hr))
                return false;

            HANDLE eventHandle = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            UINT64 value = 1;

            queue->Signal(fence.Get(), value);

            if (fence->GetCompletedValue() < value)
            {
                fence->SetEventOnCompletion(value, eventHandle);
                WaitForSingleObject(eventHandle, INFINITE);
            }

            CloseHandle(eventHandle);
            ComPtr<IDXGIDevice3> dxgiDevice;

            if (SUCCEEDED(device.As(&dxgiDevice))) dxgiDevice->Trim();

            queue.Reset();
            fence.Reset();
            device.Reset();

            adapter3->QueryVideoMemoryInfo(
                0,
                DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
                &after
            );

            double beforeMB = before.CurrentUsage / 1024.0 / 1024.0;
            double afterMB  = after.CurrentUsage  / 1024.0 / 1024.0;

            // std::cout << "VRAM Before : " << beforeMB << " MB\n";
            // std::cout << "VRAM After  : " << afterMB  << " MB\n";
            // std::cout << "Delta       : " << (afterMB - beforeMB) << " MB\n";

            return true;
        }
    ```
    """

    idx = 0

    if torch_available and empty_cache_ctypes is not None and os.path.exists(empty_cache_dll):
        while 1:
            results = empty_cache_ctypes.empty_cache(idx)
            idx += 1

            if not bool(results): break

        gc.collect()

def forward_dml(ctx, x, scale):
    """Newer versions of DirectML can function well without this component; it was developed based on the original RVC."""

    ctx.scale = scale
    res = x.clone().detach()
    return res

def cpu_conv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    """
    Wrapper function routing specific structural 1D convolution calculations through the CPU.
    Fixes compatibility errors since certain DirectML frameworks.
    """

    return orig_conv_transpose1d(
        input.cpu(), 
        weight.cpu(), 
        bias.cpu() if bias is not None else bias, 
        stride, 
        padding, 
        output_padding, 
        groups, 
        dilation
    ).to(input.device)

# Patch classes and functions to ensure everything operates correctly
if torch_available: 
    torch.nn.GRU = GRU # The GRU layer does not work on this backend, so I switched to a GRU Wrapper layer
    torch.inference_mode = torch.no_grad # Using `inference_mode` causes errors in some cases; therefore, I switched to using `no_grad`
    fairseq.GradMultiply.forward = forward_dml # Patch specialized forward properties 
    torch.nn.functional.conv_transpose1d = cpu_conv_transpose1d # This function might not work or may not function correctly, so it should be moved to the CPU