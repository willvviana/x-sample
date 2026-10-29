"""Windows GPU information from the registry, DXGI and PDH performance counters.

WMI's Win32_VideoController.AdapterRAM is a 32-bit field capped at 4 GiB, so VRAM is
read from the display driver registry key and DXGI instead. PDH "GPU Process Memory"
counters identify which adapter (by LUID) a process actually allocated memory on.
"""

from __future__ import annotations

import ctypes
import re
import winreg
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


@dataclass(frozen=True)
class RegistryGpu:
    name: str
    vram_bytes: int | None
    driver_version: str | None


def _reg_value(key: winreg.HKEYType, name: str) -> Any:
    try:
        return winreg.QueryValueEx(key, name)[0]
    except OSError:
        return None


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, bytes) and value:
        return int.from_bytes(value, "little")
    return None


def registry_gpus() -> list[RegistryGpu]:
    gpus: list[RegistryGpu] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return gpus
    with root:
        index = 0
        while True:
            try:
                sub = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = _reg_value(key, "DriverDesc")
                    if not isinstance(name, str) or not name:
                        continue
                    vram = _as_int(_reg_value(key, "HardwareInformation.qwMemorySize"))
                    if vram is None:
                        vram = _as_int(_reg_value(key, "HardwareInformation.MemorySize"))
                    driver = _reg_value(key, "DriverVersion")
                    gpus.append(
                        RegistryGpu(name, vram, driver if isinstance(driver, str) else None)
                    )
            except OSError:
                continue
    return gpus


def cpu_name() -> str | None:
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        ) as key:
            value = _reg_value(key, "ProcessorNameString")
            return value.strip() if isinstance(value, str) else None
    except OSError:
        return None


# --- DXGI -------------------------------------------------------------------------


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
        ("Flags", wintypes.UINT),
    ]


_IID_IDXGIFactory1 = _GUID(
    0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87)
)
_DXGI_ERROR_NOT_FOUND = 0x887A0002
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2
# vtable slots: IUnknown(0-2), IDXGIObject(3-6), IDXGIFactory(7-11), IDXGIFactory1(12-13)
_SLOT_RELEASE = 2
_SLOT_ENUM_ADAPTERS1 = 12
# IDXGIAdapter(7-9), IDXGIAdapter1.GetDesc1 = 10
_SLOT_GET_DESC1 = 10


@dataclass(frozen=True)
class DxgiAdapter:
    index: int
    name: str
    vendor_id: int
    device_id: int
    dedicated_video_memory: int
    shared_system_memory: int
    luid: tuple[int, int]  # (high, low)
    software: bool


def _com_method(obj: ctypes.c_void_p, slot: int, *argtypes: Any) -> Any:
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    return prototype(vtable[slot])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _com_method(obj, _SLOT_RELEASE)(obj)


def dxgi_adapters() -> list[DxgiAdapter]:
    dxgi = ctypes.WinDLL("dxgi")
    factory = ctypes.c_void_p()
    hr = dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
    if hr != 0:
        raise OSError(f"CreateDXGIFactory1 failed: 0x{hr & 0xFFFFFFFF:08X}")
    adapters: list[DxgiAdapter] = []
    try:
        enum_adapters = _com_method(
            factory, _SLOT_ENUM_ADAPTERS1, ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)
        )
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            hr = enum_adapters(factory, index, ctypes.byref(adapter))
            if hr & 0xFFFFFFFF == _DXGI_ERROR_NOT_FOUND:
                break
            if hr != 0:
                raise OSError(f"EnumAdapters1({index}) failed: 0x{hr & 0xFFFFFFFF:08X}")
            try:
                desc = _DXGI_ADAPTER_DESC1()
                get_desc = _com_method(
                    adapter, _SLOT_GET_DESC1, ctypes.POINTER(_DXGI_ADAPTER_DESC1)
                )
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    adapters.append(
                        DxgiAdapter(
                            index=index,
                            name=desc.Description,
                            vendor_id=desc.VendorId,
                            device_id=desc.DeviceId,
                            dedicated_video_memory=desc.DedicatedVideoMemory,
                            shared_system_memory=desc.SharedSystemMemory,
                            luid=(desc.AdapterLuid.HighPart & 0xFFFFFFFF, desc.AdapterLuid.LowPart),
                            software=bool(desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE),
                        )
                    )
            finally:
                _release(adapter)
            index += 1
    finally:
        _release(factory)
    return adapters


# --- PDH GPU memory counters ------------------------------------------------------

_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wintypes.DWORD), ("largeValue", ctypes.c_longlong)]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PDH_FMT_COUNTERVALUE)]


def _pdh_read_array(counter_path: str) -> list[tuple[str, int]]:
    pdh = ctypes.WinDLL("pdh")
    query = ctypes.c_void_p()
    counter = ctypes.c_void_p()
    status = pdh.PdhOpenQueryW(None, ctypes.c_void_p(0), ctypes.byref(query))
    if status != 0:
        raise OSError(f"PdhOpenQueryW failed: 0x{status & 0xFFFFFFFF:08X}")
    try:
        status = pdh.PdhAddEnglishCounterW(
            query, ctypes.c_wchar_p(counter_path), ctypes.c_void_p(0), ctypes.byref(counter)
        )
        if status != 0:
            raise OSError(f"PdhAddEnglishCounterW failed: 0x{status & 0xFFFFFFFF:08X}")
        status = pdh.PdhCollectQueryData(query)
        if status != 0:
            raise OSError(f"PdhCollectQueryData failed: 0x{status & 0xFFFFFFFF:08X}")
        size = wintypes.DWORD(0)
        count = wintypes.DWORD(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None
        )
        if status & 0xFFFFFFFF != _PDH_MORE_DATA:
            return []
        buffer = ctypes.create_string_buffer(size.value)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer
        )
        if status != 0:
            raise OSError(f"PdhGetFormattedCounterArrayW failed: 0x{status & 0xFFFFFFFF:08X}")
        items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        return [(items[i].szName, int(items[i].FmtValue.largeValue)) for i in range(count.value)]
    finally:
        pdh.PdhCloseQuery(query)


_PROCESS_INSTANCE = re.compile(r"pid_(\d+)_luid_0x([0-9A-Fa-f]+)_0x([0-9A-Fa-f]+)_phys_\d+")


@dataclass(frozen=True)
class ProcessGpuMemory:
    luid: tuple[int, int]
    dedicated_bytes: int


def process_gpu_memory(pid: int) -> list[ProcessGpuMemory]:
    """Dedicated GPU memory the given process holds, per adapter LUID."""
    usage: dict[tuple[int, int], int] = {}
    for name, value in _pdh_read_array(r"\GPU Process Memory(*)\Dedicated Usage"):
        match = _PROCESS_INSTANCE.fullmatch(name)
        if not match or int(match.group(1)) != pid:
            continue
        luid = (int(match.group(2), 16), int(match.group(3), 16))
        usage[luid] = usage.get(luid, 0) + value
    return [ProcessGpuMemory(luid, value) for luid, value in usage.items()]
