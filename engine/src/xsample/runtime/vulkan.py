"""Vulkan availability detection through the system loader (vulkan-1.dll).

This reports driver and device presence only. It does not mean any X-Sample model
can run on Vulkan; that requires a registered runner and a passing probe (ADR-0003).
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field

_VK_SUCCESS = 0
_VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
_VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
_DEVICE_TYPES = {0: "other", 1: "integrated", 2: "discrete", 3: "virtual", 4: "cpu"}


class _VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(_VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


def _version(value: int) -> str:
    return f"{(value >> 22) & 0x7F}.{(value >> 12) & 0x3FF}.{value & 0xFFF}"


@dataclass(frozen=True)
class VulkanDevice:
    name: str
    device_type: str
    api_version: str
    driver_version_raw: int
    vendor_id: int
    device_id: int


@dataclass
class VulkanStatus:
    loader_present: bool
    instance_version: str | None = None
    devices: list[VulkanDevice] = field(default_factory=list)
    error: str | None = None


def detect() -> VulkanStatus:
    try:
        vk = ctypes.WinDLL("vulkan-1")
    except OSError as exc:
        return VulkanStatus(loader_present=False, error=str(exc))

    status = VulkanStatus(loader_present=True)
    try:
        version = ctypes.c_uint32(0)
        if vk.vkEnumerateInstanceVersion(ctypes.byref(version)) == _VK_SUCCESS:
            status.instance_version = _version(version.value)

        app = _VkApplicationInfo(
            _VK_STRUCTURE_TYPE_APPLICATION_INFO, None, b"X-Sample", 1, b"xsample", 1, (1 << 22)
        )
        create = _VkInstanceCreateInfo(
            _VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None
        )
        instance = ctypes.c_void_p()
        vk.vkCreateInstance.restype = ctypes.c_int32
        result = vk.vkCreateInstance(ctypes.byref(create), None, ctypes.byref(instance))
        if result != _VK_SUCCESS:
            status.error = f"vkCreateInstance returned {result}"
            return status
        try:
            count = ctypes.c_uint32(0)
            vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
            handles = (ctypes.c_void_p * count.value)()
            vk.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), handles)
            for handle in handles:
                # VkPhysicalDeviceProperties is ~800 bytes; only the leading fields are read.
                props = ctypes.create_string_buffer(4096)
                vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(handle), props)
                header = (ctypes.c_uint32 * 5).from_buffer(props)
                name = props.raw[20 : 20 + 256].split(b"\0", 1)[0].decode("utf-8", "replace")
                status.devices.append(
                    VulkanDevice(
                        name=name,
                        device_type=_DEVICE_TYPES.get(header[4], "unknown"),
                        api_version=_version(header[0]),
                        driver_version_raw=header[1],
                        vendor_id=header[2],
                        device_id=header[3],
                    )
                )
        finally:
            vk.vkDestroyInstance(instance, None)
    except (OSError, AttributeError) as exc:
        status.error = str(exc)
    return status
