
#pragma once
#include <webgpu/webgpu.h>

namespace Utils {
WGPUAdapter requestAdapter(WGPUInstance instance,
                           const WGPURequestAdapterOptions *options);

WGPUDevice requestDevice(WGPUAdapter adapter,
                         const WGPUDeviceDescriptor *descriptor);
} // namespace Utils