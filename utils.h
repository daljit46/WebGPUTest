#pragma once

#include <webgpu/webgpu.h>
#include <filesystem>

namespace Utils {
WGPUAdapter requestAdapter(WGPUInstance instance,
                           const WGPURequestAdapterOptions *options);

WGPUDevice requestDevice(WGPUAdapter adapter,
                         const WGPUDeviceDescriptor *descriptor);

WGPUShaderModule loadShaderModule(const std::filesystem::path& filePath,
                                  WGPUDevice device);

WGPUBindGroupLayoutEntry createDefaultBindingLayout();
} // namespace Utils
