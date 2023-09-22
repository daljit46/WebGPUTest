#include "utils.h"
#include <iostream>
#include <cassert>
#include <fstream>

namespace Utils {
WGPUAdapter requestAdapter(WGPUInstance instance,
                           const WGPURequestAdapterOptions *options) {
  struct UserData {
    WGPUAdapter adapter = nullptr;
    bool requestEnded = false;
  };

  UserData userData;

  auto onAdapterRequestEnded = [](WGPURequestAdapterStatus status,
                                  WGPUAdapter adapter, const char *message,
                                  void *pUserData) {
    UserData &userData = *reinterpret_cast<UserData *>(pUserData);
    if (status == WGPURequestAdapterStatus_Success) {
      userData.adapter = adapter;
    } else {
      std::cout << "Could not get WebGPU adapter: " << message << std::endl;
    }
    userData.requestEnded = true;
  };
  wgpuInstanceRequestAdapter(instance, options, onAdapterRequestEnded,
                             (void *)&userData);
  assert(userData.requestEnded);

  return userData.adapter;
}

WGPUDevice requestDevice(WGPUAdapter adapter, const WGPUDeviceDescriptor *descriptor)
{
    struct UserData {
        WGPUDevice device = nullptr;
        bool requestEnded = false;
    };

    UserData userData;

    auto onDeviceRequestEnded = [](WGPURequestDeviceStatus status,
                                   WGPUDevice device,
                                   const char *message,
                                   void *pUserData){
        UserData& userData = *reinterpret_cast<UserData*>(pUserData);
        if (status == WGPURequestDeviceStatus_Success) {
            userData.device = device;
        } else {
            std::cout << "Could not get WebGPU device: " << message << std::endl;
        }
        userData.requestEnded = true;
    };

    wgpuAdapterRequestDevice(adapter, descriptor, onDeviceRequestEnded, (void*)&userData);
    assert(userData.requestEnded);

    return userData.device;
}

WGPUShaderModule loadShaderModule(const std::filesystem::path& filePath, WGPUDevice device)
{
    std::ifstream file(filePath);
    if(!file.is_open()){
        std::cerr << "Could not open file: " << filePath << std::endl;
        return nullptr;
    }
    file.seekg(0, std::ios::end);
    std::vector<char> buffer(file.tellg());
    file.seekg(0);
    file.read(buffer.data(), buffer.size());

    WGPUShaderModuleWGSLDescriptor shaderCodeDesc = {};
    shaderCodeDesc.chain.next = nullptr;
    shaderCodeDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    shaderCodeDesc.code = buffer.data();
    WGPUShaderModuleDescriptor shaderDesc{};
    shaderDesc.nextInChain = &shaderCodeDesc.chain;
    return wgpuDeviceCreateShaderModule(device, &shaderDesc);
}

WGPUBindGroupLayoutEntry createDefaultBindingLayout ()
{
    WGPUBindGroupLayoutEntry bindingLayout;
    bindingLayout.buffer.nextInChain = nullptr;
    bindingLayout.buffer.type = WGPUBufferBindingType_Undefined;
    bindingLayout.buffer.hasDynamicOffset = false;

    bindingLayout.sampler.nextInChain = nullptr;
    bindingLayout.sampler.type = WGPUSamplerBindingType_Undefined;

    bindingLayout.storageTexture.nextInChain = nullptr;
    bindingLayout.storageTexture.access = WGPUStorageTextureAccess_Undefined;
    bindingLayout.storageTexture.format = WGPUTextureFormat_Undefined;
    bindingLayout.storageTexture.viewDimension = WGPUTextureViewDimension_Undefined;

    bindingLayout.texture.nextInChain = nullptr;
    bindingLayout.texture.multisampled = false;
    bindingLayout.texture.sampleType = WGPUTextureSampleType_Undefined;
    bindingLayout.texture.viewDimension = WGPUTextureViewDimension_Undefined;

    return bindingLayout;
}
} // namespace Utils
