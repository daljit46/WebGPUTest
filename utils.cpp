#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <cassert>
#include <fstream>
#include <vector>


namespace {
  std::string adapterTypeToString(WGPUAdapterType type) {
    switch (type) {
    case WGPUAdapterType_DiscreteGPU:
      return "Discrete GPU";
    case WGPUAdapterType_IntegratedGPU:
      return "Integrated GPU";
    case WGPUAdapterType_CPU:
      return "CPU";
    case WGPUAdapterType_Unknown:
      return "Unknown";
    case WGPUAdapterType_Force32:
      return "Force32";
    default:
      return "Invalid";
    }
  }

  std::string adapterBackendToString(WGPUBackendType backend) {
    switch (backend) {
      case WGPUBackendType_Undefined:
        return "Undefined";
      case WGPUBackendType_Null:
        return "Null";
      case WGPUBackendType_WebGPU:
        return "WebGPU";
      case WGPUBackendType_D3D11:
        return "D3D11";
      case WGPUBackendType_D3D12:
        return "D3D12";
      case WGPUBackendType_Metal:
        return "Metal";
      case WGPUBackendType_Vulkan:
        return "Vulkan";
      case WGPUBackendType_OpenGL:
        return "OpenGL";
      case WGPUBackendType_OpenGLES:
        return "OpenGLES";
      case WGPUBackendType_Force32:
        return "Force32";
      default:
        return "Invalid";
    }
    
  }
}

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

    // Get adapter properties
    WGPUAdapterProperties properties{};
    wgpuAdapterGetProperties(adapter, &properties);
    std::cout << "Adapter name: " << properties.name << std::endl;
    std::cout << "Adapter vendor: " << properties.vendorName << std::endl;
    std::cout << "Adapter device id: " << properties.deviceID << std::endl;
    std::cout << "Adapter driver version: " << properties.driverDescription << std::endl;
    std::cout << "Adapter type: " << adapterTypeToString(properties.adapterType) << std::endl;
    std::cout << "Adapter backend type: " << adapterBackendToString(properties.backendType) << std::endl;
    std::cout << "Adapter compatibility mode: " << properties.compatibilityMode << std::endl;
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
    std::string buffer(file.tellg(), ' ');
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

WGPUTexture loadTexture(const std::filesystem::path &filePath, WGPUDevice device, WGPUTextureView &textureView)
{
    int width, height, channels;
    unsigned char *pixelData = stbi_load(filePath.string().c_str(), &width, &height, &channels, 4);
    if (nullptr == pixelData) return nullptr;

    WGPUTextureDescriptor textureDesc {};
    textureDesc.nextInChain = nullptr;
    textureDesc.label = std::string("Texture: " + filePath.string()).data();
    textureDesc.dimension = WGPUTextureDimension::WGPUTextureDimension_2D;
    textureDesc.size = { (unsigned int)width, (unsigned int)height, 1 };
    textureDesc.mipLevelCount = 1;
    textureDesc.sampleCount = 1;
    textureDesc.format = WGPUTextureFormat_RGBA8Unorm;
    textureDesc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
    textureDesc.viewFormatCount = 0;
    textureDesc.viewFormats = nullptr;

    WGPUTexture texture = wgpuDeviceCreateTexture(device, &textureDesc);
    // Upload data to the texture
    WGPUImageCopyTexture destination;
    destination.nextInChain = nullptr;
    destination.texture = texture;
    destination.mipLevel = 0;
    destination.origin = { 0, 0, 0 };
    destination.aspect = WGPUTextureAspect_All;

    WGPUTextureDataLayout source;
    source.offset = 0;
    source.bytesPerRow = 4 * textureDesc.size.width;
    source.rowsPerImage = textureDesc.size.height;

    auto queue = wgpuDeviceGetQueue(device);
    wgpuQueueWriteTexture(queue,
                          &destination,
                          pixelData,
                          textureDesc.size.width * textureDesc.size.height * 4,
                          &source,
                          &textureDesc.size);
    wgpuQueueRelease(queue);

    WGPUTextureViewDescriptor textureViewDesc;
    textureViewDesc.aspect = WGPUTextureAspect_All;
    textureViewDesc.baseArrayLayer = 0;
    textureViewDesc.arrayLayerCount = 1;
    textureViewDesc.baseMipLevel = 0;
    textureViewDesc.mipLevelCount = textureDesc.mipLevelCount;
    textureViewDesc.dimension = WGPUTextureViewDimension_2D;
    textureViewDesc.format = textureDesc.format;
    textureView = wgpuTextureCreateView(texture, &textureViewDesc);

    stbi_image_free(pixelData);

    return texture;
}

} // namespace Utils
