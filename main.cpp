#include "utils.h"

#include <GLFW/glfw3.h>
#include <webgpu/webgpu.h>
#include <webgpu/webgpu_glfw.h>
#include <glfw3webgpu.h>

#include <iostream>
#include <vector>


int main()
{
    if(!glfwInit()) {
        std::cerr << "Failed to initialise GLFW!" << std::endl;
        return -1;
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow *window = glfwCreateWindow(640, 480, "WebGPU Test", nullptr, nullptr);

    if(!window) {
        std::cerr << "Failed to open window!" << std::endl;
        glfwTerminate();
        return -1;
    }

    WGPUInstanceDescriptor desc{};
    desc.nextInChain = nullptr;
    WGPUInstance instance = wgpuCreateInstance(&desc);

    if(!instance){
        std::cerr << "Failed to initialise WebGPU!" << std::endl;
        return -1;
    }

    WGPUSurface surface = glfwGetWGPUSurface(instance, window);
    if(!surface) {
        std::cerr << "Failed to initialise surface!" << std::endl;
        return -1;
    }

    WGPURequestAdapterOptions adapterOpts{};
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = surface;

    WGPUAdapter adapter = Utils::requestAdapter(instance, &adapterOpts);

    std::vector<WGPUFeatureName> featuresList;
    featuresList.resize(wgpuAdapterEnumerateFeatures(adapter, nullptr));
    wgpuAdapterEnumerateFeatures(adapter, featuresList.data());

    WGPUDeviceDescriptor deviceDesc{};
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = "Device";
    deviceDesc.requiredFeatureCount = 0;
    deviceDesc.requiredLimits = nullptr;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = "Default queue";

    WGPUDevice device = Utils::requestDevice(adapter, &deviceDesc);
    if(!device) {
        std::cerr << "Failed to get a device!" << std::endl;
        return -1;
    }

    auto onDeviceError = [](WGPUErrorType type, char const* message, void* /* pUserData */) {
        std::cout << "Uncaptured device error: type " << type;
        if (message) std::cout << " (" << message << ")";
        std::cout << std::endl;
    };
    wgpuDeviceSetUncapturedErrorCallback(device, onDeviceError, nullptr /* pUserData */);

    WGPUQueue queue = wgpuDeviceGetQueue(device);
    auto onQueueWorkDone = [](WGPUQueueWorkDoneStatus status, void* /* pUserData */) {
        std::cout << "Queued work finished with status: " << status << std::endl;
    };
    wgpuQueueOnSubmittedWorkDone(queue, 0, onQueueWorkDone, nullptr /* pUserData */);


    WGPUSwapChainDescriptor swapChainDesc{};
    swapChainDesc.nextInChain = nullptr;
    swapChainDesc.width = 640;
    swapChainDesc.height = 480;

    WGPUTextureFormat swapChainFormat = WGPUTextureFormat_BGRA8Unorm;
    swapChainDesc.format = swapChainFormat;
    swapChainDesc.usage = WGPUTextureUsage_RenderAttachment;
    swapChainDesc.presentMode = WGPUPresentMode_Fifo;
    WGPUSwapChain swapChain = wgpuDeviceCreateSwapChain(device, surface, &swapChainDesc);
    std::cout << "Swapchain: " << swapChain << std::endl;

    constexpr uint32_t vertexDataSize = 5;

    std::vector<float> vertexData = {
        -0.5f, -0.5f, 1.0f, 0.0f, 0.0f,
         0.0f,  0.5f, 0.2f, 1.0f, 0.0f,
         0.5f, -0.5f, 0.0f, 0.0f, 1.0f,
    };
    const int vertexCount = static_cast<int>(vertexData.size() / vertexDataSize);

    WGPUBufferDescriptor vertexBufferDesc{};
    vertexBufferDesc.nextInChain = nullptr;
    vertexBufferDesc.label = "Vertex buffer";
    vertexBufferDesc.size = vertexData.size() * sizeof(float);
    vertexBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
    vertexBufferDesc.mappedAtCreation = false;
    WGPUBuffer vertexBuffer = wgpuDeviceCreateBuffer(device, &vertexBufferDesc);

    // upload vertex data to the GPU
    wgpuQueueWriteBuffer(queue, vertexBuffer, 0, vertexData.data(), vertexBufferDesc.size);

    WGPUVertexAttribute positionAttrib{};
    positionAttrib.shaderLocation = 0;
    positionAttrib.format = WGPUVertexFormat_Float32x2;
    positionAttrib.offset = 0;

    WGPUVertexAttribute colorAttrib{};
    colorAttrib.shaderLocation = 1;
    colorAttrib.format = WGPUVertexFormat_Float32x3;
    colorAttrib.offset = 2 * sizeof(float);

    std::vector<WGPUVertexAttribute> vertexAttributes = { positionAttrib, colorAttrib };

    WGPUVertexBufferLayout vertexBufferLayout{};
    vertexBufferLayout.attributeCount = static_cast<uint32_t>(vertexAttributes.size());
    vertexBufferLayout.attributes = vertexAttributes.data();
    vertexBufferLayout.arrayStride = vertexDataSize * sizeof(float);
    vertexBufferLayout.stepMode = WGPUVertexStepMode_Vertex;


    WGPUShaderModule shaderModule = Utils::loadShaderModule("./shaders/shader.wgsl", device);
    std::cout << "Shader module: " << shaderModule << std::endl;

    WGPURenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.nextInChain = nullptr;

    pipelineDesc.vertex.bufferCount = 1;
    pipelineDesc.vertex.buffers = &vertexBufferLayout;
    pipelineDesc.vertex.module = shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;

    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    pipelineDesc.primitive.cullMode = WGPUCullMode_None;


    WGPUFragmentState fragmentState {};
    fragmentState.nextInChain = nullptr;
    fragmentState.module = shaderModule;
    fragmentState.entryPoint = "fs_main";
    fragmentState.constantCount = 0;
    fragmentState.constants = nullptr;
    pipelineDesc.fragment = &fragmentState;

    WGPUBlendState blendState {};
    blendState.color.srcFactor = WGPUBlendFactor_SrcAlpha;
    blendState.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
    blendState.color.operation = WGPUBlendOperation_Add;
    blendState.alpha.srcFactor = WGPUBlendFactor_Zero;
    blendState.alpha.dstFactor = WGPUBlendFactor_One;
    blendState.alpha.operation = WGPUBlendOperation_Add;

    WGPUColorTargetState colorTarget {};
    colorTarget.nextInChain = nullptr;
    colorTarget.format = swapChainFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = WGPUColorWriteMask_All;


    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    pipelineDesc.depthStencil = nullptr;
    pipelineDesc.layout = nullptr;

    pipelineDesc.multisample.count = 1;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    WGPURenderPipeline pipeline = wgpuDeviceCreateRenderPipeline(device, &pipelineDesc);
    std::cout << "Render pipeline: " << pipeline << std::endl;

    // main loop
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();
        WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(swapChain);
        if (!nextTexture) {
            std::cerr << "Cannot acquire next swap chain texture" << std::endl;
            break;
        }


        WGPUCommandEncoderDescriptor commandEncoderDesc{};
        commandEncoderDesc.nextInChain = nullptr;
        commandEncoderDesc.label = "Command Encoder";
        WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &commandEncoderDesc);

        WGPURenderPassColorAttachment renderPassColorAttachment{};
        renderPassColorAttachment.view = nextTexture;
        renderPassColorAttachment.resolveTarget = nullptr;
        renderPassColorAttachment.loadOp = WGPULoadOp_Clear;
        renderPassColorAttachment.storeOp = WGPUStoreOp_Store;
        renderPassColorAttachment.clearValue = WGPUColor{ 0.0, 0.1, 0.2, 1.0 };

        WGPURenderPassDescriptor renderPassDesc{};
        renderPassDesc.nextInChain = nullptr;
        renderPassDesc.colorAttachmentCount = 1;
        renderPassDesc.colorAttachments = &renderPassColorAttachment;
        renderPassDesc.depthStencilAttachment = nullptr;
        renderPassDesc.timestampWriteCount = 0;
        renderPassDesc.timestampWrites = nullptr;


        WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
        wgpuRenderPassEncoderSetPipeline(renderPass, pipeline);
        wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, vertexBuffer, 0, vertexData.size() * sizeof(float));
        wgpuRenderPassEncoderDraw(renderPass, vertexCount, 1, 0, 0);
        wgpuRenderPassEncoderEnd(renderPass);


        wgpuTextureViewRelease(nextTexture);

        WGPUCommandBufferDescriptor cmdBufferDescriptor{};
        cmdBufferDescriptor.nextInChain = nullptr;
        cmdBufferDescriptor.label = "Command buffer";

        WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, &cmdBufferDescriptor);
        wgpuQueueSubmit(queue, 1, &command);
        wgpuSwapChainPresent(swapChain);
    }

    wgpuSwapChainRelease(swapChain);
    wgpuDeviceRelease(device);
    wgpuSurfaceRelease(surface);
    wgpuAdapterRelease(adapter);
    wgpuInstanceRelease(instance);

    glfwDestroyWindow(window);
    glfwTerminate();
}
