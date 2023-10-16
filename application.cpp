#include "application.h"
#include "utils.h"

#include <GLFW/glfw3.h>
#include <glfw3webgpu.h>

#include <array>
#include <exception>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

namespace {
void setWGPUCallbacks(WGPUDevice device, WGPUQueue queue) {
    auto onDeviceError = [](WGPUErrorType type, char const *message,
                            void * /* pUserData */) {
        std::cout << "Uncaptured device error: type " << type;
        if (message)
            std::cout << " (" << message << ")";
        std::cout << std::endl;
        throw std::runtime_error(message);
    };
    auto onDeviceLost = [](WGPUDeviceLostReason reason, char const* message, void*){
        std::cout << "Device lost error: reason" << reason;
        if (message)
            std::cout << " (" << message << ")";
        std::cout << std::endl;
    };
    wgpuDeviceSetDeviceLostCallback(device, onDeviceLost, nullptr);
    wgpuDeviceSetUncapturedErrorCallback(device, onDeviceError,
                                         nullptr /* pUserData */);

    auto onQueueWorkDone = [](WGPUQueueWorkDoneStatus status,
                              void * /* pUserData */) {
        std::cout << "Queued work finished with status: " << status << std::endl;
    };
    wgpuQueueOnSubmittedWorkDone(queue, 0, onQueueWorkDone,
                                 nullptr /* pUserData */);
}

void onWindowResize(GLFWwindow *window, int /* width */, int /* height */) {
    // We know that even though from GLFW's point of view this is
    // "just a pointer", in our case it is always a pointer to an
    // instance of the class `Application`
    auto that = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));

    // Call the actual class-member callback
    if (that != nullptr)
        that->onResize();
}

void setGLFWcallbacks(GLFWwindow *window) {
    auto onMouseMove = [](GLFWwindow *window, double x, double y) {
        auto that =
            reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        if (that != nullptr)
            that->onMouseMove(x, y);
    };
    glfwSetCursorPosCallback(window, onMouseMove);

    auto onMouseButton = [](GLFWwindow *window, int button, int action,
                            int /* mods */) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        auto that =
            reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        if (that != nullptr)
            that->onMouseButton(button, action, 0);
    };

    glfwSetMouseButtonCallback(window, onMouseButton);

    auto onScroll = [](GLFWwindow *window, double x, double y) {
        auto that = reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        if(that != nullptr){
            that->onScroll(x, y);
        }
    };

    glfwSetScrollCallback(window, onScroll);

    auto onWindowResize = [](GLFWwindow *window, int width, int height) {
        auto that =
            reinterpret_cast<Application *>(glfwGetWindowUserPointer(window));
        if (that != nullptr)
            that->onResize();
    };

    glfwSetFramebufferSizeCallback(window, onWindowResize);
}

} // namespace

Application::Application()
{
    // Create Window
    if(!glfwInit()) {
        std::cerr << "Failed to initialise GLFW!" << std::endl;
        throw std::runtime_error("Failed to initialise GLFW!");
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(m_uniforms.windowWidth, m_uniforms.windowHeight, "WebGPU Test", nullptr, nullptr);

    if(!m_window) {
        glfwTerminate();
        throw std::runtime_error("Failed to open window!");
    }
    glfwSetWindowUserPointer(m_window, this);
    setGLFWcallbacks(m_window);
    // Create WGPU instance
    WGPUInstanceDescriptor desc{};
    desc.nextInChain = nullptr;
    m_instance = wgpuCreateInstance(&desc);

    if(!m_instance){
        throw std::runtime_error("Failed to initialise WebGPU!");
    }

    // Get surface
    m_surface = glfwGetWGPUSurface(m_instance, m_window);
    if(!m_surface) {
        throw std::runtime_error("Failed to initialise surface!");
    }

    // Get Physical Adapter
    WGPURequestAdapterOptions adapterOpts{};
    adapterOpts.nextInChain = nullptr;
    adapterOpts.compatibleSurface = m_surface;

    m_adapter = Utils::requestAdapter(m_instance, &adapterOpts);

    std::vector<WGPUFeatureName> featuresList;
    featuresList.resize(wgpuAdapterEnumerateFeatures(m_adapter, nullptr));
    wgpuAdapterEnumerateFeatures(m_adapter, featuresList.data());

    // Setup limits
    WGPUSupportedLimits supportedLimits {};
    wgpuAdapterGetLimits(m_adapter, &supportedLimits);

    WGPURequiredLimits requiredLimits {};
    requiredLimits.limits.maxVertexAttributes = 2;
    requiredLimits.limits.maxVertexBuffers = 1;
    requiredLimits.limits.maxBufferSize = 5 * 4 * sizeof(float);
    requiredLimits.limits.maxVertexBufferArrayStride = 5 * sizeof(float);
    requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;
    requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
    requiredLimits.limits.maxBindGroups = 1;
    requiredLimits.limits.maxUniformBuffersPerShaderStage = 1;
    requiredLimits.limits.maxUniformBufferBindingSize = 16 * sizeof(float);

    // Get logical device and queue
    WGPUDeviceDescriptor deviceDesc{};
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = "Device";
    deviceDesc.requiredFeatureCount = 0;
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = "Default queue";

    m_device = Utils::requestDevice(m_adapter, &deviceDesc);
    if(!m_device) {
        std::cerr << "Failed to get a device!" << std::endl;
        throw std::runtime_error("Failed to get a device!");
    }
    m_queue = wgpuDeviceGetQueue(m_device);

    setWGPUCallbacks(m_device, m_queue);

    // Setup swapchain
    buildSwapchain();

    // Upload vertex and uniform data to the GPU
    constexpr uint32_t vertexDataSize = 5;

    std::array<float, 20> vertexData = {
        // x,   y,     r,   g,   b
        -1.0F, -1.0F, 1.0F, 0.0F, 0.0F,
         1.0F, 1.0F,  0.0F, 1.0F, 0.0F,
        -1.0F, 1.0F,  0.0F, 0.0F, 1.0F,
         1.0F, -1.0F, 1.0F, 1.0F, 0.0F
    };
    std::array<uint16_t, 6> indexData = { 0, 1, 2, 0, 3, 1 };
    m_vertexCount = static_cast<int>(vertexData.size() / vertexDataSize);
    m_indexCount = static_cast<int>(indexData.size());

    WGPUBufferDescriptor vertexBufferDesc{};
    vertexBufferDesc.nextInChain = nullptr;
    vertexBufferDesc.label = "Vertex buffer";
    vertexBufferDesc.size = vertexData.size() * sizeof(float);
    vertexBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Vertex;
    vertexBufferDesc.mappedAtCreation = false;
    m_vertexBuffer = wgpuDeviceCreateBuffer(m_device, &vertexBufferDesc);

    WGPUBufferDescriptor indexBufferDesc{};
    indexBufferDesc.nextInChain = nullptr;
    indexBufferDesc.label = "Index buffer";
    indexBufferDesc.size = indexData.size() * sizeof(uint16_t);
    indexBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Index;
    indexBufferDesc.mappedAtCreation = false;
    m_indexBuffer = wgpuDeviceCreateBuffer(m_device, &indexBufferDesc);

    WGPUBufferDescriptor uniformBufferDesc {};
    uniformBufferDesc.nextInChain = nullptr;
    uniformBufferDesc.label = "Uniform buffer";
    uniformBufferDesc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Uniform;
    uniformBufferDesc.size = sizeof(Uniform);
    uniformBufferDesc.mappedAtCreation = false;
    m_uniformBuffer = wgpuDeviceCreateBuffer(m_device, &uniformBufferDesc);

    wgpuQueueWriteBuffer(m_queue, m_vertexBuffer, 0, vertexData.data(), vertexBufferDesc.size);
    wgpuQueueWriteBuffer(m_queue, m_indexBuffer, 0, indexData.data(), indexBufferDesc.size);
    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &m_uniforms, sizeof(Uniform));

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

    // Create binding group
    WGPUBindGroupLayoutEntry bindingLayout = Utils::createDefaultBindingLayout();
    bindingLayout.nextInChain = nullptr;
    bindingLayout.binding = 0;
    bindingLayout.visibility = WGPUShaderStage_Vertex | WGPUShaderStage_Fragment;
    bindingLayout.buffer.type = WGPUBufferBindingType_Uniform;
    bindingLayout.buffer.minBindingSize = sizeof(Uniform);

    WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc {};
    bindGroupLayoutDesc.nextInChain = nullptr;
    bindGroupLayoutDesc.entryCount = 1;
    bindGroupLayoutDesc.entries = &bindingLayout;
    WGPUBindGroupLayout bindingGroupLayout = wgpuDeviceCreateBindGroupLayout(m_device, &bindGroupLayoutDesc);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc {};
    pipelineLayoutDesc.nextInChain = nullptr;
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &bindingGroupLayout;
    WGPUPipelineLayout pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDesc);

    WGPUBindGroupEntry binding {};
    binding.nextInChain = nullptr;
    binding.binding = 0;
    binding.buffer = m_uniformBuffer;
    binding.offset = 0;
    binding.size = sizeof(Uniform);

    WGPUBindGroupDescriptor bindGroupDesc {};
    bindGroupDesc.nextInChain = nullptr;
    bindGroupDesc.layout = bindingGroupLayout;
    bindGroupDesc.entryCount = bindGroupLayoutDesc.entryCount;
    bindGroupDesc.entries = &binding;
    m_bindGroup = wgpuDeviceCreateBindGroup(m_device, &bindGroupDesc);


    // Load shaders and setup render pipeline
    m_shaderModule = Utils::loadShaderModule("./shaders/shader.wgsl", m_device);
    std::cout << "Shader module: " << m_shaderModule << std::endl;

    WGPURenderPipelineDescriptor pipelineDesc{};
    pipelineDesc.nextInChain = nullptr;

    pipelineDesc.vertex.bufferCount = 1;
    pipelineDesc.vertex.buffers = &vertexBufferLayout;
    pipelineDesc.vertex.module = m_shaderModule;
    pipelineDesc.vertex.entryPoint = "vs_main";
    pipelineDesc.vertex.constantCount = 0;
    pipelineDesc.vertex.constants = nullptr;

    pipelineDesc.primitive.topology = WGPUPrimitiveTopology_TriangleList;
    pipelineDesc.primitive.stripIndexFormat = WGPUIndexFormat_Undefined;
    pipelineDesc.primitive.frontFace = WGPUFrontFace_CCW;
    pipelineDesc.primitive.cullMode = WGPUCullMode_None;


    WGPUFragmentState fragmentState {};
    fragmentState.nextInChain = nullptr;
    fragmentState.module = m_shaderModule;
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
    colorTarget.format = m_swapChainFormat;
    colorTarget.blend = &blendState;
    colorTarget.writeMask = WGPUColorWriteMask_All;


    fragmentState.targetCount = 1;
    fragmentState.targets = &colorTarget;

    pipelineDesc.depthStencil = nullptr;
    pipelineDesc.layout = pipelineLayout;

    pipelineDesc.multisample.count = 1;
    pipelineDesc.multisample.mask = ~0u;
    pipelineDesc.multisample.alphaToCoverageEnabled = false;

    m_renderPipeline = wgpuDeviceCreateRenderPipeline(m_device, &pipelineDesc);
    std::cout << "Render pipeline: " << m_renderPipeline << std::endl;
    m_previousFrameTime = glfwGetTime();
}

bool Application::isRunning() const
{
    return !glfwWindowShouldClose(m_window);
}

void Application::onFrame()
{
    double currentFrameTime = glfwGetTime();
    double deltaTime = currentFrameTime - m_previousFrameTime;
    m_previousFrameTime = currentFrameTime;
    m_frameTimesList.push_back(1.0f / deltaTime);

    // Calculate average frame rate of last 10 frames
    float frameRate = [&](){
        if(m_frameTimesList.size() > 10) {
            return std::reduce(m_frameTimesList.end() - 10, m_frameTimesList.end()) / 10.0F;
        }
        return std::reduce(m_frameTimesList.begin(), m_frameTimesList.end()) / static_cast<float>(m_frameTimesList.size());
    }();

    std::string title = "WebGPU " + std::to_string(static_cast<int32_t>(frameRate)) + " FPS";
    glfwSetWindowTitle(m_window, title.c_str());
    glfwPollEvents();
    WGPUTextureView nextTexture = wgpuSwapChainGetCurrentTextureView(m_swapChain);
    if (!nextTexture) {
        std::cerr << "Cannot acquire next swap chain texture" << std::endl;
        return;
    }

    WGPUCommandEncoderDescriptor commandEncoderDesc{};
    commandEncoderDesc.nextInChain = nullptr;
    commandEncoderDesc.label = "Command Encoder";
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &commandEncoderDesc);

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

    // Update uniform buffer
//    float t = static_cast<float>(glfwGetTime()) * 2;
//    m_uniforms.scale = std::abs(std::sin(t/12));

//    std::cout << "Uniforms: \n";
//    std::cout << "  scale: " << m_uniforms.scale << std::endl;
//    std::cout <<  " center: " << m_uniforms.center[0] << ", " << m_uniforms.center[1] << std::endl;
//    std::cout << "  width: " << m_uniforms.windowWidth << std::endl;
//    std::cout << " height: " << m_uniforms.windowHeight << std::endl;

    wgpuQueueWriteBuffer(m_queue, m_uniformBuffer, 0, &m_uniforms, sizeof(Uniform));

    WGPURenderPassEncoder renderPass = wgpuCommandEncoderBeginRenderPass(encoder, &renderPassDesc);
    wgpuRenderPassEncoderSetPipeline(renderPass, m_renderPipeline);
    wgpuRenderPassEncoderSetVertexBuffer(renderPass, 0, m_vertexBuffer, 0, m_vertexCount * 5 * sizeof(float));
    wgpuRenderPassEncoderSetIndexBuffer(renderPass, m_indexBuffer, WGPUIndexFormat_Uint16, 0, m_indexCount * sizeof(uint16_t));
    wgpuRenderPassEncoderSetBindGroup(renderPass, 0, m_bindGroup, 0, nullptr);
    wgpuRenderPassEncoderDrawIndexed(renderPass, m_indexCount, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(renderPass);


    wgpuTextureViewRelease(nextTexture);

    WGPUCommandBufferDescriptor cmdBufferDescriptor{};
    cmdBufferDescriptor.nextInChain = nullptr;
    cmdBufferDescriptor.label = "Command buffer";

    WGPUCommandBuffer command = wgpuCommandEncoderFinish(encoder, &cmdBufferDescriptor);
    wgpuQueueSubmit(m_queue, 1, &command);
    wgpuSwapChainPresent(m_swapChain);

    // Check for pending errors
    wgpuDeviceTick(m_device);
}

void Application::onFinish()
{
    wgpuSwapChainRelease(m_swapChain);
    wgpuDeviceRelease(m_device);
    wgpuSurfaceRelease(m_surface);
    wgpuAdapterRelease(m_adapter);
    wgpuInstanceRelease(m_instance);

    glfwDestroyWindow(m_window);
    glfwTerminate();
}

void Application::onResize()
{
    buildSwapchain();
}

void Application::buildSwapchain()
{
    std::cout << "Buildig swap chain" << std::endl;

    int width, height;
    glfwGetFramebufferSize(m_window, &width, &height);

    m_uniforms.windowWidth = width;
    m_uniforms.windowHeight = height;

    if (m_swapChain != nullptr) {
        wgpuSwapChainRelease(m_swapChain);
    }


    WGPUSwapChainDescriptor swapChainDesc{};
    swapChainDesc.nextInChain = nullptr;
    swapChainDesc.width = width;
    swapChainDesc.height = height;

    m_swapChainFormat = WGPUTextureFormat_BGRA8Unorm;
    swapChainDesc.format = m_swapChainFormat;
    swapChainDesc.usage = WGPUTextureUsage_RenderAttachment;
    swapChainDesc.presentMode = WGPUPresentMode_Immediate;
    m_swapChain = wgpuDeviceCreateSwapChain(m_device, m_surface, &swapChainDesc);
    std::cout << "Swapchain: " << m_swapChain << std::endl;

    if(!m_swapChain) {
        throw std::runtime_error("Failed to create swapChain!");
    }
}

void Application::onMouseMove(double x, double y) {
//    std::cout << "Mouse moved to (" << x << ", " << y << ")" << std::endl;
    if(m_mouseState == MouseState::Dragging){
        double diffX = x - m_previousMouseX;
        double diffY = y - m_previousMouseY;
        m_uniforms.offset[0] += static_cast<float>(diffX);
        m_uniforms.offset[1] += static_cast<float>(diffY);
        m_previousMouseX = x;
        m_previousMouseY = y;
    }
}

void Application::onScroll(double x, double y)
{
    constexpr float minScale = 0.1F;
    constexpr float maxScale = 100000.0F;

    float desiredScale = m_uniforms.scale + static_cast<float>(y/10.F * m_uniforms.scale);
    float newScale = std::clamp(desiredScale, minScale, maxScale);
    m_uniforms.offset[0] = m_uniforms.offset[0] * newScale / m_uniforms.scale;
    m_uniforms.offset[1] = m_uniforms.offset[1] * newScale / m_uniforms.scale;
    m_uniforms.scale = newScale;
}

void Application::onMouseButton(int button, int action, int mods) {
    if(button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        m_mouseState = MouseState::Dragging;
        glfwGetCursorPos(m_window, &m_previousMouseX, &m_previousMouseY);
    }
    else {
        m_mouseState = MouseState::Idle;
    }
}
