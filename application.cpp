#include "application.h"
#include "utils.h"

#include <iostream>

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
    wgpuQueueOnSubmittedWorkDone(queue, onQueueWorkDone,
                                 nullptr /* pUserData */);
}
}

Application::Application()
{
    initDevice();
    initBindGroupLayout();
    initComputePipeline();
    initBuffers();
    initBindGroup();
}

Application::~Application()
{
    terminateBindGroup();
    terminateBuffers();
    terminateComputePipeline();
    terminateBindGroupLayout();
    terminateDevice();
}

void Application::onCompute()
{
    // Fill in the input buffer
    std::vector<float> input(m_bufferSize / sizeof(float));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<float>(i) * 0.1F;
    }
    wgpuQueueWriteBuffer(m_queue, m_inputBuffer, 0, input.data(), m_bufferSize);

    WGPUCommandEncoderDescriptor encoderDesc = {};
    encoderDesc.nextInChain = nullptr;
    encoderDesc.label = "Command encoder";

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encoderDesc);
    WGPUComputePassDescriptor computePassDesc = {};
    computePassDesc.label = "Compute pass";
    computePassDesc.nextInChain = nullptr;
    computePassDesc.timestampWrites = nullptr;

    WGPUComputePassEncoder computePass = wgpuCommandEncoderBeginComputePass(encoder, &computePassDesc);
    wgpuComputePassEncoderSetPipeline(computePass, m_computePipeline);
    wgpuComputePassEncoderSetBindGroup(computePass, 0, m_bindGroup, 0, nullptr);

    // Deduce number of workgroups
    uint32_t invocationCount = m_bufferSize / sizeof(float);
    uint32_t workgroupSize = 32;
    uint32_t workgroupCount = std::ceil(invocationCount / static_cast<float>(workgroupSize));

    std::cout << "Invocations: " << invocationCount << " Workgroup size: " << workgroupSize << " Workgroup count: " << workgroupCount << std::endl;

    wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroupCount, 1, 1);

    wgpuComputePassEncoderEnd(computePass);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, m_outputBuffer, 0, m_mapBuffer, 0, m_bufferSize);

    auto command = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(m_queue, 1, &command);
     
        // Set up callback for map buffer
    struct Context {
        WGPUBuffer inputBuffer = nullptr;
        WGPUBuffer mapBuffer = nullptr;
        int32_t size = 0;
        bool done = false;
        std::vector<float> *input = nullptr;
    };

    auto onBufferMapped = [](WGPUBufferMapAsyncStatus status, void* userdata) {
        auto *context = reinterpret_cast<Context*>(userdata);
        if (status == WGPUBufferMapAsyncStatus_Success) {
            auto *output = (const float*)wgpuBufferGetConstMappedRange(context->mapBuffer, 0, context->size);
            for(size_t i = 0; i < context->size / sizeof(float); ++i) {
                std::cout << "Input: " << context->input->at(i) << " Output: " << output[i] << std::endl;
            }
            wgpuBufferUnmap(context->mapBuffer);
        }
        else {
            throw std::runtime_error("Failed to map buffer!");
        }
        context->done = true;
    };
    Context context{m_inputBuffer, m_mapBuffer, m_bufferSize, false, &input};
    wgpuBufferMapAsync(m_mapBuffer, WGPUBufferUsage_MapRead, 0, m_bufferSize, onBufferMapped, (void*)&context);

    while(!context.done) {
        wgpuInstanceProcessEvents(m_instance);
    }
}

void Application::initDevice()
{
    WGPUInstanceDescriptor instanceDesc = {};
    instanceDesc.nextInChain = nullptr;

    m_instance = wgpuCreateInstance(&instanceDesc);

    WGPURequestAdapterOptions adapterOpts{};
    adapterOpts.nextInChain = nullptr;

    m_adapter = Utils::requestAdapter(m_instance, &adapterOpts);

        // Setup limits
    WGPUSupportedLimits supportedLimits {};
    wgpuAdapterGetLimits(m_adapter, &supportedLimits);

    WGPURequiredLimits requiredLimits {};
    requiredLimits.limits.maxVertexAttributes = 2;
    requiredLimits.limits.maxVertexBuffers = 1;
    requiredLimits.limits.maxBufferSize = 16 * 4 * sizeof(float);
    requiredLimits.limits.minStorageBufferOffsetAlignment = supportedLimits.limits.minStorageBufferOffsetAlignment;
    requiredLimits.limits.minUniformBufferOffsetAlignment = supportedLimits.limits.minUniformBufferOffsetAlignment;
    requiredLimits.limits.maxBindGroups = 2;
    requiredLimits.limits.maxStorageBuffersPerShaderStage = 2;
    requiredLimits.limits.maxStorageBufferBindingSize = m_bufferSize;
    requiredLimits.limits.maxComputeWorkgroupSizeX = 32;
    requiredLimits.limits.maxComputeWorkgroupSizeY = 1;
    requiredLimits.limits.maxComputeWorkgroupSizeZ = 1;
    requiredLimits.limits.maxComputeInvocationsPerWorkgroup = 32;
    requiredLimits.limits.maxComputeWorkgroupsPerDimension = 2;

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

    wgpuInstanceProcessEvents(m_instance);
}

void Application::initBindGroupLayout()
{
    // Input buffer
    WGPUBindGroupLayoutEntry inputBufferLayoutEntry = {};
    inputBufferLayoutEntry.nextInChain = nullptr;
    inputBufferLayoutEntry.binding = 0;
    inputBufferLayoutEntry.visibility = WGPUShaderStage_Compute;
    inputBufferLayoutEntry.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    // Output buffer
    WGPUBindGroupLayoutEntry outputBufferLayoutEntry = {};
    outputBufferLayoutEntry.nextInChain = nullptr;
    outputBufferLayoutEntry.binding = 1;
    outputBufferLayoutEntry.visibility = WGPUShaderStage_Compute;
    outputBufferLayoutEntry.buffer.type = WGPUBufferBindingType_Storage;

    std::array<WGPUBindGroupLayoutEntry, 2> bindGroupLayoutEntries = {
        inputBufferLayoutEntry,
        outputBufferLayoutEntry
    };

    WGPUBindGroupLayoutDescriptor bindGroupLayoutDesc = {};
    bindGroupLayoutDesc.nextInChain = nullptr;
    bindGroupLayoutDesc.label = "Bind group layout";
    bindGroupLayoutDesc.entryCount = static_cast<uint32_t>(bindGroupLayoutEntries.size());
    bindGroupLayoutDesc.entries = bindGroupLayoutEntries.data();
    m_bindGroupLayout = wgpuDeviceCreateBindGroupLayout(m_device, &bindGroupLayoutDesc);
}

void Application::initComputePipeline()
{
    WGPUShaderModule computeShader = Utils::loadShaderModule("shaders/compute.wgsl", m_device);

    WGPUPipelineLayoutDescriptor pipelineLayoutDesc = {};
    pipelineLayoutDesc.nextInChain = nullptr;
    pipelineLayoutDesc.label = "Pipeline layout";
    pipelineLayoutDesc.bindGroupLayoutCount = 1;
    pipelineLayoutDesc.bindGroupLayouts = &m_bindGroupLayout;
    m_pipelineLayout = wgpuDeviceCreatePipelineLayout(m_device, &pipelineLayoutDesc);

    WGPUComputePipelineDescriptor computePipelineDesc = {};
    computePipelineDesc.nextInChain = nullptr;
    computePipelineDesc.label = "Compute pipeline";
    computePipelineDesc.compute.entryPoint = "main";
    computePipelineDesc.compute.module = computeShader;
    computePipelineDesc.layout = m_pipelineLayout;

    m_computePipeline = wgpuDeviceCreateComputePipeline(m_device, &computePipelineDesc);
}

void Application::initBuffers()
{
    m_bufferSize = 64 * sizeof(float);

    // Create input buffers
    WGPUBufferDescriptor inputBufferDesc = {};
    inputBufferDesc.nextInChain = nullptr;
    inputBufferDesc.label = "Input buffer";
    inputBufferDesc.size = m_bufferSize;
    inputBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    m_inputBuffer = wgpuDeviceCreateBuffer(m_device, &inputBufferDesc);

    // Create output buffers
    WGPUBufferDescriptor outputBufferDesc = {};
    outputBufferDesc.nextInChain = nullptr;
    outputBufferDesc.label = "Output buffer";
    outputBufferDesc.size = m_bufferSize;
    outputBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;

    m_outputBuffer = wgpuDeviceCreateBuffer(m_device, &outputBufferDesc);

    WGPUBufferDescriptor mapBufferDesc = {};
    mapBufferDesc.nextInChain = nullptr;
    mapBufferDesc.label = "Map buffer";
    mapBufferDesc.size = m_bufferSize;
    mapBufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;

    m_mapBuffer = wgpuDeviceCreateBuffer(m_device, &mapBufferDesc);
}

void Application::initBindGroup()
{
    WGPUBindGroupEntry inputEntry = {};
    inputEntry.nextInChain = nullptr;
    inputEntry.binding = 0;
    inputEntry.buffer = m_inputBuffer;
    inputEntry.offset = 0;
    inputEntry.size = m_bufferSize;

    WGPUBindGroupEntry outputEntry = {};
    outputEntry.nextInChain = nullptr;
    outputEntry.binding = 1;
    outputEntry.buffer = m_outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = m_bufferSize;

    std::array<WGPUBindGroupEntry, 2> bindGroupEntries = {
        inputEntry,
        outputEntry
    };

    WGPUBindGroupDescriptor bindGroupDesc = {};
    bindGroupDesc.nextInChain = nullptr;
    bindGroupDesc.label = "Bind group";
    bindGroupDesc.layout = m_bindGroupLayout;
    bindGroupDesc.entryCount = static_cast<uint32_t>(bindGroupEntries.size());
    bindGroupDesc.entries = bindGroupEntries.data();
    m_bindGroup = wgpuDeviceCreateBindGroup(m_device, &bindGroupDesc);
}

void Application::terminateDevice()
{
    wgpuDeviceRelease(m_device);
    wgpuAdapterRelease(m_adapter);
    wgpuInstanceRelease(m_instance);
}

void Application::terminateBindGroupLayout()
{
    wgpuBindGroupLayoutRelease(m_bindGroupLayout);
}

void Application::terminateComputePipeline()
{
    wgpuComputePipelineRelease(m_computePipeline);
    wgpuPipelineLayoutRelease(m_pipelineLayout);
}

void Application::terminateBuffers()
{
    wgpuBufferDestroy(m_inputBuffer);
    wgpuBufferRelease(m_inputBuffer);
    wgpuBufferDestroy(m_outputBuffer);
    wgpuBufferRelease(m_outputBuffer);
    wgpuBufferDestroy(m_mapBuffer);
    wgpuBufferRelease(m_mapBuffer);
}

void Application::terminateBindGroup()
{
    wgpuBindGroupRelease(m_bindGroup);
}
