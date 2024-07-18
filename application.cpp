#include "application.h"
#include "utils.h"
#include "webgpu/webgpu.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>
#include <random>

constexpr int matrixSize = 10;

namespace {

bool verifyMatrixMultiplicationResult(const std::vector<float>& m1, std::span<const float> result)
{
    std::cout << "Verifying matrix multiplication result..." << std::endl;

    // compute m1 * m1
    std::vector<float> expected(matrixSize * matrixSize);
    for (auto i = 0; i < matrixSize; ++i) {
        for (auto j = 0; j < matrixSize; ++j) {
            expected[i * matrixSize + j] = 0;
            for (auto k = 0; k < matrixSize; ++k) {
                expected[i * matrixSize + j] += m1[i * matrixSize + k] * m1[k * matrixSize + j];
            }
        }
    }

    // compare the result with expected
    int mismatch_count = 0;
    for (auto i = 0; i < matrixSize; ++i) {
        for (auto j = 0; j < matrixSize; ++j) {
            if (std::abs(expected[i * matrixSize + j] - result[i * matrixSize + j]) > 1e-8) {
                std::cout << std::setprecision(10);
                std::cout << "Mismatch at (" << i << ", " << j << "): expected " << expected[i * matrixSize + j] << " but got " << result[i * matrixSize + j] << std::endl;
                std::cout << "Result differs by " << std::abs(expected[i * matrixSize + j] - result[i * matrixSize + j]) << std::endl;
                // return false;
                ++mismatch_count;
            }
        }
    }

    std::cout << "Mismatch count: " << mismatch_count << std::endl;

    std::cout << "Matrix multiplication result verified!" << std::endl;

    if(mismatch_count > 0) {
        return false;
    }
    return true;
}

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
    initBenchmark();
    initBindGroupLayout();
    initComputePipeline();
    initBuffers();
    initBindGroup();

    // Fill in the input buffer
    // generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    inputMatrix.reserve(m_bufferSize / sizeof(float));
    for (auto i = 0; i < matrixSize; ++i) {
        for (auto j = 0; j < matrixSize; ++j) {
            inputMatrix[i * matrixSize + j] = dis(gen);
        }
    }
}

Application::~Application()
{
    terminateBindGroup();
    terminateBuffers();
    terminateComputePipeline();
    terminateBindGroupLayout();
    terminateBenchmark();
    terminateDevice();
}

void Application::onCompute()
{
    std::vector<WGPUComputePassTimestampWrites> timestampWrites(2);
    timestampWrites[0].querySet = m_timestampQuerySet;
    timestampWrites[0].beginningOfPassWriteIndex = 0;
    timestampWrites[0].endOfPassWriteIndex = 1;

    wgpuQueueWriteBuffer(m_queue, m_inputBuffer1, 0, inputMatrix.data(), m_bufferSize);
    wgpuQueueWriteBuffer(m_queue, m_inputBuffer2, 0, inputMatrix.data(), m_bufferSize);

    WGPUCommandEncoderDescriptor encoderDesc = {};
    encoderDesc.nextInChain = nullptr;
    encoderDesc.label = "Command encoder";

    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(m_device, &encoderDesc);
    WGPUComputePassDescriptor computePassDesc = {};
    computePassDesc.label = "Compute pass";
    computePassDesc.nextInChain = nullptr;
    computePassDesc.timestampWrites = timestampWrites.data();

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
    resolveTimestamps(encoder);

    auto onQueueWorkDone = [](WGPUQueueWorkDoneStatus status, void*) {
        std::cout << "Queue work done with status: " << status << std::endl;
    };
    wgpuQueueOnSubmittedWorkDone(m_queue, onQueueWorkDone, (void*)this);

    auto command = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(m_queue, 1, &command);
    fetchTimestamps();

    // Set up callback for map buffer
    struct Context {
        WGPUBuffer mapBuffer = nullptr;
        int32_t size = 0;
        bool done = false;
        std::vector<float> *input = nullptr;
    };

    auto onBufferMapped = [](WGPUBufferMapAsyncStatus status, void* userdata) {
        auto *context = reinterpret_cast<Context*>(userdata);
        if (status == WGPUBufferMapAsyncStatus_Success) {
            auto *output = (const float*)wgpuBufferGetConstMappedRange(context->mapBuffer, 0, context->size);
            std::span<const float> outputSpan(output, context->size / sizeof(float));
            verifyMatrixMultiplicationResult(*context->input, outputSpan);
            std::cout << "Output buffer mapped!" << std::endl;
            wgpuBufferUnmap(context->mapBuffer);
        }
        else {
            throw std::runtime_error("Failed to map output buffer!");
        }
        context->done = true;
    };
    Context context{m_mapBuffer, m_bufferSize, false, &inputMatrix};
    wgpuBufferMapAsync(m_mapBuffer, WGPUBufferUsage_MapRead, 0, m_bufferSize, onBufferMapped, (void*)&context);

    while (!m_timestampFetched) {
        wgpuDeviceTick(m_device);
    }
}

void Application::initBenchmark()
{
    // init the timestamp query set
    WGPUQuerySetDescriptor querySetDesc = {};
    querySetDesc.nextInChain = nullptr;
    querySetDesc.type = WGPUQueryType_Timestamp;
    querySetDesc.count = 2;
    m_timestampQuerySet = wgpuDeviceCreateQuerySet(m_device, &querySetDesc);

    // init the timestamp resolve buffer
    WGPUBufferDescriptor timeStampBufferDesc = {};
    timeStampBufferDesc.nextInChain = nullptr;
    timeStampBufferDesc.label = "Timestamp buffer";
    timeStampBufferDesc.size = querySetDesc.count * sizeof(uint64_t);
    timeStampBufferDesc.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
    timeStampBufferDesc.mappedAtCreation = false;
    m_timestampResolveBuffer = wgpuDeviceCreateBuffer(m_device, &timeStampBufferDesc);

    // init the timestamp map buffer
    timeStampBufferDesc.label = "Timestamp map buffer";
    timeStampBufferDesc.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    m_timestampMapBuffer = wgpuDeviceCreateBuffer(m_device, &timeStampBufferDesc);
}

void Application::initDevice()
{
    WGPUInstanceDescriptor instanceDesc = {};
    WGPUDawnTogglesDescriptor dawnTogglesDesc = {};
    dawnTogglesDesc.chain.next = nullptr;
    dawnTogglesDesc.chain.sType = WGPUSType_DawnTogglesDescriptor;

    std::vector<const char*> enabledToggles = {
        "allow_unsafe_apis"
    };
    dawnTogglesDesc.enabledToggles = enabledToggles.data();
    dawnTogglesDesc.enabledToggleCount = enabledToggles.size();
    dawnTogglesDesc.disabledToggleCount = 0;

    instanceDesc.nextInChain = &dawnTogglesDesc.chain;

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
    requiredLimits.limits.maxComputeWorkgroupsPerDimension = 32;

    std::vector<WGPUFeatureName> features;
    if(wgpuAdapterHasFeature(m_adapter, WGPUFeatureName::WGPUFeatureName_TimestampQuery)) {
        features.push_back(WGPUFeatureName::WGPUFeatureName_TimestampQuery);
    }

    // Get logical device and queue
    WGPUDeviceDescriptor deviceDesc{};
    deviceDesc.nextInChain = nullptr;
    deviceDesc.label = "Device";
    deviceDesc.requiredFeatures = features.data();
    deviceDesc.requiredFeatureCount = static_cast<uint32_t>(features.size());
    deviceDesc.requiredLimits = &requiredLimits;
    deviceDesc.defaultQueue.nextInChain = nullptr;
    deviceDesc.defaultQueue.label = "Default queue";

    m_device = Utils::requestDevice(m_adapter, &deviceDesc);
    if(!m_device) {
        std::cerr << "Failed to get a device!" << std::endl;
        throw std::runtime_error("Failed to get a device!");
    }

    if(!wgpuDeviceHasFeature(m_device, WGPUFeatureName_TimestampQuery)) {
        std::cerr << "Device does not support timestamp query!" << std::endl;
        throw std::runtime_error("Device does not support timestamp query!");
    }

    m_queue = wgpuDeviceGetQueue(m_device);
    setWGPUCallbacks(m_device, m_queue);

    wgpuInstanceProcessEvents(m_instance);
}

void Application::initBindGroupLayout()
{
    // Input buffer 1
    WGPUBindGroupLayoutEntry inputBufferLayoutEntry1 = {};
    inputBufferLayoutEntry1.nextInChain = nullptr;
    inputBufferLayoutEntry1.binding = 0;
    inputBufferLayoutEntry1.visibility = WGPUShaderStage_Compute;
    inputBufferLayoutEntry1.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    // Input buffer 2
    WGPUBindGroupLayoutEntry inputBufferLayoutEntry2 = {};
    inputBufferLayoutEntry2.nextInChain = nullptr;
    inputBufferLayoutEntry2.binding = 1;
    inputBufferLayoutEntry2.visibility = WGPUShaderStage_Compute;
    inputBufferLayoutEntry2.buffer.type = WGPUBufferBindingType_ReadOnlyStorage;

    // Output buffer
    WGPUBindGroupLayoutEntry outputBufferLayoutEntry = {};
    outputBufferLayoutEntry.nextInChain = nullptr;
    outputBufferLayoutEntry.binding = 2;
    outputBufferLayoutEntry.visibility = WGPUShaderStage_Compute;
    outputBufferLayoutEntry.buffer.type = WGPUBufferBindingType_Storage;

    std::array<WGPUBindGroupLayoutEntry, 3> bindGroupLayoutEntries = {
        inputBufferLayoutEntry1,
        inputBufferLayoutEntry2,
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
    m_bufferSize = matrixSize * matrixSize * sizeof(float);

    // Create input buffers
    WGPUBufferDescriptor inputBufferDesc = {};
    inputBufferDesc.nextInChain = nullptr;
    inputBufferDesc.label = "Input buffer 1";
    inputBufferDesc.size = m_bufferSize;
    inputBufferDesc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;

    m_inputBuffer1 = wgpuDeviceCreateBuffer(m_device, &inputBufferDesc);
    inputBufferDesc.label = "Input buffer 2";
    m_inputBuffer2 = wgpuDeviceCreateBuffer(m_device, &inputBufferDesc);

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
    WGPUBindGroupEntry inputEntry1 = {};
    inputEntry1.nextInChain = nullptr;
    inputEntry1.binding = 0;
    inputEntry1.buffer = m_inputBuffer1;
    inputEntry1.offset = 0;
    inputEntry1.size = m_bufferSize;

    WGPUBindGroupEntry inputEntry2 = {};
    inputEntry2.nextInChain = nullptr;
    inputEntry2.binding = 1;
    inputEntry2.buffer = m_inputBuffer2;
    inputEntry2.offset = 0;
    inputEntry2.size = m_bufferSize;

    WGPUBindGroupEntry outputEntry = {};
    outputEntry.nextInChain = nullptr;
    outputEntry.binding = 2;
    outputEntry.buffer = m_outputBuffer;
    outputEntry.offset = 0;
    outputEntry.size = m_bufferSize;

    std::array<WGPUBindGroupEntry, 3> bindGroupEntries = {
        inputEntry1,
        inputEntry2,
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

void Application::terminateBenchmark()
{
    wgpuQuerySetRelease(m_timestampQuerySet);
    wgpuBufferDestroy(m_timestampResolveBuffer);
    wgpuBufferRelease(m_timestampResolveBuffer);
    wgpuBufferDestroy(m_timestampMapBuffer);
    wgpuBufferRelease(m_timestampMapBuffer);
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
    wgpuBufferDestroy(m_inputBuffer1);
    wgpuBufferRelease(m_inputBuffer1);
    wgpuBufferDestroy(m_inputBuffer2);
    wgpuBufferRelease(m_inputBuffer2);
    wgpuBufferDestroy(m_outputBuffer);
    wgpuBufferRelease(m_outputBuffer);
    wgpuBufferDestroy(m_mapBuffer);
    wgpuBufferRelease(m_mapBuffer);
}

void Application::terminateBindGroup()
{
    wgpuBindGroupRelease(m_bindGroup);
}

void Application::resolveTimestamps(WGPUCommandEncoder encoder)
{
    wgpuCommandEncoderResolveQuerySet(encoder, m_timestampQuerySet, 0, 2, m_timestampResolveBuffer, 0);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, m_timestampResolveBuffer, 0, m_timestampMapBuffer, 0, 2 * sizeof(uint64_t));
}

void Application::fetchTimestamps()
{
    auto onTimeStampBufferMapped = [](WGPUBufferMapAsyncStatus status, void* userdata) {
        auto app = reinterpret_cast<Application*>(userdata);
        std::cout << "Timestamp buffer mapped!" << std::endl;
        if (status == WGPUBufferMapAsyncStatus_Success) {
            auto *timestamps = (const uint64_t*)wgpuBufferGetConstMappedRange(app->m_timestampMapBuffer, 0, 2 * sizeof(uint64_t));
            // Calculate the time taken
            double timeTakenNanosecs = (timestamps[1] - timestamps[0]);
            double timeTakenMillisecs = timeTakenNanosecs / 1000000;
            std::cout << "Time taken: " << timeTakenMillisecs << "ms" << std::endl;
            std::cout << "Time taken: " << timeTakenNanosecs << "ns" << std::endl;
            wgpuBufferUnmap(app->m_timestampMapBuffer);
        }
        else {
            throw std::runtime_error("Failed to map buffer!");
        }
        app->m_timestampFetched = true;
    };
    wgpuBufferMapAsync(m_timestampMapBuffer, WGPUMapMode_Read, 0, 2 * sizeof(uint64_t), onTimeStampBufferMapped, (void*)this);
}

