#include <webgpu/webgpu.h>

#include <array>
#include <vector>


class Application
{
public:
    Application();
    ~Application();

    void onCompute();

private:
    void initBenchmark();
    void initDevice();
    void initBindGroupLayout();
    void initComputePipeline();
    void initBuffers();
    void initBindGroup();

    void terminateBenchmark();
    void terminateDevice();
    void terminateBindGroupLayout();
    void terminateComputePipeline();
    void terminateBuffers();
    void terminateBindGroup();

    void resolveTimestamps(WGPUCommandEncoder commandEncoder);
    void fetchTimestamps();

    WGPUInstance m_instance = nullptr;
    WGPUAdapter m_adapter = nullptr;
    WGPUDevice m_device = nullptr;
    WGPUQueue m_queue = nullptr;
    WGPUComputePipeline m_computePipeline = nullptr;
    WGPUPipelineLayout m_pipelineLayout = nullptr;
    WGPUBuffer m_inputBuffer1 = nullptr;
    WGPUBuffer m_inputBuffer2 = nullptr;
    WGPUBuffer m_outputBuffer = nullptr;
    WGPUBuffer m_mapBuffer = nullptr;
    WGPUBindGroupLayout m_bindGroupLayout = nullptr;
    WGPUBindGroup m_bindGroup = nullptr;
    WGPUShaderModule m_shaderModule = nullptr;
    WGPUQuerySet m_timestampQuerySet = nullptr;
    WGPUBuffer m_timestampResolveBuffer = nullptr;
    WGPUBuffer m_timestampMapBuffer = nullptr;
    bool m_timestampFetched = false;

    int32_t m_bufferSize = 0;
};
