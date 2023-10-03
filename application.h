#include <webgpu/webgpu.h>

#include <array>

class GLFWwindow;

class Application
{
public:
    Application();

    bool isRunning() const;

    void onFrame();

    void onFinish();

    void onResize();

    void onMouseMove(double x, double y);

    void onMouseButton(int button, int action, int mods);
private:
    void buildSwapchain();

    struct Uniform {
        std::array<float, 2> center = { 0.0F, 0.0F }; // 8 bytes
        float scale = 1.0F; // 4 bytes
        // add padding to make the size of the struct to be multiple of 8 bytes
        // this is required by the WebGPU spec
        float padding; // bytes
    };
    static_assert(sizeof(Uniform) % sizeof(std::array<float, 2>) == 0);

    Uniform m_uniforms;
    WGPUInstance m_instance = nullptr;
    WGPUAdapter m_adapter = nullptr;
    WGPUDevice m_device = nullptr;
    WGPUSurface m_surface = nullptr;
    WGPUQueue m_queue = nullptr;
    WGPURenderPipeline m_renderPipeline;
    WGPUSwapChain m_swapChain = nullptr;
    WGPUTextureFormat m_swapChainFormat = WGPUTextureFormat_Undefined;
    WGPUBuffer m_indexBuffer = nullptr;
    WGPUBuffer m_vertexBuffer = nullptr;
    WGPUBuffer m_uniformBuffer = nullptr;
    WGPUBindGroup m_bindGroup = nullptr;
    WGPUShaderModule m_shaderModule = nullptr;
    GLFWwindow *m_window = nullptr;
    int m_vertexCount = 0;
    int m_indexCount = 0;

    double m_previousFrameTime = 0.0;
};
