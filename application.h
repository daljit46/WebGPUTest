#include <webgpu/webgpu.h>

#include <array>
#include <vector>

struct GLFWwindow;

class Application
{
public:
    Application();

    enum class MouseState { Idle, Dragging };

    bool isRunning() const;

    void onFrame();

    void onFinish();

    void onResize();

    void onMouseMove(double x, double y);

    void onScroll(double x, double y);

    void onMouseButton(int button, int action, int mods);
private:
    void buildSwapchain();
    bool initGui();
    void terminateGui();
    void updateGui(WGPURenderPassEncoder pass);

    struct Uniform {
        int32_t windowWidth = 800;
        int32_t windowHeight = 600;
    };
    static_assert(sizeof(Uniform) % sizeof(int32_t) == 0);

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
    WGPUTexture m_texture = nullptr;
    WGPUBindGroup m_bindGroup = nullptr;
    WGPUShaderModule m_shaderModule = nullptr;
    GLFWwindow *m_window = nullptr;
    int m_vertexCount = 0;
    int m_indexCount = 0;

    std::vector<float> m_frameTimesList;
    double m_previousFrameTime = 0.0;
    MouseState m_mouseState = MouseState::Idle;
    double m_previousMouseX = 0.0;
    double m_previousMouseY = 0.0;

    float m_monitorScale = 1.0F;
};
