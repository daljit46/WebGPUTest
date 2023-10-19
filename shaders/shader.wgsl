struct Uniforms {
    windowWidth: i32,
    windowHeight: i32,
};

struct VertexInput {
    @location(0) position: vec2f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> uUniformData: Uniforms;
@group(0) @binding(1) var texture: texture_2d<f32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.position = vec4f(in.position, 0.0, 1.0);
    return out;
}


@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f{
    let color = textureLoad(texture, vec2i(in.position.xy), 0).rgb;

    return vec4f(color, 1.0);
}
