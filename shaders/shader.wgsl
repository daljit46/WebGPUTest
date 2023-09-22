@group(0) @binding(0) var<uniform> uTime: f32;

struct VertexInput {
    @location(0) position: vec2f,
    @location(1) color: vec3f,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) color: vec3f,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    var offset = vec2f(cos(uTime), sin(uTime));
    var new_position = in.position + 0.5 * offset;
    out.position = vec4f(new_position, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f{
    return vec4f(in.color, 1.0);
}
