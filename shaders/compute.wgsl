@group(0) @binding(0) var<storage,read> inputBuffer: array<f32,64>;
@group(0) @binding(1) var<storage,read_write> outputBuffer: array<f32,64>;

fn f (x: f32) -> f32 {
    return x * x * x;
}

@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    outputBuffer[global_id.x] = f(inputBuffer[global_id.x]);
}