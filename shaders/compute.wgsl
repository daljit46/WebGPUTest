@group(0) @binding(0) var<uniform> matrix_length: u32;
@group(0) @binding(1) var<storage,read> inputBuffer1: array<f32>;
@group(0) @binding(2) var<storage,read> inputBuffer2: array<f32>;
@group(0) @binding(3) var<storage,read_write> outputBuffer: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>
)
{
    // let's get the row and column of the current invocation
    let row : u32 = group_id.x * 16 + local_id.x / 16u;
    let col : u32 = group_id.y * 16 + local_id.x % 16u;

    if (row >= matrix_length || col >= matrix_length) {
        return;
    }

    var sum = 0.0;
    for (var i: u32 = 0; i < matrix_length; i = i + 1) {
        sum = sum + inputBuffer1[row * matrix_length + i] * inputBuffer2[i * matrix_length + col];
    }

    outputBuffer[row * matrix_length + col] = sum;
}
