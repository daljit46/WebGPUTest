@group(0) @binding(0) var<storage,read> inputBuffer1: array<f32,25>;
@group(0) @binding(1) var<storage,read> inputBuffer2: array<f32,25>;
@group(0) @binding(2) var<storage,read_write> outputBuffer: array<f32,25>;


@compute @workgroup_size(32, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    const matrix_size : u32 = 5;
    // the input buffer is a 100x100 matrix of floats
    // let's get the row and column of the current invocation
    let row = global_id.x / matrix_size;
    let col = matrix_size - (global_id.x % matrix_size) - 1;

    // c_mn = a_mk * b_kn

    // we need to calculate the dot product of the row of the first matrix and the column of the second matrix
    var sum = 0.0;
    for (var i: u32 = 0; i < matrix_size; i = i + 1) {
        sum = sum + inputBuffer1[row * matrix_size + i] * inputBuffer2[i * matrix_size + col];
    }

    // store the result in the output buffer
    outputBuffer[row * matrix_size + col] = sum;
}