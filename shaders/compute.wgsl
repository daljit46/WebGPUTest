const matrix_length : u32 = 10;
const matrix_size : u32 = matrix_length * matrix_length;

@group(0) @binding(0) var<storage,read> inputBuffer1: array<f32,matrix_size>;
@group(0) @binding(1) var<storage,read> inputBuffer2: array<f32,matrix_size>;
@group(0) @binding(2) var<storage,read_write> outputBuffer: array<f32,matrix_size>;


@compute @workgroup_size(32)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if(global_id.x >= matrix_size || global_id.y != 0 || global_id.z != 0) {
        return;
    }
    // let's get the row and column of the current invocation
    let row : u32 = global_id.x / matrix_length;
    let col : u32 = (global_id.x % matrix_length);

    // c_mn = a_mk * b_kn

    // we need to calculate the dot product of the row of the first matrix and the column of the second matrix
    var sum = 0.0;
    for (var i: u32 = 0; i < matrix_length; i = i + 1) {
        sum = sum + inputBuffer1[row * matrix_length + i] * inputBuffer2[i * matrix_length + col];
    }

    // store the result in the output buffer
    outputBuffer[global_id.x] = sum;
}
