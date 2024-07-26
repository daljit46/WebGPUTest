// We are going to blockwise dot product of a row in inputBuffer1 and a column in inputBuffer2
// The size of the block will be the size of the workgroup
@group(0) @binding(0) var<uniform> matrix_length: u32;
@group(0) @binding(1) var<storage,read> inputBuffer1: array<f32>;
@group(0) @binding(2) var<storage,read> inputBuffer2: array<f32>;
@group(0) @binding(3) var<storage,read_write> outputBuffer: array<f32>;

// Submatrices of inputBuffer1 and inputBuffer2 that are loaded into shared memory
// within a workgroup

const block_size : u32 = 8;
var<workgroup> subA: array<array<f32, block_size>, block_size>;
var<workgroup> subB: array<array<f32, block_size>, block_size>;

@compute @workgroup_size(1, block_size, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>)
{
    let x : u32 = group_id.x * block_size + local_id.x;
    let y : u32 = group_id.y * block_size + local_id.y;

    // Each thread will be responsible for computing 8 entries in the row y in the outputBuffer
    var dotProducts : array<f32, block_size>;
    for (var i = 0u; i < block_size; i = i + 1u) {
        dotProducts[i] = 0.0;
    }

    for(var blockIndex = 0u; blockIndex < matrix_length; blockIndex += block_size) {
        // Load data into shared memory submatrices
        for(var i = 0u; i < block_size; i = i + 1u) {
            subA[local_id.y][i] = inputBuffer1[y * matrix_length + (blockIndex + i)];
            subB[local_id.y][i] = inputBuffer2[(blockIndex + local_id.y) * matrix_length + x + i];
        }
        // Synchronize to make sure all threads have loaded their data
        workgroupBarrier();

        // Dot product on the cached row of subA and cached column of subB
        for (var k = 0u; k < block_size; k++) {
            for(var j = 0u; j < block_size; j++) {
                dotProducts[j] += subA[local_id.y][k] * subB[k][local_id.x + j];
            }
        }
        // Synchronize before the next iteration
        workgroupBarrier();
    }

    if (y < matrix_length && x < matrix_length) {
        for (var j = 0u; j < block_size; j++) {
            let col = x + j;
            outputBuffer[y * matrix_length + col] = dotProducts[j];
        }
    }
}
