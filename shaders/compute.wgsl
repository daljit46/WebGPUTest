// We are going to blockwise dot product of a row in inputBuffer1 and a column in inputBuffer2
// The size of the block will be the size of the workgroup
@group(0) @binding(0) var<uniform> matrix_length: u32;
@group(0) @binding(1) var<storage,read> inputBuffer1: array<f32>;
@group(0) @binding(2) var<storage,read> inputBuffer2: array<f32>;
@group(0) @binding(3) var<storage,read_write> outputBuffer: array<f32>;

// Submatrices of inputBuffer1 and inputBuffer2 that are loaded into shared memory
// within a workgroup
var<workgroup> subA: array<f32, 64>;
var<workgroup> subB: array<f32, 64>;

const block_size : u32 = 8;

@compute @workgroup_size(block_size, block_size, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) group_id: vec3<u32>
)
{
    let x : u32 = group_id.x * block_size + local_id.x;
    let y : u32 = group_id.y * block_size + local_id.y;
    var tmp : f32 = 0.0;

    for(var blockIndex = 0u; blockIndex < matrix_length; blockIndex += block_size) {
        // Load data into shared memory
        // TODO: checks to ensure we don't access out-of-bound elements if matrix is not a multiple of block_size
        subA[local_id.y * block_size + local_id.x] = inputBuffer1[y * matrix_length + (blockIndex + local_id.x)];
        subB[local_id.y * block_size + local_id.x] = inputBuffer2[(blockIndex + local_id.y) * matrix_length + x];

        // Finish for all threads in the workgroup to finish
        workgroupBarrier();

        // Dot product on the cached row of subA and cached column of subB
        for(var dotIndex = 0u; dotIndex < block_size; dotIndex += 1) {
            tmp += subA[local_id.y * block_size + dotIndex] * subB[dotIndex * block_size + local_id.x];
        }

        // Wait until all threads are done
        // We don't want some threads start fetching the next block
        workgroupBarrier();
    }

    if(y < matrix_length || x < matrix_length) {
        outputBuffer[y * matrix_length + x] = tmp;
    }
}
