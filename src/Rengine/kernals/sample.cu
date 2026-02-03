#include <cuda.h>
#include <cuda_runtime.h>

__global__ void greedy_sample_kernel(
    const float* logits,     // [batch, vocab]
    int vocab_size,
    float temperature,
    int* out_token_ids,      // [batch]
    float* out_logprobs      // [batch]
){
    int tid = threadIdx.x;
    int b = blockIdx.x;
    logits += b * vocab_size;

    float local_max = -1e38f;
    int local_idx = 0;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float val = logits[i] / temperature;
        if (val > local_max) { local_max = val; local_idx = i; }
    }

    __shared__ float s_max[32];
    __shared__ int s_idx[32];
    
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
        if (other_max > local_max) { local_max = other_max; local_idx = other_idx; }
    }
    
    int warp_id = tid / 32, lane = tid % 32;
    if (lane == 0) { s_max[warp_id] = local_max; s_idx[warp_id] = local_idx; }
    __syncthreads();

    if (warp_id == 0) {
        local_max = (lane < blockDim.x / 32) ? s_max[lane] : -1e38f;
        local_idx = (lane < blockDim.x / 32) ? s_idx[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (other_max > local_max) { local_max = other_max; local_idx = other_idx; }
        }
    }

    __shared__ float global_max;
    __shared__ int global_idx;
    if (tid == 0) { global_max = local_max; global_idx = local_idx; }
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x)
        sum += expf((logits[i] / temperature) - global_max);
    
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    
    __shared__ float s_sum[32];
    if (lane == 0) s_sum[warp_id] = sum;
    __syncthreads();
    
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < blockDim.x / 32; i++) total += s_sum[i];
        out_token_ids[b] = global_idx;
        out_logprobs[b] = (logits[global_idx] / temperature) - global_max - logf(total);
    }
}