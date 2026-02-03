#include <torch/extension.h>
#include <cuda_runtime.h>

// Forward declare the CUDA kernel
__global__ void greedy_sample_kernel(
    const float* logits,
    int vocab_size,
    float temperature,
    int* out_token_ids,
    float* out_logprobs
);

// C++ wrapper that launches the kernel
torch::Tensor greedy_sample_cuda(
    torch::Tensor logits,      // [batch, vocab_size]
    float temperature
) {
    TORCH_CHECK(logits.is_cuda(), "logits must be a CUDA tensor");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D [batch, vocab_size]");
    TORCH_CHECK(logits.scalar_type() == torch::ScalarType::Float, "logits must be float32");
    
    int batch_size = logits.size(0);
    int vocab_size = logits.size(1);
    
    auto options = torch::TensorOptions()
        .dtype(torch::kInt32)
        .device(logits.device());
    auto token_ids = torch::empty({batch_size}, options);
    
    auto logprob_options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(logits.device());
    auto logprobs = torch::empty({batch_size}, logprob_options);
    
    int threads = 256;
    int blocks = batch_size;
    
    greedy_sample_kernel<<<blocks, threads>>>(
        logits.data_ptr<float>(),
        vocab_size,
        temperature,
        token_ids.data_ptr<int>(),
        logprobs.data_ptr<float>()
    );
    
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess, 
        "CUDA kernel failed: ", cudaGetErrorString(error));
    
    return token_ids;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("greedy_sample", &greedy_sample_cuda, 
          "Greedy sampling using custom CUDA kernel",
          py::arg("logits"),
          py::arg("temperature") = 1.0f);
}
