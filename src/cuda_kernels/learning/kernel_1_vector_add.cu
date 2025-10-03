// Kernel 1: Element-wise Vector Addition
// Learning objectives:
//   - CUDA kernel syntax (__global__ vs __device__)
//   - Thread indexing (blockIdx, blockDim, threadIdx)
//   - Grid-stride loops for handling large arrays
//   - PyTorch integration

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: runs on GPU
// __global__ means: "this function can be called from CPU, runs on GPU"
__global__ void vector_add_kernel(
    const float* a,      // Input array 1
    const float* b,      // Input array 2
    float* c,            // Output array
    int n                // Array size
) {
    // ========================================
    // YOUR TASK 1: Calculate this thread's global index
    // ========================================
    // 
    // CUDA organizes threads in a 2-level hierarchy:
    //   - Grid: collection of blocks
    //   - Block: collection of threads
    //
    // Built-in variables you can use:
    //   blockIdx.x  = which block this thread is in (0, 1, 2, ...)
    //   blockDim.x  = how many threads per block (typically 256 or 512)
    //   threadIdx.x = which thread within the block (0 to blockDim.x-1)
    //
    // Formula: global_index = (block_number * threads_per_block) + thread_within_block
    //
    // Replace the ??? below:
    
    int idx = ???;  // Calculate: blockIdx.x * blockDim.x + threadIdx.x
    
    // ========================================
    // YOUR TASK 2: Grid-stride loop
    // ========================================
    //
    // If we have 10,000 elements but only 1024 threads, each thread needs
    // to process multiple elements.
    //
    // Pattern:
    //   for (int i = my_starting_index; i < total_elements; i += total_threads) {
    //       process element i
    //   }
    //
    // total_threads = blockDim.x * gridDim.x
    //   - blockDim.x = threads per block
    //   - gridDim.x  = number of blocks
    //
    // Replace the ??? below:
    
    int stride = ???;  // Calculate: blockDim.x * gridDim.x
    
    for (int i = idx; i < n; i += stride) {
        // ========================================
        // YOUR TASK 3: The actual computation
        // ========================================
        //
        // For element i:
        //   - Read a[i] from global memory
        //   - Read b[i] from global memory  
        //   - Add them together
        //   - Write result to c[i]
        //
        // Replace the ??? below:
        
        c[i] = ???;  // Calculate: a[i] + b[i]
    }
}

// C++ wrapper function: called from Python
// This handles the PyTorch tensor → CUDA kernel interface
torch::Tensor vector_add_cuda(torch::Tensor a, torch::Tensor b) {
    // Input validation
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "Input tensors must have same size");
    
    const int n = a.size(0);
    
    // Allocate output tensor on GPU
    auto c = torch::zeros_like(a);
    
    // ========================================
    // YOUR TASK 4: Kernel launch configuration
    // ========================================
    //
    // We need to decide:
    //   - How many threads per block?
    //   - How many blocks total?
    //
    // Common choices:
    //   - threads_per_block = 256 (good default for most GPUs)
    //   - num_blocks = (total_work + threads_per_block - 1) / threads_per_block
    //     This rounds up to ensure we have enough threads
    //
    // Example: 1000 elements, 256 threads/block
    //   num_blocks = (1000 + 256 - 1) / 256 = 1255 / 256 = 4 blocks
    //   Total threads = 4 * 256 = 1024 threads (enough to cover 1000 elements)
    //
    // Replace the ??? below:
    
    const int threads_per_block = 256;
    const int num_blocks = ???;  // Calculate: (n + threads_per_block - 1) / threads_per_block
    
    // Launch the kernel
    // Syntax: kernel_name<<<num_blocks, threads_per_block>>>(args...)
    vector_add_kernel<<<num_blocks, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        c.data_ptr<float>(),
        n
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel launch failed: ") + 
                                cudaGetErrorString(err));
    }
    
    return c;
}

// Python binding using pybind11
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("vector_add", &vector_add_cuda, "Vector addition (CUDA)");
}

