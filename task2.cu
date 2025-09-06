#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

void fill_array(unsigned char* A, const int& N) {
    for (int i = 0; i < N; ++i)
        A[i] = std::rand() % 256;
}

///////////////////////////////////////////////////////////
// CPU

void thresholdFilterCPU(
    unsigned char* input,
    unsigned char* output,
    const int width,
    const int height,
    unsigned char threshold
) {
    const int size = width * height;
    for (int i = 0; i < size; ++i)
        output[i] = (input[i] > threshold) ? 255 : 0;
}

void test_cpu(const int width, const int height, unsigned char threshold) {
    const int size = width * height;
    const int size_bytes = size * sizeof(unsigned char);

    unsigned char* A = (unsigned char*) malloc(size_bytes);
    unsigned char* B = (unsigned char*) malloc(size_bytes);
    fill_array(A, size);

    auto start = std::chrono::high_resolution_clock::now();
    thresholdFilterCPU(A, B, width, height, threshold);
    auto end = std::chrono::high_resolution_clock::now();

    free(A);
    free(B);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU: " << duration.count() / 1000.0 << "ms" << std::endl;
}

///////////////////////////////////////////////////////////
// GPU

__global__ void thresholdFilterCUDA(
    unsigned char* input,
    unsigned char* output, 
    const int width,
    const int height,
    unsigned char threshold
) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < width && y < height) {
        const int idx = x + y * width;
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

void test_GPU(const int width, const int height, unsigned char threshold) {
    const auto BLOCK_WIDTH = 8;  // 8 * 8 = 64 threads per block
    const int size = width * height;
    const int size_bytes = size * sizeof(unsigned char);

    unsigned char* h_A = (unsigned char*) malloc(size_bytes);
    unsigned char* h_B = (unsigned char*) malloc(size_bytes);
    fill_array(h_A, size);

    unsigned char *d_A, *d_B;
    cudaMalloc(&d_A, size_bytes);
    cudaMalloc(&d_B, size_bytes);
    cudaMemcpy(d_A, h_A, size_bytes, cudaMemcpyHostToDevice);
    dim3 blockSize(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridSize(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

    float duration;
    cudaEvent_t start_e, end_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&end_e);
    cudaEventRecord(start_e);

    thresholdFilterCUDA<<<gridSize, blockSize>>>(d_A, d_B, width, height, threshold);
    
    cudaEventRecord(end_e);
    cudaEventSynchronize(end_e);
    cudaEventElapsedTime(&duration, start_e, end_e);
    cudaEventDestroy(start_e);
    cudaEventDestroy(end_e);
    std::cout << "GPU: " << duration << "ms" << std::endl;
    
    cudaMemcpy(h_B, d_B, size_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);
}

///////////////////////////////////////////////////////////

int main() {
    const int W = 1024, H = 1024, T = 128;
    test_cpu(W, H, T);
    test_GPU(W, H, T);
}
