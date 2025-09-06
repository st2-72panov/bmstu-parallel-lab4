#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

void fill_array(float* A, const int& N) {
    for (int i = 0; i < N; ++i)
        A[i] = 1.0f / std::rand();
}

///////////////////////////////////////////////////////////
// CPU

void scaleVectorCPU(const float* A, float* B, const int N, const float k) {
    for (int i = 0; i < N; ++i)
        B[i] = A[i] * k;
}

void test_cpu(const int N, const float k) {
    float* A = (float*) malloc(N * sizeof(float));
    float* B = (float*) malloc(N * sizeof(float));
    fill_array(A, N);

    auto start = std::chrono::high_resolution_clock::now();
    scaleVectorCPU(A, B, N, k);
    auto end = std::chrono::high_resolution_clock::now();

    free(A);
    free(B);

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU: " << duration.count() / 1000.0 << "ms" << std::endl;
}

///////////////////////////////////////////////////////////
// GPU

__global__ void scaleVectorCUDA(const float* A, float* B, int N, float k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        B[idx] = A[idx] * k;
}

void test_GPU(const int N, const float k) {
    const auto THREADS_PER_BLOCK = 32;

    float* h_A = (float*) malloc(N * sizeof(float));
    float* h_B = (float*) malloc(N * sizeof(float));
    fill_array(h_A, N);
    
    float duration;
    float *d_A, *d_B;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start_e, end_e;
    cudaEventCreate(&start_e);
    cudaEventCreate(&end_e);
    cudaEventRecord(start_e);
    scaleVectorCUDA<<<std::ceil(1.0 * N / THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>(d_A, d_B, N, k);
    cudaEventRecord(end_e);
    cudaEventSynchronize(end_e);
    cudaEventElapsedTime(&duration, start_e, end_e);
    cudaEventDestroy(start_e);
    cudaEventDestroy(end_e);

    cudaMemcpy(h_B, d_B, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    std::cout << "GPU: " << duration << "ms" << std::endl;
}

///////////////////////////////////////////////////////////

int main() {
    const int N = 1 * 1000 * 1000;
    const float k = 2.5f;
    test_cpu(N, k);
    test_GPU(N, k);
}
