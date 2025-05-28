#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
using namespace std;

#define BLOCK_SIZE 16

__global__ void matMulSummaKernel(double* A, double* B, double* C, int A_rows, int A_cols, int B_cols) {
    __shared__ double A_tile[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double B_tile[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double sum = 0;

    for (int k = 0; k < (A_cols + BLOCK_SIZE - 1) / BLOCK_SIZE; ++k) {
        if (row < A_rows && k * BLOCK_SIZE + threadIdx.x < A_cols)
            A_tile[threadIdx.y][threadIdx.x] = A[row * A_cols + k * BLOCK_SIZE + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0;

        if (k * BLOCK_SIZE + threadIdx.y < A_cols && col < B_cols)
            B_tile[threadIdx.y][threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * B_cols + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int n = 0; n < BLOCK_SIZE; ++n)
            sum += A_tile[threadIdx.y][n] * B_tile[n][threadIdx.x];

        __syncthreads();
    }

    if (row < A_rows && col < B_cols)
        C[row * B_cols + col] = sum;
}

void readMatrix(const string& filename, vector<double>& mat, int& rows, int& cols) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        exit(1);
    }
    file >> rows >> cols;
    mat.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i)
        file >> mat[i];
    file.close();
}

void writeMatrix(const string& filename, const vector<double>& mat, int rows, int cols) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot write file: " << filename << endl;
        return;
    }
    file << rows << " " << cols << "\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            file << fixed << setprecision(2) << mat[i * cols + j] << " ";
        file << "\n";
    }
    file.close();
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: ./matmul_cuda A.txt B.txt C.txt\n";
        return 1;
    }

    vector<double> h_A, h_B, h_C;
    int A_rows, A_cols, B_rows, B_cols;

    readMatrix(argv[1], h_A, A_rows, A_cols);
    readMatrix(argv[2], h_B, B_rows, B_cols);

    if (A_cols != B_rows) {
        cerr << "Matrix dimension mismatch: " << A_cols << " vs " << B_rows << endl;
        return 1;
    }

    h_C.resize(A_rows * B_cols, 0);

    double* d_A, * d_B, * d_C;
    size_t size_A = A_rows * A_cols * sizeof(double);
    size_t size_B = B_rows * B_cols * sizeof(double);
    size_t size_C = A_rows * B_cols * sizeof(double);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((B_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (A_rows + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matMulSummaKernel<<<blocks, threads>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cerr << "CUDA kernel error: " << cudaGetErrorString(err) << endl;
        return 1;
    }

    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    writeMatrix(argv[3], h_C, A_rows, B_cols);

    cout << "CUDA SUMMA matrix multiplication done. Time: " << milliseconds / 1000.0 << " sec\n";

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
