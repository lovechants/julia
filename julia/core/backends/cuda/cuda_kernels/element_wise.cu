/* 
Might write some ptx for some of these functions since its a bit easier to understand 
Need to test 1080s -> 4090s at some point 
*/

// Multiply two vec 
#include <cmath>
__global__ void multiply_vectors(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] * b[idx];
  }
}

// Divide two vec 
__global__ void divide_vectors(const float *a, const float *b, float *c, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] / b[idx];
  }
}

// ReLU activiation 
__global__ void relu(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = fmaxf(0.0f, input[idx]);
  }
}

// Sigmoid activation
__global__ void sigmoid(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = 1.0f / (1.0f + expf(-input[idx]));
  }
}

// Tanh activation
__global__ void tanh_kernel(const float *input, float *output, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = tanhf(input[idx]) 
  }
}

// Element-wise power function 
__global__ void pow_kernel(const float *input, float *output, float exponent, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = powf(input[idx], exponent)
  }
}

// Element-wise sqrt 
__global__ void sqrt_kernel(const float *input, float *output, float exponent, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = sqrtf(input[idx])
  }
}

// Add scalar to vec 
__global__ void add_scalar(const float *input, float *output, float scalar, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] + scalar;
  }
}

// Multiply vec by scalar 
__global__ void multiply_scalar(const float *input, float *output, float scalar, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    output[idx] = input[idx] * scalar;
  }
}
