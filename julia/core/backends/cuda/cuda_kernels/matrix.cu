/* 
   A bunch of kernels to write 
   Basics 
   1. Transpose
   2. Matrix Mul with shared mem 
   3. Simple mat mul 
   4. Batch mat mul 
   5. Matrix element-wise add 
*/ 

// Transpose 
__global__ void transpose(const float *input, float *output, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx < col && idy < rows) {
    output[idx * rows + idy] = input[idy * cols + idx];
  }
}

// Simple mat mul 
__global__ void matrix_multiply(const float *a, const float *b, float *c, int a_rows, int a_cols, int b_cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < a_rows && col < b_cols) {
    float sum = 0.0f;
    for (int k = 0; k < a_cols; k++) {
      sum += a[row * a_cols + k] * b[k * b_cols + col];
    }
    c[row * b_cols + col] = sum;
  }
}

// Shared mem mat mul 
