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
__global__ void matrix_multiply_shared( const float *a, const float *b, float *c, int a_rows, int a_cols, int b_cols) {
  __shared__ float shared_a[16][16];
  __shared__ float shared_b[16][16];

  int bx - blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * 16 + ty;
  int col = bx * 16 + tx;

  for (int m=0; m < (a_cols + 15) / 16; m++) {
    if (row < a_rows && m * 16 + tx < a_cols) {
      shared_a[ty][tx] = a[row * a_cols + m * 16 + tx];
    } else {
      shared_a[ty][tx] = 0.0f;
    }

    if (m * 16 + ty < a_cols && col < b_cols) {
      shared_b[ty][tx] = b[(m * 16 * ty) * b_cols + col];
    } else {
      shared_b[ty][tx] = 0.0f;
    }

    __syncthreads();

    // Partial Dot Product 
    for(int k = 0; k < 16; k++) {
      sum += shared_a[ty][k] * shared_b[k][tx]; 
    }

    __syncthreads();
  }

  if (row < a_rows && cols < b_cols) {
    c[row * b_cols + cols] = sum;
  }
}

// Batched matrix mul 
__global__ void matrix_multiply_batch( const float *a, const float *b, float *c, int a_rows, int a_cols, int b_cols, int batch_size) {
  int batch = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.y;

  if(batch < batch_size && row < a_rows && col < b_cols) {
    float sum = 0.0f;
    int a_offset = batch * a_rows * a_cols;
    int b_offset = batch * a_rows * b_cols;
    int c_offset = batch * a_rows * b_cols;

    for (int k = 0; k < a_cols; k++) {
      sum += a[a_offset + row * a_cols + k] * b[b_offset + k * b_cols + col];
    }

    c[c_offset + row * b_cols + col] = sum;
  }
}


// Matrix element-wise add 
__global__ void matrix_add(const float *a, const float *b, float *c, int rows, int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < rows && cols < cols) {
    int idx = rows * cols + cols;
    c[idx] = a[idx] + b[idx];
  }
}
