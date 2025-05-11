/* 
   Just because I'm really into them right now and the second order gradients 
   1. compute the finite difference 
   2. compute diagonal 
   3. compute the hessian vector product = [H(\theta + epsilon{v}) - H(theta)] / epsilon
  */ 

/** 
  Kernel to compute finite difference approximation of Hessian matrix 
  Uses central difference formula 

  @param params: Parameters at which to compute the Hessian 
  @param hessian: Output Hessian matrix (nParams x nParams)
  @param nParamas: Number of parameters 
  @param epsiolon: Small value for finite difference approximation 
  @param compute_f: Function pointer to compute objective Function
*/

#include <cmath>
template <typename FuncType>
__global__ void compute_hessian_finite_diff(const float* params, float* hessian, int nParams, float epsilon, FuncType compute_f) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < nParams && j < nParams) {
    extern __shared__ float shared_params[];
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      for (int k = 0; k < nParams; k++) {
        shared_params[k] == params[k];
      }
    }

    __syncthreads();
    float h_i = epsilon * (fabsf(shared_params[i]) < epsilon ? 1.0f : fabsf(shared_params[i]));
    float h_j = epsilon * (fabsf(shared_params[j]) < epsilon ? 1.0f : fabsf(shared_params[j]));
    // TODO finish the rest of this 
  }
}
