import numpy as np
from julia.core.tensor import Tensor
"""
Optimizers 
"""

"""
SGD 
https://www.semanticscholar.org/paper/A-Stochastic-Approximation-Method-Robbins/34ddd8865569c2c32dec9bf7ffc817ff42faaa01?p2df
https://projecteuclid.org/journalArticle/Download?urlId=10.1214%2Faoms%2F1177729392&isResultClick=False
https://arxiv.org/pdf/1606.04838
For Nesterov -> https://proceedings.mlr.press/v28/sutskever13.pdf
"""

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.0, dampening=0, weight_decay=0, nesterov=False):
        """
        Implements stochastic gradient descent with optional momentum.
        
        Args:
            params: list of parameters to optimize
            lr: learning rate
            momentum: momentum factor (default: 0)
            dampening: dampening for momentum (default: 0)
            weight_decay: weight decay (L2 penalty) (default: 0)
            nesterov: enables Nesterov momentum (default: False)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        self.state = {}  # Parameter-specific state

    def step(self):
        """Performs a single optimization step."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            # Apply momentum
            if self.momentum > 0:
                param_id = id(param)
                if param_id not in self.state:
                    self.state[param_id] = {'momentum_buffer': np.zeros_like(grad)}
                
                buf = self.state[param_id]['momentum_buffer']
                buf = self.momentum * buf + (1 - self.dampening) * grad
                
                if self.nesterov:
                    grad = grad + self.momentum * buf
                else:
                    grad = buf
                    
                self.state[param_id]['momentum_buffer'] = buf
                
            # Update parameters
            param.data -= self.lr * grad

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()

class Muon: 
    """
    Really quick + simple optimized version of SGD 
    kellerjordan.github.io/posts/muon/
    """
    def __init__(self, params, lr=0.01, momentum=0.9):
            self.params = params
            self.lr = lr
            self.momentum = momentum
            self.velocity = [np.zeros_like(param.data) for param in params]
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            # Update velocity with momentum
            self.velocity[i] = self.momentum * self.velocity[i] - self.lr * param.grad.data
            
            # Update parameters
            param.data += self.velocity[i]
            
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class Adam:
    """
    Implements Adam algorithm.
    
    Adam is a stochastic gradient-based optimization algorithm that uses
    adaptive estimation of first and second-order moments.
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay (L2 penalty)
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}
        self.t = 0
        
    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            # Apply weight decay
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            param_id = id(param)
            if param_id not in self.state:
                self.state[param_id] = {
                    'exp_avg': np.zeros_like(param.data),
                    'exp_avg_sq': np.zeros_like(param.data)
                }
                
            exp_avg, exp_avg_sq = self.state[param_id]['exp_avg'], self.state[param_id]['exp_avg_sq']
            beta1, beta2 = self.betas
            
            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
            
            # Store back into state
            self.state[param_id]['exp_avg'] = exp_avg
            self.state[param_id]['exp_avg_sq'] = exp_avg_sq
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** self.t
            bias_correction2 = 1 - beta2 ** self.t
            
            # Compute bias-corrected moment estimates
            exp_avg_corrected = exp_avg / bias_correction1
            exp_avg_sq_corrected = exp_avg_sq / bias_correction2
            
            # Update parameters
            step_size = self.lr / (np.sqrt(exp_avg_sq_corrected) + self.eps)
            param.data -= step_size * exp_avg_corrected
            
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()


class AdamW:
    """
    Implements AdamW algorithm.
    
    The original Adam algorithm was proposed in "Adam: A Method for Stochastic Optimization".
    AdamW modifies the weight decay in Adam to implement decoupled weight decay as described
    in "Decoupled Weight Decay Regularization".
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay factor (decoupled from gradient-based update)
            amsgrad: whether to use the AMSGrad variant (default: False)
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad
        self.state = {}
        self.t = 0
        
    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
                
            grad = param.grad.data
            param_id = id(param)
            
            if param_id not in self.state:
                self.state[param_id] = {
                    'exp_avg': np.zeros_like(param.data),
                    'exp_avg_sq': np.zeros_like(param.data)
                }
                
                if self.amsgrad:
                    self.state[param_id]['max_exp_avg_sq'] = np.zeros_like(param.data)
                    
            exp_avg, exp_avg_sq = self.state[param_id]['exp_avg'], self.state[param_id]['exp_avg_sq']
            beta1, beta2 = self.betas
            
            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
            
            # Store back into state
            self.state[param_id]['exp_avg'] = exp_avg
            self.state[param_id]['exp_avg_sq'] = exp_avg_sq
            
            if self.amsgrad:
                # Maintains the maximum of all second moment running averages
                max_exp_avg_sq = self.state[param_id]['max_exp_avg_sq']
                max_exp_avg_sq = np.maximum(max_exp_avg_sq, exp_avg_sq)
                self.state[param_id]['max_exp_avg_sq'] = max_exp_avg_sq
                denom = np.sqrt(max_exp_avg_sq) + self.eps
            else:
                denom = np.sqrt(exp_avg_sq) + self.eps
                
            # Bias correction
            bias_correction1 = 1 - beta1 ** self.t
            bias_correction2 = 1 - beta2 ** self.t
            
            # Compute bias-corrected moment estimates
            step_size = self.lr / (denom / np.sqrt(bias_correction2))
            
            # Update parameters
            param.data = (1 - self.weight_decay * self.lr) * param.data - step_size * (exp_avg / bias_correction1)
            
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()


class RMSProp:
    """
    Implements RMSProp algorithm.
    
    Proposed by G. Hinton in his course.
    """
    def __init__(self, params, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            alpha: smoothing constant (default: 0.99)
            eps: term added to the denominator for numerical stability
            weight_decay: weight decay (L2 penalty)
            momentum: momentum factor (default: 0)
            centered: if True, compute the centered RMSProp, gradients normalized by their variance
        """
        self.params = params
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.centered = centered
        self.state = {}
        
    def step(self):
        """Performs a single optimization step."""
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            param_id = id(param)
            if param_id not in self.state:
                self.state[param_id] = {}
                self.state[param_id]['square_avg'] = np.zeros_like(param.data)
                
                if self.momentum > 0:
                    self.state[param_id]['momentum_buffer'] = np.zeros_like(param.data)
                    
                if self.centered:
                    self.state[param_id]['grad_avg'] = np.zeros_like(param.data)
                    
            state = self.state[param_id]
            
            # Update running average of squared gradients
            square_avg = state['square_avg']
            square_avg = self.alpha * square_avg + (1 - self.alpha) * grad * grad
            state['square_avg'] = square_avg
            
            if self.centered:
                # Update running average of gradients
                grad_avg = state['grad_avg']
                grad_avg = self.alpha * grad_avg + (1 - self.alpha) * grad
                state['grad_avg'] = grad_avg
                avg = square_avg - grad_avg * grad_avg
            else:
                avg = square_avg
                
            if self.momentum > 0:
                # Update with momentum
                buf = state['momentum_buffer']
                buf = self.momentum * buf + grad / (np.sqrt(avg) + self.eps)
                state['momentum_buffer'] = buf
                param.data -= self.lr * buf
            else:
                # Simple update
                param.data -= self.lr * grad / (np.sqrt(avg) + self.eps)
                
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()


class LAMB:
    """
    LAMB is a layer-wise adaptive large batch optimization
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, adam_w=True):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate
            betas: coefficients used for computing running averages of gradient and its square
            eps: term added to the denominator to improve numerical stability
            weight_decay: weight decay factor
            adam_w: whether to use AdamW style weight decay (decoupled) or L2 regularization
        """
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.adam_w = adam_w
        self.state = {}
        self.t = 0
        
    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            param_id = id(param)
            
            if param_id not in self.state:
                self.state[param_id] = {
                    'exp_avg': np.zeros_like(param.data),
                    'exp_avg_sq': np.zeros_like(param.data)
                }
                
            exp_avg, exp_avg_sq = self.state[param_id]['exp_avg'], self.state[param_id]['exp_avg_sq']
            beta1, beta2 = self.betas
            
            # Apply weight decay
            if self.adam_w:
                # AdamW style: Decoupled weight decay
                weight_decay = 0
            else:
                # L2 regularization
                weight_decay = self.weight_decay
                grad = grad + weight_decay * param.data
                
            # Update biased first moment estimate
            exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            
            # Update biased second raw moment estimate
            exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad
            
            # Store back into state
            self.state[param_id]['exp_avg'] = exp_avg
            self.state[param_id]['exp_avg_sq'] = exp_avg_sq
            
            # Bias correction
            bias_correction1 = 1 - beta1 ** self.t
            bias_correction2 = 1 - beta2 ** self.t
            
            # Compute bias-corrected moment estimates
            exp_avg_corrected = exp_avg / bias_correction1
            exp_avg_sq_corrected = exp_avg_sq / bias_correction2
            
            # Compute update
            update = exp_avg_corrected / (np.sqrt(exp_avg_sq_corrected) + self.eps)
            
            if self.adam_w and self.weight_decay > 0:
                update = update + self.weight_decay * param.data
                
            # Compute the trust ratio
            w_norm = np.linalg.norm(param.data)
            g_norm = np.linalg.norm(update)
            
            # Use trust ratio clipping
            if w_norm > 0 and g_norm > 0:
                trust_ratio = w_norm / (g_norm + self.eps)
            else:
                trust_ratio = 1.0
                
            # Update parameters
            param.data -= self.lr * trust_ratio * update
            
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()


class AdaGrad:
    """
    It adapts the learning rate to the parameters, performing smaller updates 
    for parameters associated with frequently occurring features, and larger updates 
    for parameters associated with infrequent features.
    """
    def __init__(self, params, lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
        """
        Args:
            params: iterable of parameters to optimize
            lr: learning rate (default: 1e-2)
            lr_decay: learning rate decay (default: 0)
            weight_decay: weight decay (L2 penalty) (default: 0)
            initial_accumulator_value: initial value for accumulator (default: 0)
            eps: term added to the denominator to improve numerical stability (default: 1e-10)
        """
        self.params = params
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.state = {}
        self.t = 0
        
    def step(self):
        """Performs a single optimization step."""
        self.t += 1
        
        # Update learning rate based on decay
        lr = self.lr / (1 + self.lr_decay * self.t)
        
        for param in self.params:
            if param.grad is None:
                continue
                
            grad = param.grad.data
            
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
                
            param_id = id(param)
            if param_id not in self.state:
                self.state[param_id] = {
                    'sum': np.ones_like(param.data) * self.initial_accumulator_value
                }
                
            state = self.state[param_id]
            state['sum'] += grad * grad
            
            # Update parameters
            param.data -= lr * grad / (np.sqrt(state['sum']) + self.eps)
            
    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for param in self.params:
            param.zero_grad()



class NewtonOptimizer:
    """
    Uses the second derivative for more informed steps noting the curvature of the function as well 
    NOT optimal for large scale problems but kind of cool for small problems and niche solutions (quasi-newton for better scale)
    Update rule: x_new = x_old - H^(-1)deltaf(x)
    x_new = x_old - H^(-1)∇f(x) 
    ∇f(x) -> Gradient (first derivative)
    H is the Hessian matrix (second derivative)
    H^(-1) is the inverse of that matrix 
    Hessian matrix: The square matrix of second-order partial derivatives of a scalar valued function. Describes the local curvature of a function of many variables. 

    Args: 
        params: list of parameters to optimize 
        lr: learning rate 
        damping: value added to the diagonal of the hessian matrix for numerical stability 
        max_newton_iter: maximum iterations for solving the Newton system
    """

    # TODO Finish the full impl later 
    def __init__(self, params, lr=1.0, damping=1e-4, max_newton_iter=20):
        self.params = params
        self.lr = lr 
        self.damping = damping
        self.max_newton_iter = max_newton_iter

        self.total_params = sum(p.data.size for p in params)

        # param mapping 
        self.param_mapping = [] 
        offset = 0 
        for p in self.params:
            size = p.data.size
            self.param_mapping.append((offset, offset + size, p.shape))
            offset += size 


    def _pack_params(self):
        """pack into 1D array"""
        param_vec = np.zeros(self.total_params)
        for i, p in enumerate(self.total_params):
            start, end, _ = self.param_mapping[i]
            param_vec[start:end] = p.data.flatten()

        return param_vec

    def _unpack_params(self, param_vec):
        """unpack 1D array into the parameter array"""
        for i, p in enumerate(self.params):
            start, end, shape = self.param_mapping[i]
            p.data = param_vec[start:end].reshape(shape)

    
    def compute_hessian(self):
        """
        Compute the full hessian matrix using second derivatives
        This creates a NxN matrix where n is the total number of parameters
        """

        # Init hessian 
        hessian = np.zeros((self.total_params, self.total_params))
        for param_i, param in enumerate(self.params):
            if param.grad is None:
                continue 

            start_i, end_i, _ = self.param_mapping[param_i]
            param_size = param.data.size

            # Reshape 
            grad = param.grad.data.flatten()

            # Compute all second derivatives w.r.t 

            for idx in range(param_size):
                flat_idx = start_i + idx 

                # Create the tensors for the second backward pass 
                # Compute -> d²L/dθᵢdθⱼ for all params \theta_{j}

                second_grad_target = np.zeros_like(grad)
                second_grad_target[idx] = 1.0 # one hot vector 

                # Reset 
                for p in self.params:
                    if p.grad is not None:
                        p.grad = None 



                param.grad.backward(Tensor(second_grad_target), retain_graph=True)
                
                for param_j, p in enumerate(self.params):
                    if p.grad is not None:
                        start_j, end_j, _ = self.param_mapping[param_j]
                        hessian[flat_idx, start_j:end_j] = p.grad.data.flatten()

        np.fill_diagonal(hessian, hessian.diagonal() + self.damping)

        return hessian

    #TODO the rest later (this step function and then quasi-newton) 
    def step(self):
        """Single newton optimization step"""
        pass
