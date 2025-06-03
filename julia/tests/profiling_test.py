import numpy as np
import time
import sys
import os
from typing import List, Dict, Any
from julia.core.tensor import Tensor, Function
from julia.core.nn.layers import Linear, Sequential, Dropout
from julia.core.nn.conv import Conv2D
from julia.core.nn.pooling import MaxPool2D
from julia.core.optim import Adam, SGD
from julia.core.loss import mse_loss, cross_entropy_loss
from julia.core.profiler import ModelProfiler, profiler, profile_scope, enable_profiling, print_profiling_summary

class ProfiledFunction(Function):
    """
    Enhanced Function class that automatically profiles operations
    """
    
    @classmethod
    def apply(cls, *args, **kwargs):
        op_name = cls.__name__
        inputs = [arg for arg in args if isinstance(arg, Tensor)]
        
        with profiler.profile_operation(
            op_name=f"{cls.__module__}.{cls.__name__}",
            op_type=cls.__name__,
            inputs=inputs
        ) as stats:
            result = super().apply(*args, **kwargs)
            
            # Add output information to stats
            if isinstance(result, Tensor):
                stats.output_shapes = [result.shape]
                stats.output_dtypes = [str(result.data.dtype)]
                # Estimate FLOPS
                stats.flops = profiler._estimate_flops(
                    cls.__name__, 
                    stats.input_shapes, 
                    stats.output_shapes
                )
            elif isinstance(result, (list, tuple)):
                stats.output_shapes = [getattr(r, 'shape', ()) for r in result if hasattr(r, 'shape')]
                stats.output_dtypes = [str(getattr(r, 'dtype', 'unknown')) for r in result if hasattr(r, 'data')]
        
        return result

class ProfiledLinear(Linear):
    """Linear layer with integrated profiling"""
    
    def forward(self, x: Tensor) -> Tensor:
        with profiler.profile_operation(
            f"Linear_{self.in_features}x{self.out_features}",
            "Linear",
            [x]
        ):
            return super().forward(x)

class ProfiledConv2D(Conv2D):
    """Conv2D layer with integrated profiling"""
    
    def forward(self, x: Tensor) -> Tensor:
        with profiler.profile_operation(
            f"Conv2D_{self.in_channels}x{self.out_channels}_k{self.kernel_size}",
            "Conv2D",
            [x]
        ):
            return super().forward(x)

def create_simple_model(input_size: int, hidden_size: int, output_size: int):
    """Create a simple neural network with profiling"""
    return Sequential(
        ProfiledLinear(input_size, hidden_size),
        lambda x: x.relu(),
        Dropout(0.2),
        ProfiledLinear(hidden_size, hidden_size // 2),
        lambda x: x.relu(),
        ProfiledLinear(hidden_size // 2, output_size)
    )

def create_cnn_model():
    """Create a CNN model with profiling"""
    return Sequential(
        ProfiledConv2D(1, 32, 3, padding=1),
        lambda x: x.relu(),
        MaxPool2D(2, 2),
        ProfiledConv2D(32, 64, 3, padding=1), 
        lambda x: x.relu(),
        MaxPool2D(2, 2),
        lambda x: x.reshape(-1, 64 * 7 * 7),  # Assuming 28x28 input
        ProfiledLinear(64 * 7 * 7, 128),
        lambda x: x.relu(),
        Dropout(0.5),
        ProfiledLinear(128, 10)
    )

def test_basic_profiling():
    """Test basic profiling functionality"""
    print("Testing Basic Profiling")
    
    profiler.clear()
    enable_profiling()
    
    # Test tensor operations
    with profile_scope("tensor_creation"):
        x = Tensor(np.random.randn(100, 50), requires_grad=True)
        y = Tensor(np.random.randn(50, 30), requires_grad=True)
    
    with profile_scope("matrix_operations"):
        # Matrix multiplication
        z = x.matmul(y)
        
        # Element-wise operations
        w = z + z
        w = w * 2.0
        w = w.relu()
    
    with profile_scope("backward_pass"):
        loss = w.sum()
        loss.backward()
    
    print("Basic profiling completed.")
    assert(True)

def test_model_training():
    """Test profiling during model training"""
    print("Testing Model Training Profiling")
    
    profiler.clear()
    
    # Create synthetic dataset
    batch_size, input_size, output_size = 32, 784, 10
    num_batches = 5
    
    X_data = np.random.randn(num_batches * batch_size, input_size)
    y_data = np.random.randint(0, output_size, (num_batches * batch_size,))
    
    # Create model
    model = create_simple_model(input_size, 256, output_size)
    optimizer = Adam(model.parameters(), lr=0.001)
    
    print(f"Training model for {num_batches} batches")

    
    for batch in range(num_batches):
        with profile_scope(f"batch_{batch}", "training_batch"):
            # Get batch data
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = Tensor(X_data[start_idx:end_idx])
            y_batch = Tensor(y_data[start_idx:end_idx])
            
            # Forward pass
            with profile_scope("forward_pass"):
                predictions = model(X_batch)
            
            # Compute loss
            with profile_scope("loss_computation"):
                loss = cross_entropy_loss(predictions, y_batch)
            
            # Backward pass
            with profile_scope("backward_pass"):
                optimizer.zero_grad()
                loss.backward()
            
            # Optimizer step
            with profile_scope("optimizer_step"):
                optimizer.step()
            
            if batch % 2 == 0:
                print(f"  Batch {batch}, Loss: {loss.data:.4f}")
    
    print("Model training profiling completed.")
    assert(True)

def test_cnn_profiling():
    """Test CNN profiling with image-like data"""
    print("Testing CNN Profiling")
    
    profiler.clear()
    
    # Create CNN model
    model = create_cnn_model()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Create fake MNIST-like data
    batch_size = 16
    X = Tensor(np.random.randn(batch_size, 1, 28, 28))
    y = Tensor(np.random.randint(0, 10, batch_size))
    
    print("Running CNN forward and backward pass")
    
    with profile_scope("cnn_training_step"):
        # Forward pass
        with profile_scope("cnn_forward"):
            predictions = model(X)
        
        # Loss computation
        with profile_scope("cnn_loss"):
            loss = cross_entropy_loss(predictions, y)
        
        # Backward pass
        with profile_scope("cnn_backward"):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    print(f"CNN Loss: {loss.data:.4f}")
    print("CNN profiling completed.")
    print_profiling_summary(2)
    
    assert(True)
