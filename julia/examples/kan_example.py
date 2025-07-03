"""
Kolmogorov-Arnold Network (KAN) Example using Julia Framework

Implementation of KAN based on "KAN: Kolmogorov-Arnold Networks" paper.
KANs replace linear weights with learnable univariate functions on edges.
"""

import numpy as np
from julia.core.tensor import Tensor, Function
from julia.core.nn.layers import Layer, Linear
from julia.core.loss import mse_loss
from julia.core.optim import Adam

class BSplineFunction(Function):
    """B-spline basis function for learnable univariate functions"""
    
    @staticmethod
    def forward(ctx, x, control_points, knots, degree=3):
        """
        Evaluate B-spline with given control points
        
        Args:
            x: Input tensor
            control_points: Control points for the B-spline
            knots: Knot vector
            degree: Degree of B-spline
        """
        ctx.save_for_backwards(x, control_points, knots)
        ctx.save_data(degree=degree)
        
        # Simplified B-spline evaluation
        x_data = x.data
        cp_data = control_points.data
        knots_data = knots.data
        
        output = np.zeros_like(x_data)
        
        # Basic B-spline evaluation (simplified)
        for i in range(len(cp_data)):
            # Evaluate basis function
            basis = BSplineFunction._basis_function(x_data, i, degree, knots_data)
            output += cp_data[i] * basis
        
        return Tensor(output, requires_grad=x.requires_grad or control_points.requires_grad)
    
    @staticmethod
    def _basis_function(x, i, degree, knots):
        """Evaluate B-spline basis function using Cox-de Boor recursion"""
        if degree == 0:
            return ((knots[i] <= x) & (x < knots[i + 1])).astype(float)
        
        # Avoid division by zero
        denom1 = knots[i + degree] - knots[i]
        denom2 = knots[i + degree + 1] - knots[i + 1]
        
        term1 = 0.0
        term2 = 0.0
        
        if denom1 != 0:
            term1 = (x - knots[i]) / denom1 * BSplineFunction._basis_function(x, i, degree - 1, knots)
        
        if denom2 != 0:
            term2 = (knots[i + degree + 1] - x) / denom2 * BSplineFunction._basis_function(x, i + 1, degree - 1, knots)
        
        return term1 + term2
    
    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass for B-spline function"""
        x, control_points, knots = ctx.saved_tensors
        degree = ctx.saved_data['degree']
        
        # Simplified gradient computation
        x_data = x.data
        knots_data = knots.data
        
        # Gradient w.r.t. control points
        grad_cp = np.zeros_like(control_points.data)
        for i in range(len(control_points.data)):
            basis = BSplineFunction._basis_function(x_data, i, degree, knots_data)
            grad_cp[i] = np.sum(grad_output.data * basis)
        
        # Gradient w.r.t. input (simplified)
        grad_x = np.zeros_like(x.data)
        
        return Tensor(grad_x), Tensor(grad_cp), None, None

class KANLayer(Layer):
    """
    Single KAN layer with learnable univariate functions on edges
    """
    
    def __init__(self, input_dim, output_dim, grid_size=5, spline_order=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Initialize control points for B-splines
        # Each edge (i,j) has its own univariate function
        self.control_points = {}
        self.knots = {}
        
        for i in range(input_dim):
            for j in range(output_dim):
                # Initialize control points
                cp_key = f"{i}_{j}"
                n_control_points = grid_size + spline_order + 1
                
                # Random initialization
                self.control_points[cp_key] = Tensor(
                    np.random.randn(n_control_points) * 0.1, 
                    requires_grad=True
                )
                
                # Uniform knot vector
                knots = np.linspace(-1, 1, n_control_points + spline_order + 1)
                self.knots[cp_key] = Tensor(knots, requires_grad=False)
        
        # Base linear transformation (residual connection)
        self.base_linear = Linear(input_dim, output_dim)
        
        # Scale parameter
        self.scale = Tensor(np.ones(1), requires_grad=True)
    
    def forward(self, x):
        """
        Forward pass through KAN layer
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.shape[0]
        output = Tensor(np.zeros((batch_size, self.output_dim)))
        
        # Apply learnable univariate functions on each edge
        for i in range(self.input_dim):
            for j in range(self.output_dim):
                cp_key = f"{i}_{j}"
                
                # Extract input for this dimension
                x_i = x[:, i:i+1]  # (batch_size, 1)
                
                # Apply B-spline function
                spline_output = BSplineFunction.apply(
                    x_i.reshape(-1),
                    self.control_points[cp_key],
                    self.knots[cp_key],
                    self.spline_order
                )
                
                # Add to output
                output_slice = output[:, j:j+1]
                spline_reshaped = spline_output.reshape(batch_size, 1)
                output = output + Tensor(np.zeros_like(output.data))
                output.data[:, j] += spline_reshaped.data.flatten()
        
        # Add base linear transformation (residual)
        base_output = self.base_linear(x)
        output = output + self.scale * base_output
        
        return output

class KolmogorovArnoldNetwork(Layer):
    """
    Multi-layer Kolmogorov-Arnold Network
    """
    
    def __init__(self, layer_dims, grid_size=5, spline_order=3):
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        
        # Create KAN layers
        self.layers = []
        for i in range(self.num_layers):
            layer = KANLayer(
                input_dim=layer_dims[i],
                output_dim=layer_dims[i+1],
                grid_size=grid_size,
                spline_order=spline_order
            )
            self.layers.append(layer)
    
    def forward(self, x):
        """Forward pass through the KAN"""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """Get all trainable parameters"""
        params = []
        for layer in self.layers:
            # Add control points
            for cp in layer.control_points.values():
                params.append(cp)
            # Add scale parameter
            params.append(layer.scale)
            # Add base linear parameters
            params.extend(layer.base_linear.parameters())
        return params

class SimpleKAN(Layer):
    """Simplified KAN implementation using polynomial basis functions"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.degree = degree
        
        # Polynomial coefficients for each edge
        # Edge from input i to hidden j has polynomial coefficients
        self.hidden_coeffs = Tensor(
            np.random.randn(input_dim, hidden_dim, degree + 1) * 0.1,
            requires_grad=True
        )
        
        # Edge from hidden i to output j
        self.output_coeffs = Tensor(
            np.random.randn(hidden_dim, output_dim, degree + 1) * 0.1,
            requires_grad=True
        )
        
        # Bias terms
        self.hidden_bias = Tensor(np.zeros(hidden_dim), requires_grad=True)
        self.output_bias = Tensor(np.zeros(output_dim), requires_grad=True)
    
    def _polynomial_activation(self, x, coeffs):
        """Apply polynomial activation function - fixed to avoid recursion"""
        # x: (batch_size, input_dim)
        # coeffs: (input_dim, output_dim, degree+1)
        
        batch_size = x.shape[0]
        input_dim, output_dim, num_coeffs = coeffs.shape
        
        result = Tensor(np.zeros((batch_size, output_dim)))
        
        for i in range(input_dim):
            for j in range(output_dim):
                x_i = Tensor(x.data[:, i])  # Extract column i
                # Compute polynomial: sum(c_k * x^k)
                poly_output = Tensor(np.zeros(batch_size))
                for k in range(num_coeffs):
                    coeff = coeffs.data[i, j, k]
                    if k == 0:
                        x_power_data = np.ones(batch_size)
                    elif k == 1:
                        x_power_data = x_i.data
                    else:
                        x_power_data = np.power(x_i.data, k)
                    
                    x_power = Tensor(x_power_data)
                    term = Tensor(np.array(coeff)) * x_power
                    poly_output = poly_output + term
                
                # Add to result
                result.data[:, j] += poly_output.data
        
        return result
    
    def forward(self, x):
        """Forward pass through simplified KAN"""
        # Hidden layer
        hidden = self._polynomial_activation(x, self.hidden_coeffs)
        hidden = hidden + self.hidden_bias
        
        # Apply activation (tanh)
        hidden = hidden.tanh()
        
        # Output layer
        output = self._polynomial_activation(hidden, self.output_coeffs)
        output = output + self.output_bias
        
        return output
    
    def parameters(self):
        """Get all parameters"""
        return [self.hidden_coeffs, self.output_coeffs, self.hidden_bias, self.output_bias]

def generate_symbolic_data(n_samples=1000, noise_level=0.1):
    """Generate data for symbolic regression task"""
    # Function: f(x,y) = x^2 + sin(y) + x*y
    x = np.random.uniform(-2, 2, (n_samples, 1))
    y = np.random.uniform(-2, 2, (n_samples, 1))
    
    # Combine inputs
    inputs = np.concatenate([x, y], axis=1)
    
    # Target function
    targets = x**2 + np.sin(y) + x*y
    targets = targets + np.random.normal(0, noise_level, targets.shape)
    
    return Tensor(inputs), Tensor(targets.flatten())

def generate_classification_data(n_samples=1000):
    """Generate 2D classification data"""
    # Create spiral dataset
    n_per_class = n_samples // 2
    
    # Class 0: inner spiral
    t1 = np.linspace(0, 4*np.pi, n_per_class)
    r1 = 0.5 * t1 / (4*np.pi)
    x1 = r1 * np.cos(t1) + np.random.normal(0, 0.1, n_per_class)
    y1 = r1 * np.sin(t1) + np.random.normal(0, 0.1, n_per_class)
    
    # Class 1: outer spiral
    t2 = np.linspace(0, 4*np.pi, n_per_class)
    r2 = 1.0 * t2 / (4*np.pi)
    x2 = r2 * np.cos(t2 + np.pi) + np.random.normal(0, 0.1, n_per_class)
    y2 = r2 * np.sin(t2 + np.pi) + np.random.normal(0, 0.1, n_per_class)
    
    # Combine
    inputs = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    targets = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return Tensor(inputs), Tensor(targets)

def train_kan_regression():
    print("Training KAN for Symbolic Regression")
    train_inputs, train_targets = generate_symbolic_data(1000)
    test_inputs, test_targets = generate_symbolic_data(200)
    
    # Create simplified KAN model
    model = SimpleKAN(
        input_dim=2,
        hidden_dim=10,
        output_dim=1,
        degree=3
    )
    
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Training
    num_epochs = 100
    batch_size = 32
    
    print(f"Training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batches = len(train_inputs.data) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            # Mini-batch
            batch_inputs = Tensor(train_inputs.data[start_idx:end_idx])
            batch_targets = Tensor(train_targets.data[start_idx:end_idx])
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            
            # Loss
            loss = mse_loss(predictions.flatten(), batch_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
        
        avg_loss = epoch_loss / n_batches
        
        if (epoch + 1) % 20 == 0:
            # Test performance
            test_predictions = model(test_inputs)
            test_loss = mse_loss(test_predictions.flatten(), test_targets)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.6f}, Test Loss: {test_loss.data:.6f}")
    
    print("Symbolic regression training complete")
    return model

def train_kan_classification():
    print("\nTraining KAN for Classification")
    
    # Generate spiral data
    train_inputs, train_targets = generate_classification_data(800)
    test_inputs, test_targets = generate_classification_data(200)
    
    # Create model
    model = SimpleKAN(
        input_dim=2,
        hidden_dim=8,
        output_dim=2,  # Binary classification
        degree=2
    )
    
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Training
    num_epochs = 150
    batch_size = 32
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        n_batches = len(train_inputs.data) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_inputs = Tensor(train_inputs.data[start_idx:end_idx])
            batch_targets = Tensor(train_targets.data[start_idx:end_idx])
            
            optimizer.zero_grad()
            logits = model(batch_inputs)
            
            from julia.core.loss import cross_entropy_loss 
            loss = cross_entropy_loss(logits, batch_targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
            
            # Accuracy
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == batch_targets.data)
            total += batch_size
        
        if (epoch + 1) % 30 == 0:
            train_acc = correct / total
            
            # Test accuracy
            test_logits = model(test_inputs)
            test_predictions = np.argmax(test_logits.data, axis=1)
            test_acc = np.mean(test_predictions == test_targets.data)
            
            print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.3f}, Test Acc: {test_acc:.3f}")
    
    print("Classification training complete")
    return model

def visualize_kan_function():
    """Visualize what the KAN has learned"""
    print("\nVisualizing KAN learned functions")
    
    # Create a simple 1D -> 1D KAN
    model = SimpleKAN(input_dim=1, hidden_dim=5, output_dim=1, degree=4)
    
    # Train on a simple function: f(x) = x^3 - 2*x^2 + x
    x_train = np.linspace(-2, 2, 100).reshape(-1, 1)
    y_train = x_train**3 - 2*x_train**2 + x_train
    y_train += np.random.normal(0, 0.1, y_train.shape)
    
    train_inputs = Tensor(x_train)
    train_targets = Tensor(y_train.flatten())
    
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # Quick training
    for epoch in range(200):
        optimizer.zero_grad()
        predictions = model(train_inputs)
        loss = mse_loss(predictions.flatten(), train_targets)
        loss.backward()
        optimizer.step()
    
    # Test on new points
    x_test = np.linspace(-2.5, 2.5, 50).reshape(-1, 1)
    test_inputs = Tensor(x_test)
    predictions = model(test_inputs)
    
    print("Sample predictions (input -> predicted output):")
    for i in range(0, len(x_test), 10):
        x_val = x_test[i, 0]
        # Fix: Extract scalar value from prediction tensor
        if predictions.shape[1] == 1:
            pred_val = predictions.data[i, 0]  # Extract from 2D output
        else:
            pred_val = predictions.data[i]     # Extract from 1D output
        
        # Handle case where pred_val might still be an array
        if hasattr(pred_val, 'item'):
            pred_val = pred_val.item()
        elif isinstance(pred_val, np.ndarray):
            pred_val = float(pred_val.flatten()[0])
        
        true_val = x_val**3 - 2*x_val**2 + x_val
        print(f"  x={x_val:.2f} -> pred={pred_val:.3f}, true={true_val:.3f}")

if __name__ == "__main__":
    regression_model = train_kan_regression()
    
    classification_model = train_kan_classification()
    
    visualize_kan_function()
