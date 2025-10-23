import pytest 
import numpy as np 
import torch 
from julia.core.math.jvp import JVP, jvp, jacobian, Dual

class TestJVPPrimitives:
    """Test forward-mode automatic differentiation"""
    
    def test_jvp_scalar_function(self):
        """Test JVP on f(x) = x^2"""
        def f(x):
            if isinstance(x, Dual):
                return x * x
            return x * x
        
        x = 3.0
        v = 1.0
        
        primal_out, tangent_out = jvp(f, [x], [v])
        
        assert primal_out == 9.0
        assert tangent_out == 6.0
        
        x_torch = torch.tensor(3.0, requires_grad=True)
        y_torch = x_torch ** 2
        y_torch.backward()
        assert np.isclose(tangent_out, x_torch.grad.item())
    
    def test_jvp_vector_function(self):
        """Test JVP on vector-valued function"""
        def f(x):
            if isinstance(x, Dual):
                return x * x + x
            return x * x + x
        
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 0.0, 0.0])
        
        primal_out, tangent_out = jvp(f, [x], [v])
        
        expected_primal = x * x + x
        expected_tangent = (2 * x + 1) * v
        
        assert np.allclose(primal_out, expected_primal)
        assert np.allclose(tangent_out, expected_tangent)
    
    def test_jvp_composition(self):
        """Test JVP on composed functions f(g(x))"""
        def g(x):
            if isinstance(x, Dual):
                return x * x
            return x * x
        
        def f(x):
            inner = g(x)
            if isinstance(inner, Dual):
                return inner + inner * inner
            return inner + inner * inner
        
        x = 2.0
        v = 1.0
        
        primal_out, tangent_out = jvp(f, [x], [v])
        
        x_torch = torch.tensor(2.0, requires_grad=True)
        g_torch = x_torch ** 2
        f_torch = g_torch + g_torch ** 2
        f_torch.backward()
        
        assert np.isclose(primal_out, f_torch.item())
        assert np.isclose(tangent_out, x_torch.grad.item())
    
    def test_jvp_nonlinear_function(self):
        """Test JVP on nonlinear function with sin/cos"""
        def f(x):
            if isinstance(x, Dual):
                return x.sin() * x.cos()
            return np.sin(x) * np.cos(x)
        
        x = np.pi / 4
        v = 1.0
        
        primal_out, tangent_out = jvp(f, [x], [v])
        
        x_torch = torch.tensor(np.pi / 4, requires_grad=True)
        y_torch = torch.sin(x_torch) * torch.cos(x_torch)
        y_torch.backward()
        
        assert np.isclose(primal_out, y_torch.item(), rtol=1e-5)
        assert np.isclose(tangent_out, x_torch.grad.item(), rtol=1e-5)
    
    def test_jvp_multivariate(self):
        def f(x, y):
            if isinstance(x, Dual) and isinstance(y, Dual):
                return x * y + x * x
            return x * y + x * x
        
        x, y = 3.0, 2.0
        vx, vy = 1.0, 0.0
        
        primal_out, tangent_out = jvp(f, [x, y], [vx, vy])
        
        x_torch = torch.tensor(3.0, requires_grad=True)
        y_torch = torch.tensor(2.0, requires_grad=True)
        z_torch = x_torch * y_torch + x_torch ** 2
        z_torch.backward()
        
        expected_tangent = x_torch.grad.item() * vx + y_torch.grad.item() * vy
        
        assert np.isclose(primal_out, z_torch.item())
        assert np.isclose(tangent_out, expected_tangent)
