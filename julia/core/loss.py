import numpy as np
from typing import Optional, Callable, Union, Tuple, List
from julia.core.tensor import Tensor, Function, Context, _ensure_tensor


class LossFunction:
    """
    Base class for all loss functions. Provides a consistent interface and 
    utilities for creating custom loss functions.
    
    Custom loss functions should inherit from this class and implement:
    - forward(y_pred, y_true) -> Tensor
    - backward(grad_output) -> Tuple[Tensor, Tensor] (optional for custom backward)
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Args:
            reduction: Specifies the reduction to apply to the output
                      'none' | 'mean' | 'sum'
        """
        assert reduction in ['none', 'mean', 'sum'], f"Invalid reduction: {reduction}"
        self.reduction = reduction
    
    def __call__(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Apply the loss function"""
        return self.forward(y_pred, y_true)
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError
    
    def _apply_reduction(self, loss: Tensor) -> Tensor:
        """Apply the specified reduction to the loss tensor"""
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return Tensor(np.mean(loss.data))
        elif self.reduction == 'sum':
            return Tensor(np.sum(loss.data))

class CustomLossTemplate(LossFunction):
    """
    Template for creating custom loss functions
    
    To create a custom loss function:
    1. Inherit from this class
    2. Implement forward() method with your loss computation
    3. Optionally implement backward() for custom gradients
    4. Use mathematical operations that are differentiable
    
    Example:
    class MyCustomLoss(CustomLossTemplate):
        def forward(self, y_pred, y_true):
            # Your custom loss computation here
            diff = y_pred - y_true
            loss = some_custom_function(diff)
            return self._apply_reduction(loss)
    """
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """
        Implement your custom loss function here
        
        Args:
            y_pred: Model predictions
            y_true: Ground truth labels
            
        Returns:
            Loss tensor
        """
        raise NotImplementedError("Implement your custom loss function here")



class MSELoss(Function):
    """
    Mean Squared Error Loss
    
    Mathematical Derivation:
    L(y, ŷ) = (1/n) * Σ(y_i - ŷ_i)²
    
    Gradient:
    ∂L/∂ŷ = (2/n) * (ŷ - y)
    
    Use Cases:
    - Regression tasks
    - When you want to penalize large errors heavily
    - Assumes Gaussian noise in targets
    - Sensitive to outliers
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        ctx.save_for_backwards(y_pred, y_true)
        ctx.save_data(reduction=reduction)
        
        diff = y_pred.data - y_true.data
        loss = diff ** 2
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        reduction = ctx.saved_data['reduction']
        
        diff = y_pred.data - y_true.data
        
        if reduction == 'mean':
            grad_pred = 2.0 * diff / np.prod(diff.shape)
        elif reduction == 'sum':
            grad_pred = 2.0 * diff
        else:  # 'none'
            grad_pred = 2.0 * diff
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None


class MAELoss(Function):
    """
    Mean Absolute Error Loss (L1 Loss)
    
    Mathematical Derivation:
    L(y, ŷ) = (1/n) * Σ|y_i - ŷ_i|
    
    Gradient:
    ∂L/∂ŷ = (1/n) * sign(ŷ - y)
    
    Use Cases:
    - Robust to outliers compared to MSE
    - When you want equal penalty for all errors
    - Median regression
    - Less sensitive to extreme values
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        ctx.save_for_backwards(y_pred, y_true)
        ctx.save_data(reduction=reduction)
        
        diff = y_pred.data - y_true.data
        loss = np.abs(diff)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        reduction = ctx.saved_data['reduction']
        
        diff = y_pred.data - y_true.data
        grad_pred = np.sign(diff)
        
        if reduction == 'mean':
            grad_pred = grad_pred / np.prod(diff.shape)
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None


class HuberLoss(Function):
    """
    Huber Loss (Smooth L1 Loss)
    
    Mathematical Derivation:
    L_δ(y, ŷ) = {
        0.5 * (y - ŷ)²                    if |y - ŷ| ≤ δ
        δ * (|y - ŷ| - 0.5 * δ)          otherwise
    }
    
    Gradient:
    ∂L/∂ŷ = {
        (ŷ - y)                          if |y - ŷ| ≤ δ
        δ * sign(ŷ - y)                  otherwise
    }
    
    Use Cases:
    - Combines benefits of MSE and MAE
    - Less sensitive to outliers than MSE
    - Smooth gradient near zero unlike MAE
    - Good for regression with occasional outliers
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, delta=1.0, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        ctx.save_for_backwards(y_pred, y_true)
        ctx.save_data(delta=delta, reduction=reduction)
        
        diff = y_pred.data - y_true.data
        abs_diff = np.abs(diff)
        
        # Quadratic for small errors, linear for large errors
        quadratic = 0.5 * diff ** 2
        linear = delta * (abs_diff - 0.5 * delta)
        
        loss = np.where(abs_diff <= delta, quadratic, linear)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        delta = ctx.saved_data['delta']
        reduction = ctx.saved_data['reduction']
        
        diff = y_pred.data - y_true.data
        abs_diff = np.abs(diff)
        
        # Gradient is linear for small errors, constant for large errors
        grad_pred = np.where(abs_diff <= delta, diff, delta * np.sign(diff))
        
        if reduction == 'mean':
            grad_pred = grad_pred / np.prod(diff.shape)
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None, None


class LogCoshLoss(Function):
    """
    Log-Cosh Loss
    
    Mathematical Derivation:
    L(y, ŷ) = log(cosh(ŷ - y))
    
    Gradient:
    ∂L/∂ŷ = tanh(ŷ - y)
    
    Use Cases:
    - Smooth everywhere (twice differentiable)
    - Approximately quadratic for small errors
    - Approximately linear for large errors
    - More robust than MSE, smoother than Huber
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        ctx.save_for_backwards(y_pred, y_true)
        ctx.save_data(reduction=reduction)
        
        diff = y_pred.data - y_true.data
        loss = np.log(np.cosh(diff))
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        y_pred, y_true = ctx.saved_tensors
        reduction = ctx.saved_data['reduction']
        
        diff = y_pred.data - y_true.data
        grad_pred = np.tanh(diff)
        
        if reduction == 'mean':
            grad_pred = grad_pred / np.prod(diff.shape)
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None


class CrossEntropyLoss(Function):
    """
    Cross Entropy Loss for Classification
    
    Mathematical Derivation:
    L(y, ŷ) = -Σ y_i * log(softmax(ŷ)_i)
    
    For sparse labels:
    L(y, ŷ) = -log(softmax(ŷ)_{y})
    
    Gradient:
    ∂L/∂ŷ = softmax(ŷ) - y
    
    Use Cases:
    - Multi-class classification
    - When classes are mutually exclusive
    - Natural choice for probabilistic interpretation
    - Works well with softmax activation
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        # Apply softmax to predictions for numerical stability
        max_vals = np.max(y_pred.data, axis=-1, keepdims=True)
        exp_vals = np.exp(y_pred.data - max_vals)
        softmax_vals = exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)
        
        ctx.save_data(softmax_vals=softmax_vals, reduction=reduction)
        
        # Handle both one-hot and sparse labels
        if y_true.data.ndim == y_pred.data.ndim:
            # One-hot encoded
            loss = -np.sum(y_true.data * np.log(softmax_vals + 1e-8), axis=-1)
        else:
            # Sparse labels
            batch_size = y_pred.data.shape[0]
            loss = -np.log(softmax_vals[np.arange(batch_size), y_true.data.astype(int)] + 1e-8)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax_vals = ctx.saved_data['softmax_vals']
        reduction = ctx.saved_data['reduction']
        
        # For cross-entropy with softmax, gradient is simply: softmax - true_labels
        # This is handled in the calling code since we need y_true
        grad_pred = softmax_vals.copy()  # Will be modified by caller
        
        if reduction == 'mean':
            grad_pred = grad_pred / softmax_vals.shape[0]
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None


class BinaryCrossEntropyLoss(Function):
    """
    Binary Cross Entropy Loss
    
    Mathematical Derivation:
    L(y, ŷ) = -[y * log(σ(ŷ)) + (1-y) * log(1-σ(ŷ))]
    where σ is the sigmoid function
    
    Gradient:
    ∂L/∂ŷ = σ(ŷ) - y
    
    Use Cases:
    - Binary classification
    - Multi-label classification (independent labels)
    - When you need probability outputs
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        # Apply sigmoid for numerical stability
        sigmoid_vals = 1 / (1 + np.exp(-y_pred.data))
        sigmoid_vals = np.clip(sigmoid_vals, 1e-8, 1 - 1e-8)
        
        ctx.save_data(sigmoid_vals=sigmoid_vals, reduction=reduction)
        
        loss = -(y_true.data * np.log(sigmoid_vals) + 
                (1 - y_true.data) * np.log(1 - sigmoid_vals))
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_vals = ctx.saved_data['sigmoid_vals']
        reduction = ctx.saved_data['reduction']
        
        # Gradient is sigmoid - y_true (will be modified by caller)
        grad_pred = sigmoid_vals.copy()
        
        if reduction == 'mean':
            grad_pred = grad_pred / np.prod(sigmoid_vals.shape)
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None


class FocalLoss(Function):
    """
    Focal Loss for addressing class imbalance
    
    Mathematical Derivation:
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    where p_t = p if y=1, else 1-p
    
    Gradient:
    ∂FL/∂p = α * γ * (1-p_t)^(γ-1) * p_t * log(p_t) + α * (1-p_t)^γ - α_t * (1-p_t)^γ
    
    Use Cases:
    - Heavily imbalanced datasets
    - Object detection
    - When easy examples dominate training
    - Focus learning on hard examples
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, alpha=1.0, gamma=2.0, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        # Apply sigmoid to get probabilities
        sigmoid_vals = 1 / (1 + np.exp(-y_pred.data))
        sigmoid_vals = np.clip(sigmoid_vals, 1e-8, 1 - 1e-8)
        
        # Compute p_t
        p_t = np.where(y_true.data == 1, sigmoid_vals, 1 - sigmoid_vals)
        
        # Compute focal weight
        focal_weight = alpha * ((1 - p_t) ** gamma)
        
        # Compute focal loss
        loss = -focal_weight * np.log(p_t)
        
        ctx.save_data(sigmoid_vals=sigmoid_vals, p_t=p_t, focal_weight=focal_weight,
                     alpha=alpha, gamma=gamma, reduction=reduction)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_vals = ctx.saved_data['sigmoid_vals']
        p_t = ctx.saved_data['p_t']
        focal_weight = ctx.saved_data['focal_weight']
        alpha = ctx.saved_data['alpha']
        gamma = ctx.saved_data['gamma']
        reduction = ctx.saved_data['reduction']
        
        # Complex gradient for focal loss
        term1 = alpha * gamma * ((1 - p_t) ** (gamma - 1)) * p_t * np.log(p_t + 1e-8)
        term2 = alpha * ((1 - p_t) ** gamma)
        
        grad_pred = term1 + term2 - focal_weight
        
        if reduction == 'mean':
            grad_pred = grad_pred / np.prod(sigmoid_vals.shape)
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None, None, None

class ContrastiveLoss(Function):
    """
    Contrastive Loss for Siamese Networks
    
    Mathematical Derivation:
    L = (1-Y) * (1/2) * D² + Y * (1/2) * max(0, margin - D)²
    where D = ||f(x1) - f(x2)||₂, Y=1 for different pairs, Y=0 for similar pairs
    
    Use Cases:
    - Face verification
    - Siamese networks
    - Learning embeddings where distance matters
    - Few-shot learning
    """
    
    @staticmethod
    def forward(ctx, embedding1, embedding2, labels, margin=1.0, reduction='mean'):
        embedding1 = _ensure_tensor(embedding1)
        embedding2 = _ensure_tensor(embedding2)
        labels = _ensure_tensor(labels)
        
        # Compute Euclidean distance
        diff = embedding1.data - embedding2.data
        distance = np.sqrt(np.sum(diff ** 2, axis=1) + 1e-8)
        
        # Contrastive loss
        positive_loss = (1 - labels.data) * 0.5 * distance ** 2
        negative_loss = labels.data * 0.5 * np.maximum(0, margin - distance) ** 2
        loss = positive_loss + negative_loss
        
        ctx.save_for_backwards(embedding1, embedding2, labels)
        ctx.save_data(distance=distance, margin=margin, reduction=reduction)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=embedding1.requires_grad or embedding2.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        embedding1, embedding2, labels = ctx.saved_tensors
        distance = ctx.saved_data['distance']
        margin = ctx.saved_data['margin']
        reduction = ctx.saved_data['reduction']
        
        diff = embedding1.data - embedding2.data
        
        # Gradient computation
        positive_grad = (1 - labels.data) * diff
        negative_mask = (labels.data * (margin > distance)).astype(float)
        negative_grad = -negative_mask[:, np.newaxis] * diff
        
        grad_embedding1 = positive_grad + negative_grad
        grad_embedding2 = -(positive_grad + negative_grad)
        
        if reduction == 'mean':
            grad_embedding1 = grad_embedding1 / len(distance)
            grad_embedding2 = grad_embedding2 / len(distance)
        
        return (Tensor(grad_output.data * grad_embedding1), 
                Tensor(grad_output.data * grad_embedding2), 
                None, None, None)


class TripletLoss(Function):
    """
    Triplet Loss for learning embeddings
    
    Mathematical Derivation:
    L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
    where a=anchor, p=positive, n=negative
    
    Use Cases:
    - Face recognition
    - Learning embeddings
    - Metric learning
    - When you have triplets of data (anchor, positive, negative)
    """
    
    @staticmethod
    def forward(ctx, anchor, positive, negative, margin=1.0, reduction='mean'):
        anchor = _ensure_tensor(anchor)
        positive = _ensure_tensor(positive)
        negative = _ensure_tensor(negative)
        
        # Compute distances
        pos_dist = np.sum((anchor.data - positive.data) ** 2, axis=1)
        neg_dist = np.sum((anchor.data - negative.data) ** 2, axis=1)
        
        # Triplet loss
        loss = np.maximum(0, pos_dist - neg_dist + margin)
        
        ctx.save_for_backwards(anchor, positive, negative)
        ctx.save_data(pos_dist=pos_dist, neg_dist=neg_dist, margin=margin, reduction=reduction)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=(anchor.requires_grad or 
                                         positive.requires_grad or 
                                         negative.requires_grad))
    
    @staticmethod
    def backward(ctx, grad_output):
        anchor, positive, negative = ctx.saved_tensors
        pos_dist = ctx.saved_data['pos_dist']
        neg_dist = ctx.saved_data['neg_dist']
        margin = ctx.saved_data['margin']
        reduction = ctx.saved_data['reduction']
        
        # Mask for active triplets
        mask = (pos_dist - neg_dist + margin > 0).astype(float)
        
        # Gradients
        grad_anchor = 2 * mask[:, np.newaxis] * ((positive.data - anchor.data) - 
                                                (negative.data - anchor.data))
        grad_positive = 2 * mask[:, np.newaxis] * (anchor.data - positive.data)
        grad_negative = 2 * mask[:, np.newaxis] * (negative.data - anchor.data)
        
        if reduction == 'mean':
            grad_anchor = grad_anchor / len(mask)
            grad_positive = grad_positive / len(mask)
            grad_negative = grad_negative / len(mask)
        
        return (Tensor(grad_output.data * grad_anchor),
                Tensor(grad_output.data * grad_positive),
                Tensor(grad_output.data * grad_negative),
                None, None)


class WassersteinLoss(Function):
    """
    Wasserstein Loss (Earth Mover's Distance) for GANs
    
    Mathematical Derivation:
    W(P_r, P_g) = inf_{γ∈Π(P_r,P_g)} E_{(x,y)~γ}[||x-y||]
    
    Approximated as: L = E[D(x)] - E[D(G(z))]
    
    Use Cases:
    - Wasserstein GANs (WGANs)
    - More stable GAN training
    - Better gradient flow
    - Meaningful loss values
    """
    
    @staticmethod
    def forward(ctx, real_scores, fake_scores, reduction='mean'):
        real_scores = _ensure_tensor(real_scores)
        fake_scores = _ensure_tensor(fake_scores)
        
        # Wasserstein distance approximation
        loss = np.mean(fake_scores.data) - np.mean(real_scores.data)
        
        ctx.save_data(reduction=reduction, 
                     real_shape=real_scores.shape, 
                     fake_shape=fake_scores.shape)
        
        return Tensor(loss, requires_grad=(real_scores.requires_grad or 
                                         fake_scores.requires_grad))
    
    @staticmethod
    def backward(ctx, grad_output):
        reduction = ctx.saved_data['reduction']
        real_shape = ctx.saved_data['real_shape']
        fake_shape = ctx.saved_data['fake_shape']
        
        # Gradients for Wasserstein loss
        grad_real = -np.ones(real_shape) / np.prod(real_shape)
        grad_fake = np.ones(fake_shape) / np.prod(fake_shape)
        
        return (Tensor(grad_output.data * grad_real),
                Tensor(grad_output.data * grad_fake),
                None)


class PerceptualLoss(CustomLossTemplate):
    """
    Perceptual Loss using pre-trained features
    
    Mathematical Derivation:
    L_perceptual = Σ_l λ_l * ||φ_l(x) - φ_l(y)||²
    where φ_l are features from layer l of a pre-trained network
    
    Use Cases:
    - Image super-resolution
    - Style transfer
    - Image generation tasks
    - When pixel-wise losses are insufficient
    """
    
    def __init__(self, feature_extractor, layers, weights=None, reduction='mean'):
        super().__init__(reduction)
        self.feature_extractor = feature_extractor
        self.layers = layers
        self.weights = weights if weights else [1.0] * len(layers)
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Extract features from both images
        features_pred = self.feature_extractor(y_pred, self.layers)
        features_true = self.feature_extractor(y_true, self.layers)
        
        total_loss = Tensor(np.array(0.0))
        
        for i, (feat_pred, feat_true, weight) in enumerate(
            zip(features_pred, features_true, self.weights)):
            
            layer_loss = MSELoss.apply(feat_pred, feat_true, 'mean')
            total_loss = total_loss + weight * layer_loss
        
        return total_loss


class AdversarialLoss(CustomLossTemplate):
    """
    Adversarial Loss for robust training
    
    Mathematical Derivation:
    L_adv = max_{||δ||≤ε} L(f(x + δ), y)
    
    Use Cases:
    - Adversarial training
    - Improving model robustness
    - Defense against adversarial attacks
    """
    
    def __init__(self, base_loss, epsilon=0.1, alpha=0.01, num_steps=5, reduction='mean'):
        super().__init__(reduction)
        self.base_loss = base_loss
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_steps = num_steps
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # Generate adversarial examples using PGD
        adv_pred = self._pgd_attack(y_pred, y_true)
        
        # Compute loss on adversarial examples
        adv_loss = self.base_loss(adv_pred, y_true)
        clean_loss = self.base_loss(y_pred, y_true)
        
        # Combine clean and adversarial loss
        total_loss = 0.5 * clean_loss + 0.5 * adv_loss
        
        return total_loss
    
    def _pgd_attack(self, x: Tensor, y: Tensor) -> Tensor:
        # Simplified PGD attack implementation
        adv_x = x.clone()
        
        for _ in range(self.num_steps):
            # Compute gradient
            loss = self.base_loss(adv_x, y)
            loss.backward()
            
            # Update adversarial example
            grad_sign = np.sign(adv_x.grad.data)
            adv_x.data = adv_x.data + self.alpha * grad_sign
            
            # Project to epsilon ball
            perturbation = adv_x.data - x.data
            perturbation = np.clip(perturbation, -self.epsilon, self.epsilon)
            adv_x.data = x.data + perturbation
            
            adv_x.zero_grad()
        
        return adv_x

def mse_loss(y_pred: Tensor, y_true: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for MSE Loss"""
    return MSELoss.apply(y_pred, y_true, reduction)

def mae_loss(y_pred: Tensor, y_true: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for MAE Loss"""
    return MAELoss.apply(y_pred, y_true, reduction)

def huber_loss(y_pred: Tensor, y_true: Tensor, delta: float = 1.0, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Huber Loss"""
    return HuberLoss.apply(y_pred, y_true, reduction)

def log_cosh_loss(y_pred: Tensor, y_true: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Log-Cosh Loss"""
    return LogCoshLoss.apply(y_pred, y_true, reduction)

def cross_entropy_loss(y_pred: Tensor, y_true: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Cross Entropy Loss"""
    return CrossEntropyLoss.apply(y_pred, y_true, reduction)

def binary_cross_entropy_loss(y_pred: Tensor, y_true: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Binary Cross Entropy Loss"""
    return BinaryCrossEntropyLoss.apply(y_pred, y_true, reduction)

def focal_loss(y_pred: Tensor, y_true: Tensor, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Focal Loss"""
    return FocalLoss.apply(y_pred, y_true, alpha, gamma, reduction)

def contrastive_loss(embedding1: Tensor, embedding2: Tensor, labels: Tensor, margin: float = 1.0, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Contrastive Loss"""
    return ContrastiveLoss.apply(embedding1, embedding2, labels, margin, reduction)

def triplet_loss(anchor: Tensor, positive: Tensor, negative: Tensor, margin: float = 1.0, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Triplet Loss"""
    return TripletLoss.apply(anchor, positive, negative, margin, reduction)

def wasserstein_loss(real_scores: Tensor, fake_scores: Tensor, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Wasserstein Loss"""
    return WassersteinLoss.apply(real_scores, fake_scores, reduction)


class DiceLoss(Function):
    """
    Dice Loss for segmentation tasks
    
    Mathematical Derivation:
    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Dice Loss = 1 - Dice
    
    Gradient:
    ∂L/∂p = -2 * (y * (sum_p - sum_y) - p * sum_y) / (sum_p + sum_y)²
    
    Use Cases:
    - Image segmentation
    - Medical imaging
    - When dealing with class imbalance
    - Binary and multi-class segmentation
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, smooth=1e-6, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        # Apply sigmoid to predictions
        pred_probs = 1 / (1 + np.exp(-y_pred.data))
        
        # Flatten tensors
        pred_flat = pred_probs.reshape(-1)
        true_flat = y_true.data.reshape(-1)
        
        # Compute intersection and union
        intersection = np.sum(pred_flat * true_flat)
        union = np.sum(pred_flat) + np.sum(true_flat)
        
        # Dice coefficient
        dice = (2.0 * intersection + smooth) / (union + smooth)
        loss = 1 - dice
        
        ctx.save_data(pred_probs=pred_probs, intersection=intersection, 
                     union=union, smooth=smooth, reduction=reduction)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        pred_probs = ctx.saved_data['pred_probs']
        intersection = ctx.saved_data['intersection']
        union = ctx.saved_data['union']
        smooth = ctx.saved_data['smooth']
        
        # Complex gradient computation for Dice loss
        grad_pred = -2.0 * (union * pred_probs - 2 * intersection) / ((union + smooth) ** 2)
        
        # Apply sigmoid derivative
        sigmoid_grad = pred_probs * (1 - pred_probs)
        grad_pred = grad_pred * sigmoid_grad
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None, None


class IoULoss(Function):
    """
    Intersection over Union Loss
    
    Mathematical Derivation:
    IoU = |X ∩ Y| / |X ∪ Y|
    IoU Loss = 1 - IoU
    
    Use Cases:
    - Object detection
    - Instance segmentation  
    - When overlap quality matters more than pixel accuracy
    """
    
    @staticmethod
    def forward(ctx, y_pred, y_true, smooth=1e-6, reduction='mean'):
        y_pred = _ensure_tensor(y_pred)
        y_true = _ensure_tensor(y_true)
        
        # Apply sigmoid to predictions
        pred_probs = 1 / (1 + np.exp(-y_pred.data))
        
        # Flatten tensors
        pred_flat = pred_probs.reshape(-1)
        true_flat = y_true.data.reshape(-1)
        
        # Compute intersection and union
        intersection = np.sum(pred_flat * true_flat)
        union = np.sum(pred_flat) + np.sum(true_flat) - intersection
        
        # IoU coefficient
        iou = (intersection + smooth) / (union + smooth)
        loss = 1 - iou
        
        ctx.save_data(pred_probs=pred_probs, intersection=intersection, 
                     union=union, smooth=smooth, reduction=reduction)
        
        return Tensor(loss, requires_grad=y_pred.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        pred_probs = ctx.saved_data['pred_probs']
        intersection = ctx.saved_data['intersection']
        union = ctx.saved_data['union']
        smooth = ctx.saved_data['smooth']
        
        # IoU gradient computation
        numerator = union * pred_probs - intersection
        denominator = (union + smooth) ** 2
        grad_pred = -numerator / denominator
        
        # Apply sigmoid derivative
        sigmoid_grad = pred_probs * (1 - pred_probs)
        grad_pred = grad_pred * sigmoid_grad
        
        grad_pred_tensor = Tensor(grad_output.data * grad_pred)
        return grad_pred_tensor, None, None, None


class CenterLoss(Function):
    """
    Center Loss for face recognition and feature learning
    
    Mathematical Derivation:
    L_c = (1/2) * Σ ||x_i - c_{y_i}||²
    where c_{y_i} is the center of class y_i
    
    Use Cases:
    - Face recognition
    - Feature learning with intra-class compactness
    - Reducing intra-class variation
    - Learning discriminative features
    """
    
    @staticmethod
    def forward(ctx, features, labels, centers, alpha=0.5, reduction='mean'):
        features = _ensure_tensor(features)
        labels = _ensure_tensor(labels)
        centers = _ensure_tensor(centers)
        
        batch_size = features.shape[0]
        
        # Get centers for current batch
        batch_centers = centers.data[labels.data.astype(int)]
        
        # Compute center loss
        diff = features.data - batch_centers
        loss = 0.5 * np.sum(diff ** 2, axis=1)
        
        ctx.save_for_backwards(features, labels, centers)
        ctx.save_data(batch_centers=batch_centers, alpha=alpha, reduction=reduction)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=features.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        features, labels, centers = ctx.saved_tensors
        batch_centers = ctx.saved_data['batch_centers']
        alpha = ctx.saved_data['alpha']
        reduction = ctx.saved_data['reduction']
        
        # Gradient w.r.t features
        grad_features = features.data - batch_centers
        
        if reduction == 'mean':
            grad_features = grad_features / features.shape[0]
        
        # Update centers (this would typically be done outside the loss function)
        # For completeness, we include the center update computation
        unique_labels = np.unique(labels.data)
        center_updates = np.zeros_like(centers.data)
        
        for label in unique_labels:
            mask = labels.data == label
            if np.any(mask):
                center_updates[int(label)] = alpha * np.mean(
                    batch_centers[mask] - features.data[mask], axis=0)
        
        grad_features_tensor = Tensor(grad_output.data * grad_features)
        return grad_features_tensor, None, None, None, None


class AngularLoss(Function):
    """
    Angular Loss for face recognition
    
    Mathematical Derivation:
    L_angular = -log(exp(s * cos(θ_{y_i} + m)) / 
                     (exp(s * cos(θ_{y_i} + m)) + Σ_{j≠y_i} exp(s * cos(θ_j))))
    
    Use Cases:
    - Face recognition
    - Speaker verification
    - Learning angular-based features
    - When angular distance is more important than Euclidean
    """
    
    @staticmethod
    def forward(ctx, features, labels, weight, scale=30.0, margin=0.5, reduction='mean'):
        features = _ensure_tensor(features)
        labels = _ensure_tensor(labels)
        weight = _ensure_tensor(weight)
        
        # Normalize features and weights
        features_norm = features.data / (np.linalg.norm(features.data, axis=1, keepdims=True) + 1e-8)
        weight_norm = weight.data / (np.linalg.norm(weight.data, axis=0, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        cosine = np.dot(features_norm, weight_norm)
        
        # Apply margin to target class
        batch_size = features.shape[0]
        target_cosine = cosine[np.arange(batch_size), labels.data.astype(int)]
        target_cosine_m = target_cosine - margin
        
        # Create one-hot encoding
        one_hot = np.zeros_like(cosine)
        one_hot[np.arange(batch_size), labels.data.astype(int)] = 1
        
        # Apply margin and scale
        cosine_m = cosine * (1 - one_hot) + target_cosine_m[:, np.newaxis] * one_hot
        scaled_cosine = scale * cosine_m
        
        # Compute softmax cross entropy
        max_vals = np.max(scaled_cosine, axis=1, keepdims=True)
        exp_vals = np.exp(scaled_cosine - max_vals)
        softmax = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        
        loss = -np.log(softmax[np.arange(batch_size), labels.data.astype(int)] + 1e-8)
        
        ctx.save_data(softmax=softmax, one_hot=one_hot, scale=scale, 
                     features_norm=features_norm, weight_norm=weight_norm, reduction=reduction)
        
        if reduction == 'mean':
            loss = np.mean(loss)
        elif reduction == 'sum':
            loss = np.sum(loss)
        
        return Tensor(loss, requires_grad=features.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        softmax = ctx.saved_data['softmax']
        one_hot = ctx.saved_data['one_hot']
        scale = ctx.saved_data['scale']
        reduction = ctx.saved_data['reduction']
        
        # Gradient computation for angular loss
        grad_scaled_cosine = scale * (softmax - one_hot)
        
        if reduction == 'mean':
            grad_scaled_cosine = grad_scaled_cosine / softmax.shape[0]
        
        grad_features_tensor = Tensor(grad_output.data * grad_scaled_cosine)
        return grad_features_tensor, None, None, None, None, None


class DistillationLoss(Function):
    """
    Knowledge Distillation Loss
    
    Mathematical Derivation:
    L_KD = α * L_CE(y, σ(z_s)) + (1-α) * τ² * L_CE(σ(z_t/τ), σ(z_s/τ))
    where z_s = student logits, z_t = teacher logits, τ = temperature
    
    Use Cases:
    - Model compression
    - Transfer learning
    - Training smaller models with teacher guidance
    - Knowledge transfer between models
    """
    
    @staticmethod
    def forward(ctx, student_logits, teacher_logits, labels, temperature=3.0, alpha=0.7, reduction='mean'):
        student_logits = _ensure_tensor(student_logits)
        teacher_logits = _ensure_tensor(teacher_logits)
        labels = _ensure_tensor(labels)
        
        # Hard target loss (student vs true labels)
        student_probs = 1 / (1 + np.exp(-student_logits.data))
        hard_loss = -np.sum(labels.data * np.log(student_probs + 1e-8), axis=1)
        
        # Soft target loss (student vs teacher)
        student_soft = np.exp(student_logits.data / temperature)
        student_soft = student_soft / np.sum(student_soft, axis=1, keepdims=True)
        
        teacher_soft = np.exp(teacher_logits.data / temperature)
        teacher_soft = teacher_soft / np.sum(teacher_soft, axis=1, keepdims=True)
        
        soft_loss = -np.sum(teacher_soft * np.log(student_soft + 1e-8), axis=1)
        
        # Combined loss
        total_loss = alpha * hard_loss + (1 - alpha) * (temperature ** 2) * soft_loss
        
        ctx.save_data(student_probs=student_probs, student_soft=student_soft, 
                     teacher_soft=teacher_soft, temperature=temperature, 
                     alpha=alpha, reduction=reduction)
        
        if reduction == 'mean':
            total_loss = np.mean(total_loss)
        elif reduction == 'sum':
            total_loss = np.sum(total_loss)
        
        return Tensor(total_loss, requires_grad=student_logits.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        student_probs = ctx.saved_data['student_probs']
        student_soft = ctx.saved_data['student_soft']
        teacher_soft = ctx.saved_data['teacher_soft']
        temperature = ctx.saved_data['temperature']
        alpha = ctx.saved_data['alpha']
        reduction = ctx.saved_data['reduction']
        
        # Gradient computation for distillation loss
        hard_grad = alpha * student_probs  # Will be modified by caller with labels
        soft_grad = (1 - alpha) * (temperature ** 2) * (student_soft - teacher_soft) / temperature
        
        total_grad = hard_grad + soft_grad
        
        if reduction == 'mean':
            total_grad = total_grad / student_probs.shape[0]
        
        grad_student_tensor = Tensor(grad_output.data * total_grad)
        return grad_student_tensor, None, None, None, None, None

def dice_loss(y_pred: Tensor, y_true: Tensor, smooth: float = 1e-6, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Dice Loss"""
    return DiceLoss.apply(y_pred, y_true, smooth, reduction)

def iou_loss(y_pred: Tensor, y_true: Tensor, smooth: float = 1e-6, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for IoU Loss"""
    return IoULoss.apply(y_pred, y_true, smooth, reduction)

def center_loss(features: Tensor, labels: Tensor, centers: Tensor, alpha: float = 0.5, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Center Loss"""
    return CenterLoss.apply(features, labels, centers, alpha, reduction)

def angular_loss(features: Tensor, labels: Tensor, weight: Tensor, scale: float = 30.0, margin: float = 0.5, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Angular Loss"""
    return AngularLoss.apply(features, labels, weight, scale, margin, reduction)

def distillation_loss(student_logits: Tensor, teacher_logits: Tensor, labels: Tensor, temperature: float = 3.0, alpha: float = 0.7, reduction: str = 'mean') -> Tensor:
    """Convenience wrapper for Distillation Loss"""
    return DistillationLoss.apply(student_logits, teacher_logits, labels, temperature, alpha, reduction)


class RankingLoss(CustomLossTemplate):
    """
    Ranking Loss for learning-to-rank tasks
    
    Mathematical Derivation:
    L_rank = Σ max(0, margin - (s_+ - s_-))
    where s_+ is score for positive item, s_- is score for negative item
    
    Use Cases:
    - Information retrieval
    - Recommendation systems
    - Learning to rank
    - Search result ordering
    """
    
    def __init__(self, margin=1.0, reduction='mean'):
        super().__init__(reduction)
        self.margin = margin
    
    def forward(self, positive_scores: Tensor, negative_scores: Tensor) -> Tensor:
        # Pairwise ranking loss
        diff = positive_scores - negative_scores
        loss = Tensor(np.maximum(0, self.margin - diff.data))
        return self._apply_reduction(loss)


class VariationalLoss(CustomLossTemplate):
    """
    Variational Loss for VAEs
    
    Mathematical Derivation:
    L_VAE = L_recon + β * KL(q(z|x) || p(z))
    where L_recon is reconstruction loss, KL is KL divergence
    
    Use Cases:
    - Variational Autoencoders
    - Generative modeling
    - Representation learning
    - Disentangled representations
    """
    
    def __init__(self, beta=1.0, reduction='mean'):
        super().__init__(reduction)
        self.beta = beta
    
    def forward(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor) -> Tensor:
        # Reconstruction loss
        recon_loss = mse_loss(recon_x, x, 'sum')
        
        # KL divergence loss
        kl_loss = -0.5 * Tensor(np.sum(1 + logvar.data - mu.data**2 - np.exp(logvar.data)))
        
        total_loss = recon_loss + self.beta * kl_loss
        return self._apply_reduction(total_loss)


class MultiTaskLoss(LossFunction):
    """
    Multi-task loss combiner with automatic weighting
    
    Combines multiple losses with learnable weights based on:
    "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    """
    
    def __init__(self, num_tasks: int, reduction='mean'):
        super().__init__(reduction)
        self.num_tasks = num_tasks
        # Initialize log variance parameters
        self.log_vars = [Tensor(np.array([0.0]), requires_grad=True) for _ in range(num_tasks)]
    
    def forward(self, losses: List[Tensor]) -> Tensor:
        """
        Combine multiple task losses with uncertainty weighting
        
        Args:
            losses: List of individual task losses
            
        Returns:
            Combined weighted loss
        """
        total_loss = Tensor(np.array(0.0))
        
        for i, (loss, log_var) in enumerate(zip(losses, self.log_vars)):
            # Uncertainty weighting: L_total = Σ (1/(2σ²)) * L_i + log(σ²)
            precision = Tensor(np.exp(-log_var.data))
            weighted_loss = precision * loss + log_var
            total_loss = total_loss + weighted_loss
        
        return total_loss
    
    def get_weights(self):
        """Get current task weights"""
        return [np.exp(-log_var.data) for log_var in self.log_vars]


class LossFunctionFactory:
    """
    Factory class for creating and registering custom loss functions
    """
    
    _registry = {
        'mse': MSELoss,
        'mae': MAELoss, 
        'huber': HuberLoss,
        'log_cosh': LogCoshLoss,
        'cross_entropy': CrossEntropyLoss,
        'binary_cross_entropy': BinaryCrossEntropyLoss,
        'focal': FocalLoss,
        'contrastive': ContrastiveLoss,
        'triplet': TripletLoss,
        'wasserstein': WassersteinLoss,
        'dice': DiceLoss,
        'iou': IoULoss,
        'center': CenterLoss,
        'angular': AngularLoss,
        'distillation': DistillationLoss,
    }
    
    @classmethod
    def create(cls, loss_type: str, **kwargs):
        """
        Create a loss function by name
        
        Args:
            loss_type: Name of the loss function
            **kwargs: Arguments to pass to the loss function
            
        Returns:
            Loss function instance
        """
        if loss_type not in cls._registry:
            raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(cls._registry.keys())}")
        
        return cls._registry[loss_type](**kwargs)
    
    @classmethod
    def register(cls, name: str, loss_class):
        """
        Register a custom loss function
        
        Args:
            name: Name to register the loss function under
            loss_class: Loss function class
        """
        cls._registry[name] = loss_class
    
    @classmethod
    def list_available(cls):
        """List all available loss functions"""
        return list(cls._registry.keys())
