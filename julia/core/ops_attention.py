import numpy as np
from julia.core.tensor import Function, Tensor, _ensure_tensor

class ScaledDotProductAttention(Function):
    """
    Autograd-enabled scaled dot-product attention
    
    Computes: Attention(Q,K,V) = softmax(QK^T/√d_k)V
    """
    
    @staticmethod
    def forward(ctx, query, key, value, mask=None, scale=None, dropout_p=0.0, training=True):
        """
        Forward pass for scaled dot-product attention
        
        Args:
            query: Query tensor (..., seq_len_q, d_k)
            key: Key tensor (..., seq_len_k, d_k)  
            value: Value tensor (..., seq_len_k, d_v)
            mask: Optional attention mask
            scale: Scale factor (default: 1/√d_k)
            dropout_p: Dropout probability
            training: Whether in training mode
        """
        query = _ensure_tensor(query)
        key = _ensure_tensor(key)
        value = _ensure_tensor(value)
        
        d_k = query.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(d_k)
        
        # Get shapes
        q_shape = query.shape
        k_shape = key.shape
        v_shape = value.shape
        
        # For multi-head attention, the input is typically (batch_size, num_heads, seq_len, d_k)
        # For regular attention, it's (batch_size, seq_len, d_k)
        
        if len(q_shape) == 4:
            # Multi-head case: (batch_size, num_heads, seq_len, d_k)
            batch_size, num_heads, seq_len_q, d_k = q_shape
            _, _, seq_len_k, _ = k_shape
            _, _, seq_len_v, d_v = v_shape
            
            # Reshape for batch matrix multiplication
            # (batch_size, num_heads, seq_len, d_k) -> (batch_size * num_heads, seq_len, d_k)
            q_reshaped = query.data.reshape(batch_size * num_heads, seq_len_q, d_k)
            k_reshaped = key.data.reshape(batch_size * num_heads, seq_len_k, d_k)
            v_reshaped = value.data.reshape(batch_size * num_heads, seq_len_v, d_v)
            
            effective_batch_size = batch_size * num_heads
            
        elif len(q_shape) == 3:
            # Regular case: (batch_size, seq_len, d_k)
            batch_size, seq_len_q, d_k = q_shape
            _, seq_len_k, _ = k_shape
            _, seq_len_v, d_v = v_shape
            
            q_reshaped = query.data
            k_reshaped = key.data
            v_reshaped = value.data
            
            effective_batch_size = batch_size
            num_heads = 1
            
        else:
            raise ValueError(f"Unsupported query shape: {q_shape}")
        
        # Compute attention scores: Q @ K^T
        scores_data = np.zeros((effective_batch_size, seq_len_q, seq_len_k))
        for b in range(effective_batch_size):
            scores_data[b] = np.matmul(q_reshaped[b], k_reshaped[b].T) * scale
        
        # Apply mask if provided
        if mask is not None:
            mask_data = mask.data
            
            # Handle different mask shapes and broadcast to scores shape
            if mask_data.shape != scores_data.shape:
                # For multi-head attention, we need to handle the mask correctly
                if len(q_shape) == 4 and len(mask_data.shape) == 2:
                    # Mask is (batch_size, seq_len) or (seq_len, seq_len)
                    if mask_data.shape[0] == batch_size and mask_data.shape[1] == seq_len_q:
                        # Padding mask: (batch_size, seq_len) -> expand for attention
                        # Convert to (batch_size, 1, seq_len) then to (batch_size, seq_len, seq_len)
                        mask_expanded = mask_data[:, None, :] | mask_data[:, :, None]
                        # Repeat for each head: (batch_size, seq_len, seq_len) -> (batch_size * num_heads, seq_len, seq_len)
                        mask_data = np.repeat(mask_expanded, num_heads, axis=0)
                    elif mask_data.shape[0] == mask_data.shape[1] == seq_len_q:
                        # Causal mask: (seq_len, seq_len) -> broadcast to all batches and heads
                        mask_data = np.broadcast_to(mask_data[None, :, :], (effective_batch_size, seq_len_q, seq_len_k))
                    else:
                        raise ValueError(f"Cannot handle mask shape {mask_data.shape} with query shape {q_shape}")
                        
                elif len(q_shape) == 3 and len(mask_data.shape) == 2:
                    # Regular attention case
                    if mask_data.shape[0] == batch_size and mask_data.shape[1] == seq_len_q:
                        # Padding mask: (batch_size, seq_len) -> (batch_size, seq_len, seq_len)
                        mask_data = mask_data[:, None, :] | mask_data[:, :, None]
                    elif mask_data.shape[0] == mask_data.shape[1] == seq_len_q:
                        # Causal mask: (seq_len, seq_len) -> broadcast to all batches
                        mask_data = np.broadcast_to(mask_data[None, :, :], (batch_size, seq_len_q, seq_len_k))
                    else:
                        raise ValueError(f"Cannot handle mask shape {mask_data.shape} with query shape {q_shape}")
                        
                elif len(mask_data.shape) == 3:
                    # Mask is already in (batch_size, seq_len, seq_len) format
                    if len(q_shape) == 4:
                        # Need to repeat for each head
                        mask_data = np.repeat(mask_data, num_heads, axis=0)
                else:
                    raise ValueError(f"Cannot handle mask shape {mask_data.shape} with query shape {q_shape}")
            
            # Apply mask by adding large negative values to masked positions
            scores_data = scores_data + (mask_data * -1e9)
        
        # Apply softmax
        max_scores = np.max(scores_data, axis=-1, keepdims=True)
        exp_scores = np.exp(scores_data - max_scores)
        sum_exp_scores = np.sum(exp_scores, axis=-1, keepdims=True)
        attention_weights_data = exp_scores / sum_exp_scores
        
        # Apply dropout if training
        dropout_mask = None
        if training and dropout_p > 0:
            dropout_mask = (np.random.rand(*attention_weights_data.shape) > dropout_p).astype(np.float32)
            attention_weights_data = attention_weights_data * dropout_mask / (1 - dropout_p)
        
        # Compute attention output: attention_weights @ V
        output_data = np.zeros((effective_batch_size, seq_len_q, d_v))
        for b in range(effective_batch_size):
            output_data[b] = np.matmul(attention_weights_data[b], v_reshaped[b])
        
        # Reshape back to original format
        if len(q_shape) == 4:
            # Multi-head case: reshape back to (batch_size, num_heads, seq_len_q, d_v)
            output_data = output_data.reshape(batch_size, num_heads, seq_len_q, d_v)
            attention_weights_data = attention_weights_data.reshape(batch_size, num_heads, seq_len_q, seq_len_k)
        
        # Save for backward
        ctx.save_for_backwards(query, key, value)
        if mask is not None:
            ctx.save_for_backwards(query, key, value, mask)
        
        ctx.save_data(
            attention_weights=attention_weights_data,
            scale=scale,
            dropout_p=dropout_p,
            training=training,
            dropout_mask=dropout_mask,
            q_shape=q_shape,
            k_shape=k_shape,
            v_shape=v_shape,
            has_mask=mask is not None
        )
        
        output = Tensor(output_data, requires_grad=query.requires_grad or key.requires_grad or value.requires_grad)
        attention_weights = Tensor(attention_weights_data, requires_grad=False)
        
        return output, attention_weights
    
    @staticmethod
    def backward(ctx, grad_output, grad_attention_weights=None):
        """
        Backward pass for scaled dot-product attention
        """
        saved_tensors = ctx.saved_tensors
        if ctx.saved_data['has_mask']:
            query, key, value, mask = saved_tensors
        else:
            query, key, value = saved_tensors
            mask = None
            
        attention_weights = ctx.saved_data['attention_weights']
        scale = ctx.saved_data['scale']
        dropout_p = ctx.saved_data['dropout_p']
        training = ctx.saved_data['training']
        dropout_mask = ctx.saved_data['dropout_mask']
        q_shape = ctx.saved_data['q_shape']
        k_shape = ctx.saved_data['k_shape']
        v_shape = ctx.saved_data['v_shape']
        
        grad_output = _ensure_tensor(grad_output)
        
        # Handle reshaping for backward pass similar to forward
        if len(q_shape) == 4:
            batch_size, num_heads, seq_len_q, d_k = q_shape
            _, _, seq_len_k, _ = k_shape
            _, _, seq_len_v, d_v = v_shape
            
            effective_batch_size = batch_size * num_heads
            
            grad_out_reshaped = grad_output.data.reshape(effective_batch_size, seq_len_q, d_v)
            attention_reshaped = attention_weights.reshape(effective_batch_size, seq_len_q, seq_len_k)
            
            q_reshaped = query.data.reshape(effective_batch_size, seq_len_q, d_k)
            k_reshaped = key.data.reshape(effective_batch_size, seq_len_k, d_k)
            v_reshaped = value.data.reshape(effective_batch_size, seq_len_v, d_v)
            
        elif len(q_shape) == 3:
            batch_size, seq_len_q, d_k = q_shape
            _, seq_len_k, _ = k_shape
            _, seq_len_v, d_v = v_shape
            
            effective_batch_size = batch_size
            num_heads = 1
            
            grad_out_reshaped = grad_output.data
            attention_reshaped = attention_weights
            
            q_reshaped = query.data
            k_reshaped = key.data
            v_reshaped = value.data
        
        # Initialize gradients
        grad_query_data = np.zeros_like(q_reshaped)
        grad_key_data = np.zeros_like(k_reshaped)
        grad_value_data = np.zeros_like(v_reshaped)
        
        for b in range(effective_batch_size):
            # Gradient w.r.t. value: A^T @ grad_output
            grad_value_data[b] = np.matmul(attention_reshaped[b].T, grad_out_reshaped[b])
            
            # Gradient w.r.t. attention weights: grad_output @ V^T
            grad_attention = np.matmul(grad_out_reshaped[b], v_reshaped[b].T)
            
            # Apply dropout mask in backward if it was applied in forward
            if training and dropout_p > 0 and dropout_mask is not None:
                grad_attention = grad_attention * dropout_mask[b] / (1 - dropout_p)
            
            # Gradient of softmax
            attention_b = attention_reshaped[b]
            sum_term = np.sum(attention_b * grad_attention, axis=-1, keepdims=True)
            grad_scores = attention_b * (grad_attention - sum_term)
            
            # Apply mask gradient (masked positions should have zero gradient)
            if mask is not None:
                # Handle mask for backward pass
                mask_data = mask.data
                if len(q_shape) == 4 and len(mask_data.shape) == 2:
                    if mask_data.shape[0] == batch_size and mask_data.shape[1] == seq_len_q:
                        # Padding mask case
                        batch_idx = b // num_heads
                        batch_mask = mask_data[batch_idx]
                        mask_b = batch_mask[:, None] | batch_mask[None, :]
                    elif mask_data.shape[0] == mask_data.shape[1] == seq_len_q:
                        # Causal mask case
                        mask_b = mask_data
                    else:
                        mask_b = np.zeros_like(grad_scores, dtype=bool)
                elif len(q_shape) == 3 and len(mask_data.shape) == 2:
                    if mask_data.shape[0] == batch_size and mask_data.shape[1] == seq_len_q:
                        batch_mask = mask_data[b]
                        mask_b = batch_mask[:, None] | batch_mask[None, :]
                    else:
                        mask_b = mask_data
                else:
                    mask_b = np.zeros_like(grad_scores, dtype=bool)
                
                grad_scores = grad_scores * (1 - mask_b.astype(float))
            
            # Scale the gradient
            grad_scores = grad_scores * scale
            
            # Gradient w.r.t. query: grad_scores @ K
            grad_query_data[b] = np.matmul(grad_scores, k_reshaped[b])
            
            # Gradient w.r.t. key: grad_scores^T @ Q
            grad_key_data[b] = np.matmul(grad_scores.T, q_reshaped[b])
        
        # Reshape back to original shapes
        if len(q_shape) == 4:
            grad_query_data = grad_query_data.reshape(q_shape)
            grad_key_data = grad_key_data.reshape(k_shape)
            grad_value_data = grad_value_data.reshape(v_shape)
        
        grad_query = Tensor(grad_query_data) if query.requires_grad else None
        grad_key = Tensor(grad_key_data) if key.requires_grad else None
        grad_value = Tensor(grad_value_data) if value.requires_grad else None
        
        return grad_query, grad_key, grad_value, None, None, None, None


class MultiHeadReshape(Function):
    """
    Autograd function for reshaping tensors for multi-head attention
    """
    
    @staticmethod
    def forward(ctx, x, num_heads):
        """
        Reshape tensor for multi-head attention
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            num_heads: Number of attention heads
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        x = _ensure_tensor(x)
        batch_size, seq_len, d_model = x.shape
        d_k = d_model // num_heads
        
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        reshaped = x.data.reshape(batch_size, seq_len, num_heads, d_k)
        
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        transposed = reshaped.transpose(0, 2, 1, 3)
        
        ctx.save_data(original_shape=x.shape, num_heads=num_heads)
        
        return Tensor(transposed, requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for multi-head reshape
        """
        original_shape = ctx.saved_data['original_shape']
        
        grad_output = _ensure_tensor(grad_output)
        
        # Transpose back: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        transposed_back = grad_output.data.transpose(0, 2, 1, 3)
        
        # Reshape back to original: (batch_size, seq_len, d_model)
        reshaped_back = transposed_back.reshape(original_shape)
        
        return Tensor(reshaped_back), None


class MultiHeadMerge(Function):
    """
    Autograd function for merging multi-head attention outputs
    """
    
    @staticmethod
    def forward(ctx, x, target_shape):
        """
        Merge multi-head attention outputs
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
            target_shape: Target shape (batch_size, seq_len, d_model)
            
        Returns:
            Merged tensor of shape (batch_size, seq_len, d_model)
        """
        x = _ensure_tensor(x)
        
        # Transpose: (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k)
        transposed = x.data.transpose(0, 2, 1, 3)
        
        # Reshape: (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        merged = transposed.reshape(target_shape)
        
        ctx.save_data(input_shape=x.shape)
        
        return Tensor(merged, requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for multi-head merge
        """
        input_shape = ctx.saved_data['input_shape']
        
        grad_output = _ensure_tensor(grad_output)
        batch_size, seq_len, d_model = grad_output.shape
        num_heads = input_shape[1]
        d_k = input_shape[3]
        
        # Reshape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, num_heads, d_k)
        reshaped = grad_output.data.reshape(batch_size, seq_len, num_heads, d_k)
        
        # Transpose: (batch_size, seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        transposed = reshaped.transpose(0, 2, 1, 3)
        
        return Tensor(transposed), None


class GroupedKVExpansion(Function):
    """
    Autograd function for expanding K,V tensors in Grouped Query Attention
    """
    
    @staticmethod
    def forward(ctx, x, expansion_factor):
        """
        Expand K or V tensor for grouped query attention
        
        Args:
            x: Input tensor of shape (batch_size, num_kv_heads, seq_len, d_k)
            expansion_factor: Factor to expand by (num_query_heads // num_kv_heads)
            
        Returns:
            Expanded tensor of shape (batch_size, num_query_heads, seq_len, d_k)
        """
        x = _ensure_tensor(x)
        
        # Repeat each head expansion_factor times
        expanded = np.repeat(x.data, expansion_factor, axis=1)
        
        ctx.save_data(original_shape=x.shape, expansion_factor=expansion_factor)
        
        return Tensor(expanded, requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for grouped KV expansion
        """
        original_shape = ctx.saved_data['original_shape']
        expansion_factor = ctx.saved_data['expansion_factor']
        
        grad_output = _ensure_tensor(grad_output)
        
        # Sum gradients for repeated heads
        batch_size, expanded_heads, seq_len, d_k = grad_output.shape
        original_heads = original_shape[1]
        
        grad_data = grad_output.data.reshape(batch_size, original_heads, expansion_factor, seq_len, d_k)
        grad_summed = np.sum(grad_data, axis=2)  # Sum over expansion dimension
        
        return Tensor(grad_summed), None


class AttentionMask(Function):
    """
    Autograd function for applying attention masks
    """
    
    @staticmethod
    def forward(ctx, scores, mask, mask_value=-1e9):
        """
        Apply attention mask to scores
        
        Args:
            scores: Attention scores tensor
            mask: Boolean mask tensor (True = masked)
            mask_value: Value to set for masked positions
            
        Returns:
            Masked scores tensor
        """
        scores = _ensure_tensor(scores)
        mask = _ensure_tensor(mask)
        
        # Apply mask
        masked_scores = scores.data + (mask.data * mask_value)
        
        ctx.save_data(mask=mask.data, mask_value=mask_value)
        
        return Tensor(masked_scores, requires_grad=scores.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for attention mask
        """
        mask = ctx.saved_data['mask']
        
        grad_output = _ensure_tensor(grad_output)
        
        # Gradient flows through unmasked positions only
        grad_scores = grad_output.data * (1 - mask.astype(float))
        
        return Tensor(grad_scores), None, None


class RotaryPositionalEmbedding(Function):
    """
    Rotary Positional Embedding (RoPE) autograd function
    """
    
    @staticmethod
    def forward(ctx, x, cos, sin):
        """
        Apply rotary positional embedding
        
        Args:
            x: Input tensor of shape (..., seq_len, d_k)
            cos: Cosine values for rotation
            sin: Sine values for rotation
            
        Returns:
            Tensor with rotary positional embedding applied
        """
        x = _ensure_tensor(x)
        cos = _ensure_tensor(cos)
        sin = _ensure_tensor(sin)
        
        # Split x into two halves
        d_k = x.shape[-1]
        x1, x2 = np.split(x.data, 2, axis=-1)
        
        # Apply rotation
        rotated_data = np.concatenate([
            x1 * cos.data - x2 * sin.data,
            x1 * sin.data + x2 * cos.data
        ], axis=-1)
        
        ctx.save_for_backwards(cos, sin)
        ctx.save_data(x_shape=x.shape)
        
        return Tensor(rotated_data, requires_grad=x.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for rotary positional embedding
        """
        cos, sin = ctx.saved_tensors
        x_shape = ctx.saved_data['x_shape']
        
        grad_output = _ensure_tensor(grad_output)
        
        # Split gradient into two halves
        grad1, grad2 = np.split(grad_output.data, 2, axis=-1)
        
        # Apply inverse rotation to gradient
        grad_x_data = np.concatenate([
            grad1 * cos.data + grad2 * sin.data,
            -grad1 * sin.data + grad2 * cos.data
        ], axis=-1)
        
        return Tensor(grad_x_data), None, None


# Utility functions using autograd operations

def scaled_dot_product_attention_autograd(query, key, value, mask=None, dropout_p=0.0, training=True):
    """
    Scaled dot-product attention using autograd functions
    
    Args:
        query: Query tensor (..., seq_len_q, d_k)
        key: Key tensor (..., seq_len_k, d_k)
        value: Value tensor (..., seq_len_k, d_v)
        mask: Optional attention mask
        dropout_p: Dropout probability
        training: Whether in training mode
        
    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    return ScaledDotProductAttention.apply(query, key, value, mask, None, dropout_p, training)


def multi_head_attention_autograd(query, key, value, num_heads, w_q, w_k, w_v, w_o, 
                                 mask=None, dropout_p=0.0, training=True):
    """
    Multi-head attention using autograd functions
    
    Args:
        query: Query tensor (batch_size, seq_len, d_model)
        key: Key tensor (batch_size, seq_len, d_model)
        value: Value tensor (batch_size, seq_len, d_model)
        num_heads: Number of attention heads
        w_q, w_k, w_v, w_o: Weight matrices for projections
        mask: Optional attention mask
        dropout_p: Dropout probability
        training: Whether in training mode
        
    Returns:
        output: Attention output
        attention_weights: Attention weights
    """
    # Linear projections
    Q = w_q(query)
    K = w_k(key)
    V = w_v(value)
    
    # Reshape for multi-head attention
    Q_heads = MultiHeadReshape.apply(Q, num_heads)
    K_heads = MultiHeadReshape.apply(K, num_heads)
    V_heads = MultiHeadReshape.apply(V, num_heads)
    
    # Apply attention
    output_heads, attention_weights = ScaledDotProductAttention.apply(
        Q_heads, K_heads, V_heads, mask, None, dropout_p, training
    )
    
    # Merge heads
    batch_size, seq_len = query.shape[:2]
    d_model = query.shape[2]
    target_shape = (batch_size, seq_len, d_model)
    
    output_merged = MultiHeadMerge.apply(output_heads, target_shape)
    
    # Final projection
    output = w_o(output_merged)
    
    return output, attention_weights


def apply_rotary_positional_embedding(x, seq_len):
    """
    Apply rotary positional embedding to input tensor
    
    Args:
        x: Input tensor of shape (..., seq_len, d_k)
        seq_len: Sequence length
        
    Returns:
        Tensor with RoPE applied
    """
    d_k = x.shape[-1]
    
    # Create position indices
    position = np.arange(seq_len).reshape(-1, 1)
    
    # Create frequency indices
    freq_indices = np.arange(0, d_k // 2) * 2.0 / d_k
    frequencies = 1.0 / (10000.0 ** freq_indices)
    
    # Create angles
    angles = position * frequencies
    
    # Create cos and sin values
    cos_vals = np.cos(angles)
    sin_vals = np.sin(angles)
    
    # Expand to match input dimensions
    cos_expanded = np.repeat(cos_vals, 2, axis=-1)
    sin_expanded = np.repeat(sin_vals, 2, axis=-1)
    
    # Add batch dimensions
    for _ in range(len(x.shape) - 2):
        cos_expanded = cos_expanded[None, ...]
        sin_expanded = sin_expanded[None, ...]
    
    cos_tensor = Tensor(cos_expanded, requires_grad=False)
    sin_tensor = Tensor(sin_expanded, requires_grad=False)
    
    return RotaryPositionalEmbedding.apply(x, cos_tensor, sin_tensor)


# Register shape inference functions
def register_attention_shape_inference():
    """Register shape inference functions for attention operations"""
    try:
        from julia.core.utils.op_registry import registry
        
        @registry.register_shape_inference("ScaledDotProductAttention")
        def infer_attention_shape(node, input_shapes):
            """Shape inference for scaled dot-product attention"""
            if len(input_shapes) < 3:
                return None
            
            query_shape = input_shapes[0]
            value_shape = input_shapes[2]
            
            if query_shape is None or value_shape is None:
                return None
            
            # Output shape: (..., seq_len_q, d_v)
            output_shape = query_shape[:-1] + (value_shape[-1],)
            attention_shape = query_shape[:-1] + (input_shapes[1][-2],)  # (..., seq_len_q, seq_len_k)
            
            return output_shape
        
        @registry.register_shape_inference("MultiHeadReshape")
        def infer_multihead_reshape_shape(node, input_shapes):
            """Shape inference for multi-head reshape"""
            if len(input_shapes) < 1:
                return None
            
            input_shape = input_shapes[0]
            if input_shape is None or len(input_shape) < 3:
                return None
            
            batch_size, seq_len, d_model = input_shape
            # Assuming num_heads is stored in node attributes
            num_heads = node.attributes.get('num_heads', 8)
            d_k = d_model // num_heads
            
            return (batch_size, num_heads, seq_len, d_k)
        
        @registry.register_shape_inference("MultiHeadMerge")
        def infer_multihead_merge_shape(node, input_shapes):
            """Shape inference for multi-head merge"""
            if len(input_shapes) < 1:
                return None
            
            # Target shape should be in node attributes
            target_shape = node.attributes.get('target_shape')
            if target_shape:
                return tuple(target_shape)
            
            # Fallback: infer from input shape
            input_shape = input_shapes[0]
            if input_shape is None or len(input_shape) < 4:
                return None
            
            batch_size, num_heads, seq_len, d_k = input_shape
            d_model = num_heads * d_k
            
            return (batch_size, seq_len, d_model)
        
        @registry.register_shape_inference("GroupedKVExpansion")
        def infer_grouped_kv_expansion_shape(node, input_shapes):
            """Shape inference for grouped KV expansion"""
            if len(input_shapes) < 1:
                return None
            
            input_shape = input_shapes[0]
            if input_shape is None or len(input_shape) < 4:
                return None
            
            batch_size, num_kv_heads, seq_len, d_k = input_shape
            expansion_factor = node.attributes.get('expansion_factor', 4)
            num_query_heads = num_kv_heads * expansion_factor
            
            return (batch_size, num_query_heads, seq_len, d_k)
        
        @registry.register_shape_inference("AttentionMask")
        def infer_attention_mask_shape(node, input_shapes):
            """Shape inference for attention mask - preserves input shape"""
            if len(input_shapes) < 1:
                return None
            return input_shapes[0]
        
        @registry.register_shape_inference("RotaryPositionalEmbedding")
        def infer_rope_shape(node, input_shapes):
            """Shape inference for rotary positional embedding - preserves input shape"""
            if len(input_shapes) < 1:
                return None
            return input_shapes[0]
            
    except ImportError:
        pass

# Register shape inference functions
register_attention_shape_inference()
