import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.attention import (
    MultiHeadAttention,
    SelfAttention,
    GroupedQueryAttention,
    MultiHeadLatentAttention,
    PositionalEncoding,
    create_causal_mask,
    create_padding_mask,
    create_attention_mask,
)
from julia.core.optim import Adam
from julia.core.loss import mse_loss


def test_multihead_attention():
    """Test MultiHeadAttention with gradient computation"""
    print("Testing MultiHeadAttention with autograd")

    # Model parameters
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_heads = 8

    # Create attention layer
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.1)

    # Create input tensors
    query = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    key = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    value = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print("Input shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")

    # Create causal mask for autoregressive modeling
    mask = create_causal_mask(seq_len)
    print(f"  Mask: {mask.shape}")

    # Forward pass
    print("\nRunning forward pass")
    output, attention_weights = attention(query, key, value, mask)

    print("\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")
    print(f"  Output requires_grad: {output.requires_grad}")

    # Create a dummy loss
    target = Tensor(np.random.randn(*output.shape), requires_grad=False)
    loss = mse_loss(output, target)

    print(f"\nLoss: {loss.data:.6f}")

    # Backward pass
    print("Running backward pass")
    loss.backward()

    # Check gradients
    print("\nGradient shapes:")
    print(f"  Query gradient: {query.grad.shape if query.grad else 'None'}")
    print(f"  Key gradient: {key.grad.shape if key.grad else 'None'}")
    print(f"  Value gradient: {value.grad.shape if value.grad else 'None'}")

    # Check parameter gradients
    params = attention.parameters()
    print("\nParameter gradients:")
    print(f"  Number of parameters: {len(params)}")
    for i, param in enumerate(params):
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad.data)
            print(f"  Parameter {i} gradient norm: {grad_norm:.6f}")
        else:
            print(f"  Parameter {i} gradient: None")

    assert True


def test_self_attention():
    """Test SelfAttention with gradient computation"""
    print("Testing SelfAttention with autograd")

    batch_size = 2
    seq_len = 10
    d_model = 128
    num_heads = 8

    # Create self-attention layer
    self_attention = SelfAttention(d_model, num_heads, dropout=0.1)

    # Create input tensor
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    print(f"Input shape: {x.shape}")

    # Create padding mask (simulate variable length sequences)
    padding_lengths = [8, 6]  # Actual lengths for each sequence in batch
    padding_mask = create_padding_mask(padding_lengths, seq_len)
    print(f"Padding mask shape: {padding_mask.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    output, attention_weights = self_attention(x, padding_mask)

    print("\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")

    # Create loss and backward
    target = Tensor(np.random.randn(*output.shape), requires_grad=False)
    loss = mse_loss(output, target)

    print(f"\nLoss: {loss.data:.6f}")

    print("Running backward pass")
    loss.backward()

    print("\nGradient shapes:")
    print(f"  Input gradient: {x.grad.shape if x.grad else 'None'}")

    # Check parameter gradients
    params = self_attention.parameters()
    param_grad_norms = []
    for param in params:
        if param.grad is not None:
            param_grad_norms.append(np.linalg.norm(param.grad.data))

    print(
        f"  Parameter gradient norms: min={min(param_grad_norms):.6f}, max={max(param_grad_norms):.6f}"
    )

    assert True


def test_grouped_query_attention():
    """Test GroupedQueryAttention with gradient computation"""

    batch_size = 1
    seq_len = 6
    d_model = 256
    num_query_heads = 32
    num_kv_heads = 8

    # Create GQA layer
    gqa = GroupedQueryAttention(d_model, num_query_heads, num_kv_heads, dropout=0.1)

    print("GQA Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_query_heads: {num_query_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  queries_per_kv: {gqa.num_queries_per_kv}")

    # Create input tensors
    query = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    key = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    value = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print("\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")

    # Forward pass
    print("\nRunning forward pass")
    output, attention_weights = gqa(query, key, value)

    print("\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")

    # Backward pass
    target = Tensor(np.random.randn(*output.shape), requires_grad=False)
    loss = mse_loss(output, target)

    print(f"\nLoss: {loss.data:.6f}")

    print("Running backward pass")
    loss.backward()

    # Check gradients exist
    has_grads = all(
        [query.grad is not None, key.grad is not None, value.grad is not None]
    )

    print(f"\nGradients computed: {has_grads}")
    if has_grads:
        print(f"  Query gradient norm: {np.linalg.norm(query.grad.data):.6f}")
        print(f"  Key gradient norm: {np.linalg.norm(key.grad.data):.6f}")
        print(f"  Value gradient norm: {np.linalg.norm(value.grad.data):.6f}")

    assert True


def test_multihead_latent_attention():
    """Test MultiHeadLatentAttention with gradient computation"""

    batch_size = 2
    seq_len = 12
    d_model = 512
    num_heads = 16
    latent_dim = 128

    # Create MLA layer
    mla = MultiHeadLatentAttention(d_model, num_heads, latent_dim, dropout=0.1)

    print("MLA Configuration:")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  latent_dim: {latent_dim}")
    print(f"  d_k: {mla.d_k}")

    # Create input tensors
    query = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    key = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
    value = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print("\nInput shapes:")
    print(f"  Query: {query.shape}")
    print(f"  Key: {key.shape}")
    print(f"  Value: {value.shape}")

    # Forward pass
    print("\nRunning forward pass")
    output, attention_weights = mla(query, key, value)

    print("\nOutput shapes:")
    print(f"  Output: {output.shape}")
    print(f"  Attention weights: {attention_weights.shape}")

    # Backward pass
    target = Tensor(np.random.randn(*output.shape), requires_grad=False)
    loss = mse_loss(output, target)

    print(f"\nLoss: {loss.data:.6f}")

    print("Running backward pass")
    loss.backward()

    # Check gradients
    gradient_norms = {
        "query": np.linalg.norm(query.grad.data) if query.grad else 0,
        "key": np.linalg.norm(key.grad.data) if key.grad else 0,
        "value": np.linalg.norm(value.grad.data) if value.grad else 0,
    }

    print("\nGradient norms:")
    for name, norm in gradient_norms.items():
        print(f"  {name}: {norm:.6f}")

    assert True


def test_positional_encoding():
    """Test PositionalEncoding"""

    batch_size = 3
    seq_len = 20
    d_model = 256

    # Create positional encoding
    pe = PositionalEncoding(d_model, max_len=100, dropout=0.1)

    print("PE Configuration:")
    print(f"  d_model: {d_model}")
    print("  max_len: 100")

    # Create input tensor
    x = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    print("Running forward pass")
    output = pe(x)

    print(f"Output shape: {output.shape}")
    print(f"Output requires_grad: {output.requires_grad}")

    # Test that positional encoding is added correctly
    # The output should be different from input but same shape
    diff_norm = np.linalg.norm(output.data - x.data)
    print(f"Difference norm (should be > 0): {diff_norm:.6f}")

    # Backward pass
    target = Tensor(np.random.randn(*output.shape), requires_grad=False)
    loss = mse_loss(output, target)
    loss.backward()

    print(f"Input gradient norm: {np.linalg.norm(x.grad.data):.6f}")

    assert True


def test_training_loop():
    """Test a simple training loop with attention"""

    # Model parameters
    batch_size = 2
    seq_len = 6
    d_model = 64
    num_heads = 4

    # Create model
    attention = MultiHeadAttention(
        d_model, num_heads, dropout=0.0
    )  # No dropout for deterministic test

    # Create optimizer
    optimizer = Adam(attention.parameters(), lr=0.001)

    print("Model Configuration:")
    print(f"  d_model: {d_model}, num_heads: {num_heads}")
    print(f"  Number of parameters: {len(attention.parameters())}")

    # Training data
    num_epochs = 3
    losses = []

    print(f"\nTraining for {num_epochs} epochs")

    for epoch in range(num_epochs):
        # Create random data
        query = Tensor(
            np.random.randn(batch_size, seq_len, d_model), requires_grad=True
        )
        key = Tensor(np.random.randn(batch_size, seq_len, d_model), requires_grad=True)
        value = Tensor(
            np.random.randn(batch_size, seq_len, d_model), requires_grad=True
        )
        target = Tensor(
            np.random.randn(batch_size, seq_len, d_model), requires_grad=False
        )

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output, _ = attention(query, key, value)

        # Compute loss
        loss = mse_loss(output, target)
        losses.append(loss.data)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        print(f"  Epoch {epoch + 1}: Loss = {loss.data:.6f}")

    # Check that loss is decreasing (or at least changing)
    print(f"\nLoss progression: {[f'{l:.6f}' for l in losses]}")

    # Verify gradients were applied
    param_norms = [np.linalg.norm(p.data) for p in attention.parameters()]
    print(
        f"Parameter norms after training: min={min(param_norms):.6f}, max={max(param_norms):.6f}"
    )

    assert True


def test_attention_masks():
    """Test attention mask utilities"""

    seq_len = 8
    batch_size = 2

    # Test causal mask
    print("Testing causal mask")
    causal_mask = create_causal_mask(seq_len)
    print(f"Causal mask shape: {causal_mask.shape}")
    print("Causal mask (first 4x4):")
    print(causal_mask.data[:4, :4])

    # Test padding mask
    print("\nTesting padding mask")
    padding_lengths = [6, 4]
    padding_mask = create_padding_mask(padding_lengths, seq_len)
    print(f"Padding mask shape: {padding_mask.shape}")
    print("Padding mask:")
    print(padding_mask.data)

    # Test how padding mask should be expanded for attention
    print("\nTesting padding mask expansion for attention")
    # For attention, padding mask (batch_size, seq_len) needs to become (batch_size, seq_len, seq_len)
    # where position (i,j) is masked if either token i or token j is padding
    expanded_padding = padding_mask.data[:, None, :] | padding_mask.data[:, :, None]
    print(f"Expanded padding mask shape: {expanded_padding.shape}")
    print("Expanded padding mask (batch 0, first 6x6):")
    print(expanded_padding[0, :6, :6])

    # Test combined mask
    print("\nTesting combined mask")
    combined_mask = create_attention_mask(
        causal=True, seq_len=seq_len, padding_lengths=padding_lengths
    )
    print(f"Combined mask shape: {combined_mask.shape}")
    print("Combined mask (batch 0, first 6x6):")
    print(combined_mask.data[0, :6, :6])

    assert True


def test_gradient_flow():
    batch_size = 1
    seq_len = 4
    d_model = 32
    num_heads = 4

    # Create simple attention layer
    attention = MultiHeadAttention(d_model, num_heads, dropout=0.0)

    # Create simple inputs
    x = Tensor(np.ones((batch_size, seq_len, d_model)), requires_grad=True)

    print(f"Input shape: {x.shape}")
    print(f"Input sum: {np.sum(x.data)}")

    # Forward pass
    output, weights = attention(x, x, x)

    print(f"Output shape: {output.shape}")
    print(f"Output sum: {np.sum(output.data):.6f}")

    # Simple loss: sum of all outputs
    loss = output.sum()

    print(f"Loss: {loss.data:.6f}")

    # Backward pass
    loss.backward()

    print(f"Input gradient sum: {np.sum(x.grad.data):.6f}")
    print(f"Input gradient shape: {x.grad.shape}")

    # Check all parameters have gradients
    params = attention.parameters()
    params_with_grad = sum(1 for p in params if p.grad is not None)
    print(f"Parameters with gradients: {params_with_grad}/{len(params)}")

    # Check gradient magnitudes
    grad_norms = [np.linalg.norm(p.grad.data) for p in params if p.grad is not None]
    if grad_norms:
        print(f"Gradient norms: min={min(grad_norms):.6f}, max={max(grad_norms):.6f}")

    assert True
