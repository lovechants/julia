"""
Fixed Transformer Example using Julia Framework
A complete implementation of a Transformer model for sequence-to-sequence tasks
"""

import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.layers import Layer, Linear, Sequential, Dropout
from julia.core.nn.attention import MultiHeadAttention, PositionalEncoding
from julia.core.loss import cross_entropy_loss
from julia.core.optim import Adam
from julia.core.autograd_engine import no_grad

class TransformerBlock(Layer):
    """Single Transformer block with self-attention and feed-forward"""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Multi-head self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = Sequential(
            Linear(d_model, d_ff),
            Linear(d_ff, d_model)
        )
        
        # Dropout layers
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        
        return x

class TransformerEncoder(Layer):
    """Multi-layer Transformer encoder"""
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding
        self.embedding = Linear(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.layers = []
        for _ in range(num_layers):
            layer = TransformerBlock(d_model, num_heads, d_ff, dropout)
            self.layers.append(layer)
        
        # Output projection
        self.output_projection = Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Convert token indices to embeddings (simplified)
        # In practice, you'd use proper embedding lookup
        x = self.embedding(x)
        
        # Scale embeddings
        x = x * Tensor(np.sqrt(self.d_model))
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask)
        
        # Project to vocabulary
        output = self.output_projection(x)
        
        return output

class SimpleTransformer(Layer):
    """Complete Transformer model for language modeling"""
    
    def __init__(self, vocab_size=1000, d_model=256, num_heads=8, num_layers=6, 
                 d_ff=1024, max_seq_len=512, dropout=0.1):
        super().__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        
    def forward(self, x, mask=None):
        return self.encoder(x, mask)

def create_causal_mask(seq_len):
    """Create causal mask for autoregressive generation"""
    mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
    return Tensor(mask, requires_grad=False)

def generate_sample_data(batch_size, seq_len, vocab_size):
    """Generate sample training data"""
    # Random token sequences (simplified)
    data = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    # Convert to one-hot for embedding layer (simplified approach)
    one_hot = np.zeros((batch_size, seq_len, vocab_size))
    for i in range(batch_size):
        for j in range(seq_len):
            one_hot[i, j, data[i, j]] = 1.0
    
    inputs = Tensor(one_hot, requires_grad=False)
    
    # Targets are shifted by one position
    targets = np.roll(data, -1, axis=1)
    targets[:, -1] = 0  # End token
    targets = Tensor(targets, requires_grad=False)
    
    return inputs, targets

def train_transformer():
    """Training loop for the Transformer model"""
    
    # Model parameters
    vocab_size = 1000
    d_model = 256
    num_heads = 8
    num_layers = 4
    d_ff = 512
    seq_len = 64
    batch_size = 8
    learning_rate = 0.0001
    num_epochs = 10
    
    print("Initializing Transformer model")
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=seq_len
    )
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Create causal mask for autoregressive training
    causal_mask = create_causal_mask(seq_len)
    
    print(f"Starting training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 10  # Train on 10 batches per epoch
        
        # Set model to training mode
        model.train()
        
        for batch in range(num_batches):
            # Generate training data
            inputs, targets = generate_sample_data(batch_size, seq_len, vocab_size)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, causal_mask)
            
            # Compute loss (simplified - reshape for cross entropy)
            # In practice, you'd handle this more carefully
            batch_loss = 0.0
            for b in range(batch_size):
                for t in range(seq_len):
                    pred = outputs.data[b, t, :]
                    target = int(targets.data[b, t])
                    
                    # Simple cross entropy calculation
                    pred_tensor = Tensor(pred, requires_grad=True)
                    target_tensor = Tensor(np.eye(vocab_size)[target], requires_grad=False)
                    
                    loss = cross_entropy_loss(pred_tensor, target_tensor)
                    batch_loss += loss.data
            
            batch_loss_tensor = Tensor(np.array(batch_loss / (batch_size * seq_len)), requires_grad=True)
            
            # Backward pass
            batch_loss_tensor.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += batch_loss_tensor.data
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    print("Training complete")
    return model

def generate_text(model, prompt_tokens, max_length=50, vocab_size=1000):
    """Generate text using the trained transformer"""
    # Set model to evaluation mode
    model.eval()
    
    # Convert prompt to tensor
    seq_len = len(prompt_tokens)
    one_hot = np.zeros((1, seq_len, vocab_size))
    for i, token in enumerate(prompt_tokens):
        one_hot[0, i, token] = 1.0
    
    generated = Tensor(one_hot, requires_grad=False)
    
    # Use no_grad context for inference
    with no_grad():
        for _ in range(max_length):
            # Create causal mask for current sequence
            current_len = generated.shape[1]
            mask = create_causal_mask(current_len)
            
            # Forward pass
            outputs = model(generated, mask)
            
            # Get next token probabilities
            next_token_logits = outputs.data[0, -1, :]
            
            # Simple greedy sampling
            next_token = np.argmax(next_token_logits)
            
            # Add next token to sequence
            next_one_hot = np.zeros((1, 1, vocab_size))
            next_one_hot[0, 0, next_token] = 1.0
            next_tensor = Tensor(next_one_hot, requires_grad=False)
            
            # Concatenate (simplified - in practice use proper concat operation)
            new_seq = np.concatenate([generated.data, next_tensor.data], axis=1)
            generated = Tensor(new_seq, requires_grad=False)
    
    # Extract token sequence
    token_sequence = []
    for t in range(generated.shape[1]):
        token = np.argmax(generated.data[0, t, :])
        token_sequence.append(token)
    
    return token_sequence

if __name__ == "__main__":
    # Train the model
    model = train_transformer()
    
    # Generate some text
    print("\nGenerating text")
    prompt = [1, 15, 42]  # Example prompt tokens
    generated_sequence = generate_text(model, prompt, max_length=20)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_sequence}")
    
    print("\nTransformer example completed")
