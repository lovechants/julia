"""
RNN Example using Julia Framework
Implementation of LSTM, GRU, and vanilla RNN for sequence modeling tasks
"""

import numpy as np
from julia.core.tensor import Tensor
from julia.core.nn.layers import Layer, Linear, Sequential, Dropout
from julia.core.nn.recurrent import LSTM, GRU, RNN
from julia.core.loss import mse_loss, cross_entropy_loss
from julia.core.optim import Adam

class SequencePredictor(Layer):
    """RNN-based sequence prediction model"""
    
    def __init__(self, input_size, hidden_size, output_size, rnn_type='LSTM', 
                 num_layers=2, dropout=0.2, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        
        # Input projection
        self.input_projection = Linear(input_size, hidden_size)
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif rnn_type == 'GRU':
            self.rnn = GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:  # Vanilla RNN
            self.rnn = RNN(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=bidirectional,
                batch_first=True
            )
        
        # Output projection
        rnn_output_size = hidden_size * (2 if bidirectional else 1)
        self.output_projection = Sequential(
            Linear(rnn_output_size, hidden_size),
            Dropout(dropout),
            Linear(hidden_size, output_size)
        )
        
    def forward(self, x, hidden=None):
        # Project input
        x = self.input_projection(x)
        
        # RNN forward pass
        if self.rnn_type == 'LSTM':
            rnn_output, (hidden, cell) = self.rnn(x, hidden)
            return self.output_projection(rnn_output), (hidden, cell)
        else:
            rnn_output, hidden = self.rnn(x, hidden)
            return self.output_projection(rnn_output), hidden

class TimeSeriesRNN(Layer):
    
    def __init__(self, input_features, hidden_size=64, num_layers=2, 
                 output_features=1, rnn_type='LSTM'):
        super().__init__()
        
        self.predictor = SequencePredictor(
            input_size=input_features,
            hidden_size=hidden_size,
            output_size=output_features,
            rnn_type=rnn_type,
            num_layers=num_layers
        )
        
    def forward(self, x):
        output, _ = self.predictor(x)
        return output

class TextClassifierRNN(Layer):
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=64, 
                 num_classes=2, num_layers=2, rnn_type='LSTM'):
        super().__init__()
        
        # Embedding layer (simplified as linear layer)
        self.embedding = Linear(vocab_size, embedding_dim)
        
        # RNN
        self.rnn_predictor = SequencePredictor(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            output_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers
        )
        
        # Classification head
        self.classifier = Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Embed tokens
        embedded = self.embedding(x)
        
        # RNN forward
        rnn_output, _ = self.rnn_predictor(embedded)
        
        # Use last timestep for classification
        last_output = rnn_output[:, -1, :]  # (batch_size, hidden_size)
        
        # Classify
        return self.classifier(last_output)

def generate_sine_wave_data(batch_size, seq_len, num_features=1):
    data = []
    targets = []
    
    for _ in range(batch_size):
        # Random frequency and phase
        freq = np.random.uniform(0.1, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        
        # Generate sequence
        t = np.linspace(0, 4*np.pi, seq_len + 1)
        sequence = np.sin(freq * t + phase)
        
        # Add noise
        sequence += np.random.normal(0, 0.1, len(sequence))
        
        # Input is first seq_len points, target is last point
        if num_features == 1:
            input_seq = sequence[:-1].reshape(-1, 1)
            target = sequence[-1]
        else:
            # Multi-feature: include cosine and derivative
            cos_seq = np.cos(freq * t[:-1] + phase)
            deriv_seq = freq * np.cos(freq * t[:-1] + phase)
            input_seq = np.stack([sequence[:-1], cos_seq, deriv_seq], axis=1)
            target = sequence[-1]
        
        data.append(input_seq)
        targets.append(target)
    
    return Tensor(np.array(data)), Tensor(np.array(targets))

def generate_text_classification_data(batch_size, seq_len, vocab_size, num_classes=2):
    data = []
    labels = []
    
    for _ in range(batch_size):
        # Generate random sequence
        sequence = np.random.randint(0, vocab_size, seq_len)
        
        # Convert to one-hot
        one_hot = np.zeros((seq_len, vocab_size))
        for i, token in enumerate(sequence):
            one_hot[i, token] = 1.0
        
        # Simple rule for labeling: positive if sum of token indices > threshold
        label = 1 if np.sum(sequence) > (vocab_size * seq_len * 0.5) else 0
        
        data.append(one_hot)
        labels.append(label)
    
    return Tensor(np.array(data)), Tensor(np.array(labels))

def train_time_series_model():
    print("Training Time Series RNN")
    
    # Model parameters
    input_features = 3  # sine, cosine, derivative
    hidden_size = 32
    num_layers = 2
    seq_len = 20
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 50
    
    # Create model
    model = TimeSeriesRNN(
        input_features=input_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        rnn_type='LSTM'
    )
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 20
        
        for batch in range(num_batches):
            # Generate training data
            inputs, targets = generate_sine_wave_data(batch_size, seq_len, input_features)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(inputs)
            
            # Get last timestep prediction
            last_pred = predictions[:, -1, 0]  # (batch_size,)
            
            # Compute loss
            loss = mse_loss(last_pred, targets)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.data
        
        avg_loss = epoch_loss / num_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, MSE Loss: {avg_loss:.6f}")
    
    print("Time series training completed!")
    return model

def train_text_classifier():
    print("\nTraining Text Classification RNN")
    
    # Model parameters
    vocab_size = 100
    embedding_dim = 32
    hidden_size = 64
    num_classes = 2
    seq_len = 15
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 30
    
    # Create model
    model = TextClassifierRNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_classes=num_classes,
        rnn_type='GRU'
    )
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    print(f"Training for {num_epochs} epochs")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        num_batches = 25
        
        for batch in range(num_batches):
            # Generate training data
            inputs, labels = generate_text_classification_data(
                batch_size, seq_len, vocab_size, num_classes
            )
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(inputs)
            
            # Convert labels to one-hot
            labels_one_hot = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                labels_one_hot[i, int(labels.data[i])] = 1.0
            labels_one_hot = Tensor(labels_one_hot)
            
            # Compute loss
            loss = cross_entropy_loss(logits, labels_one_hot)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            epoch_loss += loss.data
            
            # Calculate accuracy
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == labels.data)
            total += batch_size
        
        avg_loss = epoch_loss / num_batches
        accuracy = correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.3f}")
    
    print("Text classification training completed!")
    return model

def test_sequence_generation():
    print("\nTesting Sequence Generation")
    
    # Simple character-level model
    vocab_size = 26  # a-z
    hidden_size = 32
    seq_len = 10
    
    model = SequencePredictor(
        input_size=vocab_size,
        hidden_size=hidden_size,
        output_size=vocab_size,
        rnn_type='LSTM',
        num_layers=1
    )
    
    # Generate a sequence
    # Start with 'a' (index 0)
    current_char = 0
    sequence = [current_char]
    
    # Initialize hidden state
    hidden = None
    
    for _ in range(seq_len - 1):
        # One-hot encode current character
        input_vec = np.zeros((1, 1, vocab_size))
        input_vec[0, 0, current_char] = 1.0
        input_tensor = Tensor(input_vec)
        
        # Forward pass
        output, hidden = model(input_tensor, hidden)
        
        # Get next character (greedy)
        next_char_probs = output.data[0, 0, :]
        next_char = np.argmax(next_char_probs)
        
        sequence.append(next_char)
        current_char = next_char
    
    # Convert to characters
    char_sequence = ''.join([chr(ord('a') + i) for i in sequence])
    print(f"Generated sequence: {char_sequence}")

def compare_rnn_types():
    rnn_types = ['RNN', 'GRU', 'LSTM']
    seq_len = 15
    batch_size = 16
    hidden_size = 32
    num_epochs = 20
    
    for rnn_type in rnn_types:
        print(f"\nTraining {rnn_type}")
        
        model = TimeSeriesRNN(
            input_features=1,
            hidden_size=hidden_size,
            rnn_type=rnn_type
        )
        
        optimizer = Adam(model.parameters(), lr=0.001)
        
        final_loss = 0.0
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for _ in range(10):
                inputs, targets = generate_sine_wave_data(batch_size, seq_len, 1)
                
                optimizer.zero_grad()
                predictions = model(inputs)
                last_pred = predictions[:, -1, 0]
                loss = mse_loss(last_pred, targets)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.data
            
            final_loss = epoch_loss / 10
        
        print(f"{rnn_type} final loss: {final_loss:.6f}")

if __name__ == "__main__":
    ts_model = train_time_series_model()
    text_model = train_text_classifier()
    test_sequence_generation()
    compare_rnn_types()
    
