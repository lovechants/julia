from julia.core.tensor import Tensor, _ensure_tensor
from julia.core.nn.layers import Layer
from typing import Optional, Tuple
import numpy as np 

class RNN(Layer):
    """
    Mutil-Layer Recurrent Neural Network (RNN) module 
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        nonlinearity: The non-linearity to use ('tanh' or 'relu')
        bias: If False, the layer does not use bias weights
        batch_first: If True, input and output tensors are (batch, seq, feature)
        dropout: Dropout probability (0 means no dropout)
        bidirectional: If True, becomes a bidirectional RNN
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 nonlinearity: str = 'tanh', bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create RNN cells for each layer and direction
        self.rnn_cells = []
        
        # Input-to-hidden layer
        for direction in range(self.num_directions):
            cell = RNNCell(input_size, hidden_size, bias, nonlinearity)
            self.rnn_cells.append(cell)
            
        # Hidden-to-hidden layers (if num_layers > 1)
        for layer in range(1, num_layers):
            for direction in range(self.num_directions):
                layer_input_size = hidden_size * self.num_directions if layer > 0 else input_size
                cell = RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
                self.rnn_cells.append(cell)
        
        # Collect parameters from all cells
        self.parameters = []
        for cell in self.rnn_cells:
            self.parameters.extend(cell.parameters)
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for RNN
        
        Args:
            x: Input sequence tensor of shape:
               - (seq_len, batch_size, input_size) if batch_first=False
               - (batch_size, seq_len, input_size) if batch_first=True
            hidden: Initial hidden state tensor of shape:
                   (num_layers * num_directions, batch_size, hidden_size)
                   
        Returns:
            output: Output features for each time step
            h_n: Final hidden state
        """
        x = _ensure_tensor(x)
        
        # Handle batch_first
        if self.batch_first:
            # Convert (batch, seq, feature) to (seq, batch, feature)
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Get dimensions
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            # Create initial hidden state of zeros
            # Shape: (num_layers * num_directions, batch_size, hidden_size)
            hidden_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden = Tensor(np.zeros(hidden_shape), requires_grad=x.requires_grad)
        
        # Process sequence
        outputs = []
        
        # Separate hidden states for each layer and direction
        hidden_states = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                h = hidden[idx]
                hidden_states.append(h)
        
        # Process each time step
        for t in range(seq_len):
            # For storing layer outputs at this time step
            layer_outputs = []
            
            # Input at this time step
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            # Forward direction
            for layer in range(self.num_layers):
                # For bidirectional RNN, create separate forward and backward sequences
                for direction in range(self.num_directions):
                    # Get the right cell and hidden state
                    cell_idx = layer * self.num_directions + direction
                    cell = self.rnn_cells[cell_idx]
                    h = hidden_states[cell_idx]
                    
                    # For backward direction in bidirectional RNN, process sequence in reverse
                    if direction == 1:  # Backward direction
                        x_t = Tensor(x.data[seq_len - 1 - t], requires_grad=x.requires_grad)
                    
                    # For first layer, use input; for deeper layers, use output from previous layer
                    if layer == 0:
                        # Process the input with the RNN cell
                        h_next = cell(x_t, h)
                    else:
                        # Concatenate outputs from forward and backward directions
                        if self.bidirectional and direction == 0:
                            prev_outputs = layer_outputs[layer-1]
                            if isinstance(prev_outputs, list):  # Handle bidirectional outputs
                                combined = Tensor(np.concatenate([
                                    prev_outputs[0].data, 
                                    prev_outputs[1].data
                                ], axis=1), requires_grad=True)
                                h_next = cell(combined, h)
                            else:
                                h_next = cell(prev_outputs, h)
                        else:
                            h_next = cell(layer_outputs[layer-1], h)
                    
                    # Update hidden state
                    hidden_states[cell_idx] = h_next
                    
                    # For output collection
                    if layer == self.num_layers - 1:
                        if self.bidirectional:
                            # For bidirectional, collect output from both directions
                            if direction == 0:  # Forward
                                # Initialize list for this layer
                                layer_outputs.append([h_next])
                            else:  # Backward
                                # Add backward direction output
                                layer_outputs[-1].append(h_next)
                        else:
                            layer_outputs.append(h_next)
                    else:
                        if direction == 0:  # Forward direction
                            # Initialize list for this layer
                            layer_outputs.append(h_next)
                        else:  # Backward direction
                            # For bidirectional, we'll need to concatenate later
                            # Just store both outputs separately for now
                            pass
            
            # Collect output for this time step
            if self.bidirectional:
                # For bidirectional, concatenate forward and backward outputs
                fwd_out = layer_outputs[-1][0].data
                bwd_out = layer_outputs[-1][1].data
                combined_out = np.concatenate([fwd_out, bwd_out], axis=1)
                outputs.append(Tensor(combined_out, requires_grad=True))
            else:
                outputs.append(layer_outputs[-1])
        
        # Stack outputs for all time steps
        if self.bidirectional:
            # For bidirectional, outputs are already combined
            output_data = np.stack([o.data for o in outputs], axis=0)
        else:
            output_data = np.stack([o.data for o in outputs], axis=0)
        
        output = Tensor(output_data, requires_grad=True)
        
        # Stack hidden states for all layers
        h_n_data = np.stack([h.data for h in hidden_states], axis=0)
        h_n = Tensor(h_n_data, requires_grad=True)
        
        # Handle batch_first for output
        if self.batch_first:
            # Convert (seq, batch, feature) to (batch, seq, feature)
            output_data = output.data.transpose(1, 0, 2)
            output = Tensor(output_data, requires_grad=output.requires_grad)
        
        return output, h_n


class GRU(Layer):
    """
    Multi-layer Gated Recurrent Unit (GRU) module

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        bias: If False, the layer does not use bias weights
        batch_first: If True, input and output tensors are (batch, seq, feature)
        dropout: Dropout probability (0 means no dropout)
        bidirectional: If True, becomes a bidirectional GRU
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create GRU cells for each layer and direction
        self.gru_cells = []
        
        # Input-to-hidden layer
        for direction in range(self.num_directions):
            cell = GRUCell(input_size, hidden_size, bias)
            self.gru_cells.append(cell)
            
        # Hidden-to-hidden layers (if num_layers > 1)
        for layer in range(1, num_layers):
            for direction in range(self.num_directions):
                layer_input_size = hidden_size * self.num_directions if layer > 0 else input_size
                cell = GRUCell(layer_input_size, hidden_size, bias)
                self.gru_cells.append(cell)
        
        # Collect parameters from all cells
        self.parameters = []
        for cell in self.gru_cells:
            self.parameters.extend(cell.parameters)
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for GRU
        
        Args:
            x: Input sequence tensor of shape:
               - (seq_len, batch_size, input_size) if batch_first=False
               - (batch_size, seq_len, input_size) if batch_first=True
            hidden: Initial hidden state tensor of shape:
                   (num_layers * num_directions, batch_size, hidden_size)
                   
        Returns:
            output: Output features for each time step
            h_n: Final hidden state
        """
        x = _ensure_tensor(x)
        
        # Handle batch_first
        if self.batch_first:
            # Convert (batch, seq, feature) to (seq, batch, feature)
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Get dimensions
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            # Create initial hidden state of zeros
            # Shape: (num_layers * num_directions, batch_size, hidden_size)
            hidden_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            hidden = Tensor(np.zeros(hidden_shape), requires_grad=x.requires_grad)
        
        # Process sequence
        outputs = []
        
        # Separate hidden states for each layer and direction
        hidden_states = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                h = hidden[idx]
                hidden_states.append(h)
        
        # Process each time step
        for t in range(seq_len):
            # For storing layer outputs at this time step
            layer_outputs = []
            
            # Input at this time step
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            # Process through each layer
            for layer in range(self.num_layers):
                # For bidirectional GRU, process forward and backward directions
                for direction in range(self.num_directions):
                    # Get the right cell and hidden state
                    cell_idx = layer * self.num_directions + direction
                    cell = self.gru_cells[cell_idx]
                    h = hidden_states[cell_idx]
                    
                    # For backward direction in bidirectional GRU, process sequence in reverse
                    if direction == 1:  # Backward direction
                        x_t = Tensor(x.data[seq_len - 1 - t], requires_grad=x.requires_grad)
                    
                    # For first layer, use input; for deeper layers, use output from previous layer
                    if layer == 0:
                        # Process the input with the GRU cell
                        h_next = cell(x_t, h)
                    else:
                        # Concatenate outputs from forward and backward directions
                        if self.bidirectional and direction == 0:
                            prev_outputs = layer_outputs[layer-1]
                            if isinstance(prev_outputs, list):  # Handle bidirectional outputs
                                combined = Tensor(np.concatenate([
                                    prev_outputs[0].data, 
                                    prev_outputs[1].data
                                ], axis=1), requires_grad=True)
                                h_next = cell(combined, h)
                            else:
                                h_next = cell(prev_outputs, h)
                        else:
                            h_next = cell(layer_outputs[layer-1], h)
                    
                    # Update hidden state
                    hidden_states[cell_idx] = h_next
                    
                    # For output collection
                    if layer == self.num_layers - 1:
                        if self.bidirectional:
                            # For bidirectional, collect output from both directions
                            if direction == 0:  # Forward
                                # Initialize list for this layer
                                layer_outputs.append([h_next])
                            else:  # Backward
                                # Add backward direction output
                                layer_outputs[-1].append(h_next)
                        else:
                            layer_outputs.append(h_next)
                    else:
                        if direction == 0:  # Forward direction
                            # Initialize list for this layer
                            layer_outputs.append(h_next)
                        else:  # Backward direction
                            # For bidirectional, we'll need to concatenate later
                            # Just store both outputs separately for now
                            pass
            
            # Collect output for this time step
            if self.bidirectional:
                # For bidirectional, concatenate forward and backward outputs
                fwd_out = layer_outputs[-1][0].data
                bwd_out = layer_outputs[-1][1].data
                combined_out = np.concatenate([fwd_out, bwd_out], axis=1)
                outputs.append(Tensor(combined_out, requires_grad=True))
            else:
                outputs.append(layer_outputs[-1])
        
        # Stack outputs for all time steps
        if self.bidirectional:
            # For bidirectional, outputs are already combined
            output_data = np.stack([o.data for o in outputs], axis=0)
        else:
            output_data = np.stack([o.data for o in outputs], axis=0)
        
        output = Tensor(output_data, requires_grad=True)
        
        # Stack hidden states for all layers
        h_n_data = np.stack([h.data for h in hidden_states], axis=0)
        h_n = Tensor(h_n_data, requires_grad=True)
        
        # Handle batch_first for output
        if self.batch_first:
            # Convert (seq, batch, feature) to (batch, seq, feature)
            output_data = output.data.transpose(1, 0, 2)
            output = Tensor(output_data, requires_grad=output.requires_grad)
        
        return output, h_n


class LSTMCell(Layer):
    """
    Long Short-Term Memory (LSTM) cell implementation
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, the layer does not use bias weights
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights
        k = np.sqrt(1 / hidden_size)
        
        # Input gate weights
        self.weight_ii = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hi = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        # Forget gate weights
        self.weight_if = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hf = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        # Cell gate weights
        self.weight_ig = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hg = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        # Output gate weights
        self.weight_io = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_ho = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        self.parameters = [
            self.weight_ii, self.weight_hi,
            self.weight_if, self.weight_hf,
            self.weight_ig, self.weight_hg,
            self.weight_io, self.weight_ho
        ]
        
        if bias:
            # Input gate bias
            self.bias_ii = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hi = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            # Forget gate bias - initialize to 1.0 to avoid vanishing gradients
            self.bias_if = Tensor(np.ones(hidden_size), requires_grad=True)
            self.bias_hf = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            # Cell gate bias
            self.bias_ig = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hg = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            # Output gate bias
            self.bias_io = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_ho = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            self.parameters.extend([
                self.bias_ii, self.bias_hi,
                self.bias_if, self.bias_hf,
                self.bias_ig, self.bias_hg,
                self.bias_io, self.bias_ho
            ])
        else:
            self.bias_ii = self.bias_hi = None
            self.bias_if = self.bias_hf = None
            self.bias_ig = self.bias_hg = None
            self.bias_io = self.bias_ho = None
    
    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of LSTM cell
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hx: Tuple of (hidden_state, cell_state) each of shape (batch_size, hidden_size)
            
        Returns:
            Tuple of (new_hidden_state, new_cell_state)
        """
        x = _ensure_tensor(x)
        batch_size = x.shape[0]
        
        # Initialize hidden and cell states if not provided
        if hx is None:
            h_0 = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
            c_0 = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
        else:
            h_0, c_0 = hx
        
        # Calculate input gate
        i_t = x.matmul(self.weight_ii.transpose()) + h_0.matmul(self.weight_hi.transpose())
        if self.bias_ii is not None and self.bias_hi is not None:
            i_t = i_t + self.bias_ii + self.bias_hi
        i_t = i_t.sigmoid()
        
        # Calculate forget gate
        f_t = x.matmul(self.weight_if.transpose()) + h_0.matmul(self.weight_hf.transpose())
        if self.bias_if is not None and self.bias_hf is not None:
            f_t = f_t + self.bias_if + self.bias_hf
        f_t = f_t.sigmoid()
        
        # Calculate cell gate
        g_t = x.matmul(self.weight_ig.transpose()) + h_0.matmul(self.weight_hg.transpose())
        if self.bias_ig is not None and self.bias_hg is not None:
            g_t = g_t + self.bias_ig + self.bias_hg
        g_t = g_t.tanh()
        
        # Calculate output gate
        o_t = x.matmul(self.weight_io.transpose()) + h_0.matmul(self.weight_ho.transpose())
        if self.bias_io is not None and self.bias_ho is not None:
            o_t = o_t + self.bias_io + self.bias_ho
        o_t = o_t.sigmoid()
        
        # Update cell state
        c_1 = f_t * c_0 + i_t * g_t
        
        # Update hidden state
        h_1 = o_t * c_1.tanh()
        
        return h_1, c_1

class LSTM(Layer):
    """
    Multi-layer Long Short-Term Memory (LSTM) module
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        bias: If False, the layer does not use bias weights
        batch_first: If True, input and output tensors are (batch, seq, feature)
        dropout: Dropout probability (0 means no dropout)
        bidirectional: If True, becomes a bidirectional LSTM
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 bias: bool = True, batch_first: bool = False,
                 dropout: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create LSTM cells for each layer and direction
        self.lstm_cells = []
        
        # Input-to-hidden layer
        for direction in range(self.num_directions):
            cell = LSTMCell(input_size, hidden_size, bias)
            self.lstm_cells.append(cell)
            
        # Hidden-to-hidden layers (if num_layers > 1)
        for layer in range(1, num_layers):
            for direction in range(self.num_directions):
                layer_input_size = hidden_size * self.num_directions if layer > 0 and self.bidirectional else hidden_size
                cell = LSTMCell(layer_input_size, hidden_size, bias)
                self.lstm_cells.append(cell)
        
        # Collect parameters from all cells
        self.parameters = []
        for cell in self.lstm_cells:
            self.parameters.extend(cell.parameters)
            
    def forward(self, x: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass for LSTM
        
        Args:
            x: Input sequence tensor of shape:
               - (seq_len, batch_size, input_size) if batch_first=False
               - (batch_size, seq_len, input_size) if batch_first=True
            hx: Tuple of (hidden_state, cell_state) each of shape:
                (num_layers * num_directions, batch_size, hidden_size)
                
        Returns:
            output: Output features for each time step
            (h_n, c_n): Final hidden state and cell state
        """
        x = _ensure_tensor(x)
        
        # Handle batch_first
        if self.batch_first:
            # Convert (batch, seq, feature) to (seq, batch, feature)
            x_data = x.data.transpose(1, 0, 2)
            x = Tensor(x_data, requires_grad=x.requires_grad)
        
        # Get dimensions
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden and cell states if not provided
        if hx is None:
            # Create initial hidden and cell states of zeros
            # Shape: (num_layers * num_directions, batch_size, hidden_size)
            h_shape = (self.num_layers * self.num_directions, batch_size, self.hidden_size)
            h_0 = Tensor(np.zeros(h_shape), requires_grad=x.requires_grad)
            c_0 = Tensor(np.zeros(h_shape), requires_grad=x.requires_grad)
        else:
            h_0, c_0 = hx
        
        # Process sequence
        outputs = []
        
        # Separate hidden and cell states for each layer and direction
        hidden_states = []
        cell_states = []
        for layer in range(self.num_layers):
            for direction in range(self.num_directions):
                idx = layer * self.num_directions + direction
                h = h_0[idx]
                c = c_0[idx]
                hidden_states.append(h)
                cell_states.append(c)
        
        # Process each time step
        for t in range(seq_len):
            # For storing layer outputs at this time step
            layer_outputs = []
            
            # Input at this time step
            x_t = Tensor(x.data[t], requires_grad=x.requires_grad)
            
            # Process through each layer
            for layer in range(self.num_layers):
                # For bidirectional LSTM, process forward and backward directions
                for direction in range(self.num_directions):
                    # Get the right cell and states
                    cell_idx = layer * self.num_directions + direction
                    cell = self.lstm_cells[cell_idx]
                    h = hidden_states[cell_idx]
                    c = cell_states[cell_idx]
                    
                    # For backward direction in bidirectional LSTM, process sequence in reverse
                    if direction == 1:  # Backward direction
                        x_t = Tensor(x.data[seq_len - 1 - t], requires_grad=x.requires_grad)
                    
                    # For first layer, use input; for deeper layers, use output from previous layer
                    if layer == 0:
                        # Process with LSTM cell
                        h_next, c_next = cell(x_t, (h, c))
                    else:
                        # For deeper layers, use output from previous layer
                        if self.bidirectional and direction == 0:
                            # Concatenate outputs from forward and backward directions
                            prev_outputs = layer_outputs[layer-1]
                            if isinstance(prev_outputs, list):  # Handle bidirectional outputs
                                combined = Tensor(np.concatenate([
                                    prev_outputs[0].data, 
                                    prev_outputs[1].data
                                ], axis=1), requires_grad=True)
                                h_next, c_next = cell(combined, (h, c))
                            else:
                                h_next, c_next = cell(prev_outputs, (h, c))
                        else:
                            h_next, c_next = cell(layer_outputs[layer-1], (h, c))
                    
                    # Update states
                    hidden_states[cell_idx] = h_next
                    cell_states[cell_idx] = c_next
                    
                    # For output collection
                    if layer == self.num_layers - 1:
                        if self.bidirectional:
                            # For bidirectional, collect output from both directions
                            if direction == 0:  # Forward
                                # Initialize list for this layer
                                layer_outputs.append([h_next])
                            else:  # Backward
                                # Add backward direction output
                                layer_outputs[-1].append(h_next)
                        else:
                            layer_outputs.append(h_next)
                    else:
                        if direction == 0:  # Forward direction
                            # Initialize list for this layer
                            layer_outputs.append(h_next)
                        else:  # Backward direction
                            # For bidirectional, we'll store outputs separately
                            pass
            
            # Collect output for this time step
            if self.bidirectional:
                # For bidirectional, concatenate forward and backward outputs
                fwd_out = layer_outputs[-1][0].data
                bwd_out = layer_outputs[-1][1].data
                combined_out = np.concatenate([fwd_out, bwd_out], axis=1)
                outputs.append(Tensor(combined_out, requires_grad=True))
            else:
                outputs.append(layer_outputs[-1])
        
        # Stack outputs for all time steps
        output_data = np.stack([o.data for o in outputs], axis=0)
        output = Tensor(output_data, requires_grad=True)
        
        # Stack hidden and cell states for all layers
        h_n_data = np.stack([h.data for h in hidden_states], axis=0)
        c_n_data = np.stack([c.data for c in cell_states], axis=0)
        h_n = Tensor(h_n_data, requires_grad=True)
        c_n = Tensor(c_n_data, requires_grad=True)

        if self.batch_first:
                # Convert (seq, batch, feature) to (batch, seq, feature)
                output_data = output.data.transpose(1, 0, 2)
                output = Tensor(output_data, requires_grad=output.requires_grad)
            
        return output, (h_n, c_n)
        


class RNNCell(Layer):
    """
    Basic RNN cell: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, the layer does not use bias weights
        nonlinearity: The non-linearity to use ('tanh' or 'relu')
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True, nonlinearity: str = 'tanh'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        
        # Initialize weights and biases
        k = np.sqrt(1 / hidden_size)
        
        # Input-to-hidden weights
        self.weight_ih = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        # Hidden-to-hidden weights
        self.weight_hh = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        self.parameters = [self.weight_ih, self.weight_hh]
        
        if bias:
            # Input-to-hidden bias
            self.bias_ih = Tensor(np.zeros(hidden_size), requires_grad=True)
            # Hidden-to-hidden bias
            self.bias_hh = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            self.parameters.extend([self.bias_ih, self.bias_hh])
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of RNN cell
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Previous hidden state of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        x = _ensure_tensor(x)
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
        
        # Calculate input-to-hidden contribution
        ih = x.matmul(self.weight_ih.transpose())
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
        
        # Calculate hidden-to-hidden contribution
        hh = hidden.matmul(self.weight_hh.transpose())
        if self.bias_hh is not None:
            hh = hh + self.bias_hh
        
        # Combine contributions and apply nonlinearity
        if self.nonlinearity == 'tanh':
            h_next = (ih + hh).tanh()
        elif self.nonlinearity == 'relu':
            h_next = (ih + hh).relu()
        else:
            raise ValueError(f"Unknown nonlinearity: {self.nonlinearity}")
        
        return h_next


class GRUCell(Layer):
    """
    Gated Recurrent Unit (GRU) cell implementation

    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, the layer does not use bias weights
    """
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for input-to-hidden
        k = np.sqrt(1 / hidden_size)
        
        # Reset gate weights
        self.weight_ir = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hr = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        # Update gate weights
        self.weight_iz = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hz = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        # New gate weights
        self.weight_in = Tensor(np.random.uniform(-k, k, (hidden_size, input_size)), requires_grad=True)
        self.weight_hn = Tensor(np.random.uniform(-k, k, (hidden_size, hidden_size)), requires_grad=True)
        
        self.parameters = [
            self.weight_ir, self.weight_hr, 
            self.weight_iz, self.weight_hz, 
            self.weight_in, self.weight_hn
        ]
        
        if bias:
            # Reset gate biases
            self.bias_ir = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hr = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            # Update gate biases
            self.bias_iz = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hz = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            # New gate biases
            self.bias_in = Tensor(np.zeros(hidden_size), requires_grad=True)
            self.bias_hn = Tensor(np.zeros(hidden_size), requires_grad=True)
            
            self.parameters.extend([
                self.bias_ir, self.bias_hr,
                self.bias_iz, self.bias_hz,
                self.bias_in, self.bias_hn
            ])
        else:
            self.bias_ir = self.bias_hr = None
            self.bias_iz = self.bias_hz = None
            self.bias_in = self.bias_hn = None
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of GRU cell
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            hidden: Previous hidden state of shape (batch_size, hidden_size)
            
        Returns:
            New hidden state
        """
        x = _ensure_tensor(x)
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = Tensor(np.zeros((batch_size, self.hidden_size)), requires_grad=x.requires_grad)
        
        # Calculate reset gate
        r_t = x.matmul(self.weight_ir.transpose()) + hidden.matmul(self.weight_hr.transpose())
        if self.bias_ir is not None and self.bias_hr is not None:
            r_t = r_t + self.bias_ir + self.bias_hr
        r_t = r_t.sigmoid()
        
        # Calculate update gate
        z_t = x.matmul(self.weight_iz.transpose()) + hidden.matmul(self.weight_hz.transpose())
        if self.bias_iz is not None and self.bias_hz is not None:
            z_t = z_t + self.bias_iz + self.bias_hz
        z_t = z_t.sigmoid()
        
        # Calculate candidate hidden state
        n_t = x.matmul(self.weight_in.transpose()) + (r_t * hidden).matmul(self.weight_hn.transpose())
        if self.bias_in is not None and self.bias_hn is not None:
            n_t = n_t + self.bias_in + self.bias_hn
        n_t = n_t.tanh()
        
        # Calculate new hidden state
        h_next = (1 - z_t) * n_t + z_t * hidden
        
        return h_next
