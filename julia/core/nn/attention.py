import numpy as np 
from julia.core.tensor import Tensor, Function, _ensure_tensor 
from julia.core.layers import Layer, Linear
from typing import Optional, Tuple

"""
Attentions: 
    MLA 
    MHA 
    GQA
    Flash 
"""
class MultiHeadAttention(Layer):
    """
    Classic attention is all you need mechanism implementation 
    """
    pass 

class MultiHeadLatentAttention(Layer):
    pass 

class GroupQueryAttention(Layer):
    pass 

class FlashAttention(Layer):
    pass 

class SelfAttention(Layer):
    """
    Simple self-attention 
    """

    pass 

class PostionalEncoding(Layer):
    pass 

# TODO -> BeamSearch, Position wise feed forward
# Flip a coin on encode + decode layers 


