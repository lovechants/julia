import uuid

"""
Autograd Graph 
"""


class Node:
    def __init__(self, op_name, ctx, inputs):
        self.id = str(uuid.uuid4())
        self.op_name = op_name
        self.ctx = ctx
        self.inputs = inputs
        self.next_nodes = []
        for input in inputs:
            if isinstance(input, Tensor) and input.requires_grad and input._node:
                self.next_nodes.append(input._node)


class Graph:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
