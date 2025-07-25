import weakref
import numpy as np
import uuid
import inspect

"""
Core Tensor class (with autograd)
i miss title fight
"""


class Context:
    """
    Fixed Context class with proper memory management
    """

    def __init__(self):
        self.saved_tensors = ()
        self.saved_data = {}
        self._tensor_refs = []  # Keep strong references

    def save_for_backwards(self, *tensors):
        self.saved_tensors = tensors
        # Keep strong references to prevent garbage collection
        self._tensor_refs = [t for t in tensors if isinstance(t, Tensor)]

    def save_data(self, **kwargs):
        self.saved_data.update(kwargs)

    def __del__(self):
        try:
            self._tensor_refs.clear()
            self.saved_tensors = ()
            self.saved_data.clear()
        except Exception:
            pass


class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @classmethod
    def apply(cls, *args, **kwargs):
        ctx = Context()
        # Forward pass produces the result tensor(s)
        result = cls.forward(
            ctx, *args, **kwargs
        )  # Assume forward returns Tensor or tuple of Tensors

        # Determine if any input requires grad
        op_requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad for arg in args
        )

        if op_requires_grad:
            # Create the backward node using the original inputs
            backward_node = BackwardNode(cls, ctx, args)

            # Assign the node AND set requires_grad flag on the output tensor(s)
            if isinstance(result, Tensor):
                result._backward_node = backward_node
                result.requires_grad = True  # <--- Explicitly set the flag
                result._is_leaf = False
                result._grad_fn = cls.__name__
            elif isinstance(result, tuple):
                # multiple output
                processed_result = []
                for i, r in enumerate(result):
                    if isinstance(r, Tensor):
                        r._backward_node = backward_node
                        r.requires_grad = True
                        r._is_leaf = False
                        r._grad_fn = cls.__name__
                        r._output_index = i
                        processed_result.append(r)
                    else:
                        processed_result.append(r)
        # If op_requires_grad is False, result tensors keep their original flag
        return result

    def __init_subclass__(cls, **kwargs):
        """auto register on class definitions"""
        super().__init_subclass__(**kwargs)

        if (
            hasattr(cls, "forward")
            and hasattr(cls, "backward")
            and cls.__name__ != "Function"
        ):
            try:
                from julia.core.utils.op_registry import registry

                op_name = getattr(cls, "_op_nane", cls.__name__)
                registry.register(op_name, cls)

                # For onnx and shape inference import/export
                if hasattr(cls, "infer_shape"):
                    registry.register_shape_inference(op_name, cls.infer_shape)

            except ImportError:
                pass


class BackwardNode:
    def __init__(self, fn_cls, ctx, inputs):
        self.fn_cls = fn_cls
        self.ctx = ctx
        self.inputs = inputs
        self.input_refs = []
        self.next_functions = []
        self.grad_outputs = {}
        self.grad_output_count = 0
        self.num_expect_grads = None
        for inp in inputs:
            if isinstance(inp, Tensor) and inp.requires_grad and inp._backward_node:
                self.next_functions.append(inp._backward_node)
                self.input_refs.append(weakref.ref(inp))
            else:
                self.input_refs.append(None)

    def accmulate_grad(self, grad, idx=0):
        """
        Accumlate gradient for a specfic output tensor
        Args:
            frad: the gradient to accmulate
            idx: the index of the output tensor
        """
        if idx in self.grad_outputs:
            self.grad_outputs[idx] = self.grad_outputs[idx] + grad
        else:
            self.grad_outputs[idx] = grad
            self.grad_output_count += 1

        if (
            self.num_expect_grads is not None
            and self.grad_output_count >= self.num_expect_grads
        ):
            return True
        return False

    def cleanup(self):
        """MEMORY LEAK FIX: Explicitly break reference cycles"""
        self.inputs.clear()
        self.input_refs.clear()
        self.ctx = None
        self.fn_cls = None


class Tensor:
    def __init__(self, data, requires_grad=False, device=None, dtype=None):
        self.id = str(uuid.uuid4())
        if isinstance(data, Tensor):
            self.data = data.data
            self.requires_grad = requires_grad
            self.grad = None
            self._backward_node = None
        elif isinstance(data, np.ndarray):
            self.data = data
            self.requires_grad = requires_grad
            self.grad = None
            self._backward_node = None
        else:
            self.data = np.array(data, dtype=dtype)
            self.requires_grad = requires_grad
            self.grad = None
            self._backward_node = None

        self.device = device or "cpu"
        try:
            from julia.core.memory import try_allocate_raw_backed_array

            self.data, self._raw_ptr, self._raw_size = try_allocate_raw_backed_array(
                self.data, self.device
            )
        except:
            self._raw_ptr, self._raw_size = None, 0

        self.shape = self.data.shape

        # For hooks
        self._is_leaf = True  # user & parameter created tensors
        self._retain_grad = False
        self._grad_fn = None
        self._backward_hooks = {}
        self._next_hook_id = 0
        # ^^^ Think about tensor bindings to be added for the compiled autograd engine

    def __getitem__(self, key):
        """Enable indexing for Tensor objects"""
        return Tensor(self.data[key], requires_grad=self.requires_grad)

    def __setitem__(self, key, value):
        """Enable item assignment for Tensor objects"""
        if isinstance(value, Tensor):
            self.data[key] = value.data
        else:
            self.data[key] = value

    def __del__(self):
        try:
            if hasattr(self, "_raw_ptr") and self._raw_ptr is not None:
                from julia.core.memory import cleanup_raw_memory

                cleanup_raw_memory(self._raw_ptr, self._raw_size)
                self._raw_ptr = None
                self._raw_size = 0
        except Exception:
            # Ignore cleanup errors during destruction
            pass

    def _cleanup_autograd(self):
        if self._backward_node is not None:
            # Clean up the backward node
            self._backward_node.cleanup()
            self._backward_node = None

        # Clear gradient
        if self.grad is not None:
            if hasattr(self.grad, "_cleanup_autograd"):
                self.grad._cleanup_autograd()
            self.grad = None

    def zero_grad(self):
        self.grad = None

    def dropout(self, p: float, training: bool):
        from julia.core.ops import Dropout

        return Dropout.apply(self, p, training)

    # Tensor Detach stuff

    def detach(self):
        """
        Return a new tensor detached from computation graph
        Same data but does not require gradients

        Returns:
            Tensor: A new Tensor with the same data but no grad history
        """

        return Tensor(self.data.copy(), requires_grad=False, device=self.device)

    def detach_(self):
        """
        Inplace version
        Clears gradient and backward_node

        Returns:
            self: The same Tensor but it is detached from the computation graph
        """

        self.requires_grad = False
        self._backward_node = None
        self.grad = None

        return self

    def clone(self):
        """
        Clone -> returns a new tensor with the same data and grad requirements
        Tracks gradients independely from the original tensor it cloned

        Returns:
             Tensor: new tensor with a the same data and grad requirements
        """

        return Tensor(
            self.data.copy(), requires_grad=self.requires_grad, device=self.device
        )

    def retain_grad(self):
        """
        Gradient retention for non-leaf Tensors
        Default behavior: leaf Tensors created by the user or model params retain their gradients after computing the backward gradient.
        This forces a Tensor to retain their gradient
        """
        self._retain_grad = True
        return self

    def backward(self, gradient=None, retain_graph=False, create_graph=False):
        """
        Enhanced backward pass with comprehensive profiling
        """
        if not self.requires_grad:
            return

        from julia.core.profiler import profiler
        import time

        backward_start_time = time.perf_counter()

        with profiler.profile_operation("autograd_backward_full", "Backward"):
            with profiler.profile_operation("build_computation_graph", "Backward"):
                visited_nodes = set()
                topo_order_nodes = []

                def build_topo(node):
                    if node and node not in visited_nodes:
                        visited_nodes.add(node)
                        for inp in node.inputs:
                            if (
                                isinstance(inp, Tensor)
                                and inp.requires_grad
                                and inp._backward_node
                            ):
                                build_topo(inp._backward_node)
                        topo_order_nodes.append(node)

                if self._backward_node:
                    build_topo(self._backward_node)

            with profiler.profile_operation("initialize_gradients", "Backward"):
                if gradient is None:
                    if self.shape == () or self.shape == (1,):
                        gradient_data = np.ones_like(self.data)
                    else:
                        raise ValueError(
                            "Must specify gradient for non-scalar tensors used as roots"
                        )
                elif isinstance(gradient, Tensor):
                    gradient_data = gradient.data.copy()
                else:
                    gradient_data = np.array(gradient).copy()

                grads_tensor_keyed = {self: gradient_data}

                if self.grad is None:
                    self.grad = Tensor(gradient_data.copy(), requires_grad=create_graph)
                else:
                    if self.grad.requires_grad != create_graph:
                        self.grad = Tensor(
                            self.grad.data + gradient_data, requires_grad=create_graph
                        )
                    else:
                        self.grad.data += gradient_data

            total_backward_ops = len(topo_order_nodes)

            with profiler.profile_operation(
                f"backward_propagation_{total_backward_ops}_ops", "Backward"
            ):
                for i, node in enumerate(reversed(topo_order_nodes)):
                    output_tensor = None
                    output_grad_data = None
                    for t, grad_d in grads_tensor_keyed.items():
                        if hasattr(t, "_backward_node") and t._backward_node is node:
                            output_tensor = t
                            output_grad_data = grad_d
                            break

                    if output_tensor is None or output_grad_data is None:
                        continue

                    # Profile individual backward operation
                    op_name = f"backward_{node.fn_cls.__name__}"
                    input_info = f"step_{i+1}_of_{total_backward_ops}"

                    with profiler.profile_operation(
                        f"{op_name}_{input_info}", "Backward"
                    ):
                        grad_out_tensor = Tensor(
                            output_grad_data, requires_grad=create_graph
                        )

                        if (
                            hasattr(output_tensor, "_backward_hooks")
                            and output_tensor._backward_hooks
                        ):
                            with profiler.profile_operation(
                                f"backward_hooks_{node.fn_cls.__name__}", "Backward"
                            ):
                                for hook in output_tensor._backward_hooks.values():
                                    hook_result = hook(grad_out_tensor)
                                    if hook_result is not None:
                                        if not isinstance(hook_result, Tensor):
                                            raise TypeError(
                                                f"Hook returned invalid type: {type(hook_result)}"
                                            )
                                        grad_out_tensor = hook_result

                        backward_func_start = time.perf_counter()
                        grads_in = node.fn_cls.backward(node.ctx, grad_out_tensor)
                        backward_func_time = time.perf_counter() - backward_func_start

                        if (
                            hasattr(profiler, "operation_history")
                            and profiler.operation_history
                        ):
                            last_op = profiler.operation_history[-1]
                            last_op.custom_metrics[
                                "backward_func_time"
                            ] = backward_func_time

                        if not isinstance(grads_in, tuple):
                            grads_in = (grads_in,)

                        forward_sig = inspect.signature(node.fn_cls.forward)
                        expected_grads = (
                            len(forward_sig.parameters) - 1
                        )  # Exclude 'ctx' parameter

                        if expected_grads != len(grads_in):
                            raise RuntimeError(
                                f"Backwards function {node.fn_cls.__name__} returned {len(grads_in)} gradients, forward function expects {expected_grads} gradients (excluding ctx)"
                            )

                        # Distribute gradients with profiling
                        with profiler.profile_operation(
                            f"distribute_grads_{node.fn_cls.__name__}", "Backward"
                        ):
                            for inp, grads_in_tensor in zip(node.inputs, grads_in):
                                if isinstance(inp, Tensor) and inp.requires_grad:
                                    if grads_in_tensor is not None:
                                        grad_in_data = grads_in_tensor.data

                                        if inp not in grads_tensor_keyed:
                                            grads_tensor_keyed[
                                                inp
                                            ] = grad_in_data.copy()
                                        else:
                                            grads_tensor_keyed[inp] += grad_in_data

                                        should_retain_grad = inp._is_leaf or hasattr(
                                            inp, "_retain_grad"
                                        )

                                        if should_retain_grad:
                                            if inp.grad is None:
                                                inp.grad = Tensor(
                                                    grad_in_data.copy(),
                                                    requires_grad=create_graph,
                                                )
                                            else:
                                                if (
                                                    inp.grad.requires_grad
                                                    != create_graph
                                                ):
                                                    inp.grad = Tensor(
                                                        inp.grad.data + grad_in_data,
                                                        requires_grad=create_graph,
                                                    )
                                                else:
                                                    inp.grad.data += grad_in_data

            # Clean up computation graph if not retaining
            with profiler.profile_operation("cleanup_computation_graph", "Backward"):
                if not retain_graph:
                    self._backward_node = None

        total_backward_time = time.perf_counter() - backward_start_time
        profiler.total_backward_time += total_backward_time

        if hasattr(profiler, "operation_history") and profiler.operation_history:
            # Find the most recent autograd_backward_full operation and update its metrics
            for op in reversed(profiler.operation_history):
                if op.op_name == "autograd_backward_full":
                    op.custom_metrics["total_backward_time"] = total_backward_time
                    op.custom_metrics["num_backward_ops"] = total_backward_ops
                    op.custom_metrics["avg_time_per_op"] = total_backward_time / max(
                        1, total_backward_ops
                    )
                    break

    # def backward(self, gradient=None, retain_graph=False, create_graph=False):
    #     """
    #     Compute gradient of tensor w.r.t
    #     Args:
    #         gradient: Gradient of the current tensor (default=None, treated as 1.0 (implied))
    #         retain_graph: If True, computation graph is kept for future backward calls (default=False)
    #         create_graph: If True, graph of the derivative is constructed for higher order derivatives
    #     """
    #
    #     if not self.requires_grad:
    #         return
    #
    #     visited_nodes = set()
    #     topo_order_nodes = []
    #
    #     def build_topo(node):
    #         if node and node not in visited_nodes:
    #             visited_nodes.add(node)
    #             for inp in node.inputs:
    #                 if isinstance(inp, Tensor) and inp.requires_grad and inp._backward_node:
    #                     build_topo(inp._backward_node)
    #             topo_order_nodes.append(node)
    #
    #
    #     if self._backward_node:
    #         build_topo(self._backward_node)
    #
    #
    #     # Init Gradient
    #     if gradient is None:
    #         if self.shape == () or self.shape == (1,):
    #             gradient_data = np.ones_like(self.data)
    #         else:
    #             raise ValueError("Must specify gradient for non-scalr tensors used as roots")
    #
    #     elif isinstance(gradient, Tensor):
    #         gradient_data = gradient.data.copy()
    #     else:
    #         gradient_data = np.array(gradient).copy()
    #
    #     grads_tensor_keyed = {self: gradient_data}
    #
    #     # Assign init gradient to self.grad
    #     if self.grad is None:
    #         self.grad = Tensor(gradient_data.copy(), requires_grad=create_graph)
    #     else:
    #         if self.grad.requires_grad != create_graph:
    #             self.grad = Tensor(self.grad.data + gradient_data, requires_grad=create_graph)
    #         else:
    #             self.grad.data += gradient_data
    #
    #     # Backprop
    #     for node in reversed(topo_order_nodes):
    #         output_tensor = None
    #         output_grad_data = None
    #         for t, grad_d in grads_tensor_keyed.items():
    #             if hasattr(t, '_backward_node') and t._backward_node is node:
    #                 output_tensor = t
    #                 output_grad_data = grad_d
    #                 break
    #
    #         if output_tensor is None or output_grad_data is None:
    #             continue
    #
    #         grad_out_tensor = Tensor(output_grad_data, requires_grad=create_graph)
    #
    #         # Apply hooks
    #         if hasattr(output_tensor, '_backward_hooks') and output_tensor._backward_hooks:
    #             for hook in output_tensor._backward_hooks.values():
    #                 hook_result = hook(grad_out_tensor)
    #                 if hook_result is not None:
    #                     if not isinstance(hook_result, Tensor):
    #                         raise TypeError(f"Hook returned invalid type: {type(hook_result)}")
    #                     grad_out_tensor = hook_result
    #
    #         grads_in = node.fn_cls.backward(node.ctx, grad_out_tensor)
    #
    #         if not isinstance(grads_in, tuple):
    #             grads_in = (grads_in, )
    #
    #         # Distribute gradients
    #         if len(node.inputs) != len(grads_in):
    #             raise RuntimeError(f"Backwards function {node.fn_cls.__name__} returned {len(grads_in)} "
    #                                f"gradients, forward took {len(node.inputs)} inputs")
    #
    #         for inp, grads_in_tensor in zip(node.inputs, grads_in):
    #             if isinstance(inp, Tensor) and inp.requires_grad:
    #                 if grads_in_tensor is not None:
    #                     grad_in_data = grads_in_tensor.data
    #
    #                     if inp not in grads_tensor_keyed:
    #                         grads_tensor_keyed[inp] = grad_in_data.copy()
    #                     else:
    #                         grads_tensor_keyed[inp] += grad_in_data
    #
    #                     should_retain_grad = inp._is_leaf or hasattr(inp, '_retain_grad')
    #
    #                     if should_retain_grad:
    #                         if inp.grad is None:
    #                             inp.grad = Tensor(grad_in_data.copy(), requires_grad=create_graph)
    #                         else:
    #                             if inp.grad.requires_grad != create_graph:
    #                                 inp.grad = Tensor(inp.grad.data + grad_in_data, requires_grad=create_graph)
    #                             else:
    #                                 inp.grad.data += grad_in_data
    #
    #     if not retain_graph:
    #         # Clear backward node for the output tensor
    #         self._backward_node = None

    # Operator overloads

    def __add__(self, other):
        from julia.core.ops import Add

        return Add.apply(self, _ensure_tensor(other))

    def __mul__(self, other):
        from julia.core.ops import Mul

        return Mul.apply(self, _ensure_tensor(other))

    def __sub__(self, other):
        from julia.core.ops import Sub

        return Sub.apply(self, _ensure_tensor(other))

    def __truediv__(self, other):
        from julia.core.ops import Div

        return Div.apply(self, _ensure_tensor(other))

    def matmul(self, other):
        from julia.core.ops import MatMul

        return MatMul.apply(self, _ensure_tensor(other))

    def relu(self):
        from julia.core.ops import ReLU

        return ReLU.apply(self)

    def sigmoid(self):
        from julia.core.ops import Sigmoid

        return Sigmoid.apply(self)

    def reshape(self, *shape):
        from julia.core.ops import Reshape

        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return Reshape.apply(self, shape)

    def transpose(self):
        from julia.core.ops import Transpose

        return Transpose.apply(self)

    def sum(self):
        from julia.core.ops import Sum

        return Sum.apply(self)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            power = other
        else:
            power = _ensure_tensor(other)
        from julia.core.ops import Pow

        return Pow.apply(self, power)

    def __rpow__(self, other):
        if isinstance(other, (int, float)):
            base = Tensor(np.array(other, dtype=np.float32))
        else:
            base = _ensure_tensor(other)
        from julia.core.ops import Pow

        return Pow.apply(base, self)

    def __rmul__(self, other):
        from julia.core.ops import Mul

        return Mul.apply(_ensure_tensor(other), self)

    def __radd__(self, other):
        from julia.core.ops import Add

        return Add.apply(_ensure_tensor(other), self)

    def __rsub__(self, other):
        from julia.core.ops import Sub

        return Sub.apply(_ensure_tensor(other), self)

    def __rtruediv__(self, other):
        from julia.core.ops import Div

        return Div.apply(_ensure_tensor(other), self)

    # Tensor dimension functions
    def flatten(self, start_dim=0, end_dim=-1):
        """
        Flatten tensor dimensions

        Args:
            start_dim: First dimension to flatten (inclusive)
            end_dim: Last dimension to flatten (inclusive)

        Returns:
            Flattened tensor
        """
        if end_dim < 0:
            end_dim = len(self.shape) + end_dim

        new_shape = list(self.shape)

        if start_dim == end_dim:
            return self

        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= new_shape[i]

        final_shape = new_shape[:start_dim] + [flat_size] + new_shape[end_dim + 1 :]

        return self.reshape(final_shape)

    def squeeze(self, dim=None):
        """
        Remove single-dimensional entries from tensor shape

        Args:
            dim: Dimension to squeeze. If None, squeeze all dimensions of size 1

        Returns:
            Squeezed tensor
        """
        if dim is None:
            # Remove all dimensions of size 1
            new_shape = [s for s in self.shape if s != 1]
        else:
            # Remove specific dimension if it has size 1
            if dim < 0:
                dim = len(self.shape) + dim

            if self.shape[dim] != 1:
                raise ValueError(
                    f"Cannot squeeze dimension {dim} with size {self.shape[dim]}"
                )

            new_shape = list(self.shape)
            new_shape.pop(dim)

        if not new_shape:  # If all dimensions were squeezed, result is scalar
            new_shape = []

        return self.reshape(new_shape)

    def unsqueeze(self, dim):
        """
        Add a dimension of size 1 at the specified position

        Args:
            dim: Position to insert the new dimension

        Returns:
            Tensor with added dimension
        """
        new_shape = list(self.shape)

        if dim < 0:
            dim = len(new_shape) + dim + 1

        new_shape.insert(dim, 1)
        return self.reshape(new_shape)


def _ensure_tensor(data):
    if isinstance(data, Tensor):
        return data
    if isinstance(data, (int, float)):
        tensor = object.__new__(Tensor)
        tensor.id = str(uuid.uuid4())
        tensor.data = np.array(data, dtype=np.float32)
        tensor.requires_grad = False
        tensor.grad = None
        tensor._backward_node = None
        tensor.device = "cpu"
        tensor._raw_ptr, tensor._raw_size = None, 0
        tensor.shape = tensor.data.shape
        tensor._is_leaf = True
        tensor._retain_grad = False
        tensor._grad_fn = None
        tensor._backward_hooks = {}
        tensor._next_hook_id = 0
        return tensor

    return Tensor(data)
