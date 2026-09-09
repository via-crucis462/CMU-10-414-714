"""The module.
"""
from importlib.metadata import requires
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from functools import reduce


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        _weight = init.kaiming_uniform(in_features , out_features)
        self.weight = Parameter(_weight, dtype = "float32")
        if bias is True :
            _bias = init.kaiming_uniform(out_features , 1).reshape((1,out_features))
            self.bias = Parameter(_bias, dtype = "float32")
        else :
            self.bias = None

        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X @ self.weight
        if self.bias is None:
            return out
        return out + self.bias.reshape((1, self.out_features)).broadcast_to(out.shape)
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flattened_size = reduce(lambda left, right: left * right, X.shape[1:], 1)
        return X.reshape((batch_size, flattened_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for i in self.modules :
            out = i(out)
        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, num_classes = logits.shape

        labels = init.one_hot(num_classes, y)
        correct_logits = ops.summation(
            logits * labels,
            axes=(1,),
        )

        loss = ops.logsumexp(logits, axes=(1,)) - correct_logits
        return ops.summation(loss) / batch_size
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype)
        self.running_var = init.ones(dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = ops.summation(x, axes=(0,)) / x.shape[0]
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            centered = x - mean_broadcast
            variance = ops.summation(centered ** 2, axes=(0,)) / x.shape[0]

            self.running_mean.data = (
                (1 - self.momentum) * self.running_mean
                + self.momentum * mean
            )
            self.running_var.data = (
                (1 - self.momentum) * self.running_var
                + self.momentum * variance
            )
        else:
            mean = self.running_mean
            variance = self.running_var
            mean_broadcast = mean.reshape((1, self.dim)).broadcast_to(x.shape)
            centered = x - mean_broadcast

        scale = (variance + self.eps) ** 0.5
        scale = scale.reshape((1, self.dim)).broadcast_to(x.shape)
        normalized = centered / scale
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return normalized * weight + bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)) / self.dim
        mean = mean.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        centered = x - mean
        variance = ops.summation(centered ** 2, axes=(1,)) / self.dim
        scale = (variance.reshape((x.shape[0], 1)) + self.eps) ** 0.5
        scale = scale.broadcast_to(x.shape)
        normalized = centered / scale
        weight = self.weight.reshape((1, self.dim)).broadcast_to(x.shape)
        bias = self.bias.reshape((1, self.dim)).broadcast_to(x.shape)
        return normalized * weight + bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if not self.training:
            return x
        mask = init.randb(
            *x.shape,
            p=1 - self.p,
            device=x.device,
            dtype=x.dtype,
        )
        return x * mask / (1 - self.p)
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



