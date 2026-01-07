"""
CODE ADAPTED FROM https://github.com/pumpikano/tf-dann/blob/master/flip_gradient.py
"""

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def flip_gradient(x, alpha):
    """
    Gradient reversal layer - reverses the gradient during backpropagation.

    Args:
        x: input tensor
        alpha: scaling factor for gradient reversal

    Returns:
        output tensor (same as input in forward pass)
    """
    return GradientReversalFunction.apply(x, alpha)

