import torch
from torch.cuda.amp import custom_bwd, custom_fwd


class DifferentiableClamp(torch.autograd.Function):
    """
    In the forward pass this operation behaves like torch.clamp.
    But in the backward pass its gradient is 1 everywhere, as if instead of clamp one had used the identity function.
    """

    @staticmethod
    @custom_fwd
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def differential_clamp(inputs, min_val, max_val):
    """
    Like torch.clamp, but with a constant 1-gradient.
    :param inputs: The input that is to be clamped.
    :param min_val: The minimum value of the output.
    :param max_val: The maximum value of the output.
    """
    return DifferentiableClamp.apply(inputs, min_val, max_val)
