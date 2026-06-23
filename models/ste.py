import torch


class BinarizeFunction(torch.autograd.Function):
    """Straight-Through Estimator: forward binarizes, backward passes gradient unchanged."""

    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


binarize = BinarizeFunction.apply
