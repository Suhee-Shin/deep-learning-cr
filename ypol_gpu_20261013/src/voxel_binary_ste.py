"""Exact hard forward with an explicitly approximate projection/filter backward."""
import torch


class _HardForward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, hard):
        return hard.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient, None


def binary_forward_ste(projected, filtered, threshold=.5):
    hard = (filtered.detach() >= threshold).to(projected.dtype)
    return _HardForward.apply(projected, hard), hard


def gray_penalty_weight(step, maximum, start, ramp):
    return maximum * min(1., max(0., (step-start)/max(1, ramp)))
