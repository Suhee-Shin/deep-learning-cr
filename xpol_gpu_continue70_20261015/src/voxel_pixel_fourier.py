"""Exact Fourier material convolution for piecewise-constant rectangular pixels."""
import torch


def pixel_convolution(eps_maps, order_x, order_y):
    """eps_maps [..., Nx, Ny], cells [i/N,(i+1)/N); exp(-i*2*pi*k*x) convention.

    A cell integral multiplies its DFT coefficient by sinc(k/N)*exp(-i*pi*k/N).
    The signed difference order is retained even when looking up a wrapped DFT bin.
    """
    nx, ny = eps_maps.shape[-2:]
    real_dtype = eps_maps.real.dtype
    mx, my = torch.meshgrid(torch.arange(-order_x, order_x + 1, device=eps_maps.device),
                            torch.arange(-order_y, order_y + 1, device=eps_maps.device), indexing="ij")
    mx, my = mx.flatten(), my.flatten()
    dx, dy = mx[:, None] - mx[None, :], my[:, None] - my[None, :]
    hx, hy = dx.to(real_dtype)/nx, dy.to(real_dtype)/ny
    factor = torch.sinc(hx) * torch.sinc(hy) * torch.exp(-1j * torch.pi * (hx + hy))
    spec = torch.fft.fft2(eps_maps) / (nx*ny)
    return spec[..., dx % nx, dy % ny] * factor
