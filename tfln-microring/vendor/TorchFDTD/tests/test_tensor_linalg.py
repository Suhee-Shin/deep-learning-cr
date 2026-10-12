import torch
from torchfdtd.tensor_linalg import inverse_3x3, symmetric_eigenvalue_bounds


def test_chunked_inverse_values_gradients_and_spectral_bounds():
    generator = torch.Generator().manual_seed(12)
    raw = torch.randn(5, 7, 3, 3, dtype=torch.float64, generator=generator)
    value = (raw @ raw.transpose(-1, -2) + torch.eye(3)).requires_grad_()
    reference = torch.linalg.inv(value)
    result = inverse_3x3(value, batch_size=4)
    torch.testing.assert_close(result, reference)
    expected_gradient, = torch.autograd.grad(reference.square().sum(), value)
    gradient, = torch.autograd.grad(result.square().sum(), value)
    torch.testing.assert_close(gradient, expected_gradient)
    low, high = symmetric_eigenvalue_bounds(value, batch_size=4)
    eigenvalues = torch.linalg.eigvalsh(value)
    torch.testing.assert_close(low, eigenvalues.min())
    torch.testing.assert_close(high, eigenvalues.max())
