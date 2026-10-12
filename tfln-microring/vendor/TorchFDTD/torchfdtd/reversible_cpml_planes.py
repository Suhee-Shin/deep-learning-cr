"""Fixed spectral planes over recorded-interface CPML differentiation."""
import torch

from .adjoint_planes import DifferentiablePlaneSimulation
from .reversible_cpml import ReversibleCPMLOptions, ReversibleCPMLSimulation


class ReversibleCPMLPlaneSimulation(DifferentiablePlaneSimulation):
    """Collocated six-component spectra with an explicit fixed exterior.

    Construction builds fixed host quadrature/interpolation maps before plan().
    Their allowance is included in subsequent execution and planning admission.
    Frequencies, source settings and monitor layouts are fixed. The supplied
    material maps may depend on differentiable geometry or density parameters.
    """

    _resident_model_type = ReversibleCPMLSimulation

    def __init__(self, project, options=None, *, quadrature_counts=None):
        if options is not None and not isinstance(options, ReversibleCPMLOptions):
            raise ValueError('Recorded CPML planes require ReversibleCPMLOptions.')
        super().__init__(project, options, quadrature_counts=quadrature_counts)
        # Fixed Yee observations outlive individual solves. Keep only host
        # index tables here, with no fields, cotangents or device allocations.
        # Each solve uploads its own maps on its current stream, so changing
        # devices/streams cannot reuse stale CUDA pointers or retain VRAM.
        self._observation_cache = {}

    @property
    def interior_z(self):
        return self.model.interior_z

    def _spectral(self, epsilon, frequency_hz, block_size):
        spectral = super()._spectral(epsilon, frequency_hz, block_size)
        spectral.layout_reservation_bytes = self.layout_reservation_bytes
        spectral.observation_cache = self._observation_cache
        return spectral

    def forward(self, epsilon, frequency_hz, *, fixed_epsilon, block_size=32):
        """Return monitor-ID plane results without full-time plane histories."""
        return self._planes(epsilon, frequency_hz, block_size,
            lambda spectral: self.model._run(epsilon, spectral,
                                               fixed_epsilon=fixed_epsilon))

    def plan(self, frequency_hz, *, device='cpu', material_components=1,
             block_size=32):
        """Admit solver and existing host layout, without allocating fields.

        material_components=1 admits scalar maps, 3 admits diagonal Yee maps.
        Both epsilon inputs must match that shape at execution.
        """
        if (self.project.model_dump() != self._project_snapshot
                or self.model.project.model_dump() != self._internal_snapshot):
            raise ValueError('Plane configuration changed. Rebuild the model to regenerate fixed interpolation and source plans.')
        project, interval = self.model._snapshot()
        mock = torch.empty((), dtype=torch.float32, device='cpu')
        spectral = self._spectral(mock, frequency_hz, block_size)
        from .reversible_cpml_memory import _cpml_reversible_reservation
        return _cpml_reversible_reservation(project, self.model.options,
            torch.device(device), interval, material_components=material_components,
            spectral=spectral)
