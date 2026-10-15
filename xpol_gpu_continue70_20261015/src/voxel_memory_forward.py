"""Memory scheduling only: identical RCWA equations, no wavelength/layer reduction."""
import torch
from torch.utils.checkpoint import checkpoint
from contextlib import contextmanager


class EigenReplay:
    """One checkpoint invocation only; never a warm start across design updates."""

    def __init__(self):
        self.values = {}

    @contextmanager
    def activate(self, owner, replay):
        if getattr(owner, "_eigen_replay", None) is not None:
            raise RuntimeError("Nested eigen replay is not supported")
        owner._eigen_replay = (self, replay)
        try:
            yield
        finally:
            owner._eigen_replay = None
            if replay:
                self.values.clear()


class EigenReplayMixin:
    def _chunk_eig(self, e_all, einv_all, rows):
        active = getattr(self, "_eigen_replay", None)
        if active is None:
            return super()._chunk_eig(e_all, einv_all, rows)
        bank, replay = active
        # The owner's geometry cache returns the same immutable row object on replay.
        key = (id(rows), tuple(e_all.shape), e_all.dtype, e_all.device)
        stats = getattr(self, "eigen_replay_stats", None)
        if stats is None:
            stats = self.eigen_replay_stats = dict(recorded=0, replayed=0, stored_bytes=0)
        if replay:
            if key not in bank.values:
                raise RuntimeError("Eigen replay geometry mismatch; refusing an unvalidated fallback")
            vals, vecs = bank.values[key]
            stats["replayed"] += 1
            return vals.to(e_all.device), vecs.to(e_all.device)
        vals, vecs = super()._chunk_eig(e_all, einv_all, rows)
        bank.values[key] = tuple(t.detach().to(device="cpu", copy=True) for t in (vals, vecs))
        stats["recorded"] += 1
        stats["stored_bytes"] += sum(t.numel()*t.element_size() for t in (vals, vecs))
        return vals, vecs


class LayerCheckpointMixin:
    def _finish(self, ps, valss, xs, g):
        if not getattr(self, "checkpoint_layers", False):
            return super()._finish(ps, valss, xs, g)
        eye = torch.eye(2*self.n_orders, dtype=xs.dtype, device=xs.device).expand_as(xs[:, 0])

        def first(p, vals, x):
            t, r = self._layer_s(p, vals, x, g, self.layer_h[0])
            sol = torch.linalg.solve(eye-g.sin12.left(r), torch.cat((g.sin11.left(g.e_in), g.sin12.left(t)), -1))
            return t @ sol[..., :2], r + t @ sol[..., 2:]

        def run(fn, *tensors):
            return checkpoint(fn, *tensors, use_reentrant=False) if torch.is_grad_enabled() else fn(*tensors)

        s1v, s12 = run(first, ps[:, 0], valss[:, 0], xs[:, 0])
        for li in range(1, self.n_layers):
            def layer(s1v, s12, p, vals, x, height=self.layer_h[li]):
                t, r = self._layer_s(p, vals, x, g, height)
                sol = torch.linalg.solve(eye-s12 @ r, torch.cat((s1v, s12 @ t), -1))
                return t @ sol[..., :2], r + t @ sol[..., 2:]
            s1v, s12 = run(layer, s1v, s12, ps[:, li], valss[:, li], xs[:, li])
        return g.sout11.left(torch.linalg.solve(eye-g.sout21.right(s12), s1v))


def spectral_recompute(children, densities, tags, grad, evaluate):
    outputs = []
    for d, tag in enumerate(tags):
        spectra = []
        want = torch.is_grad_enabled() and (grad is None or grad[d])
        for child in children:
            bank = EigenReplay() if want and getattr(child, "cache_eigenpairs", False) else None

            def one(density, child=child, tag=tag, bank=bank):
                progress = getattr(child, "progress_callback", None)
                replay = torch.is_grad_enabled() and density.requires_grad
                phase = "gradient_recompute" if replay else "forward"
                if progress:
                    progress("start", tag, phase, child.wavelengths_nm)
                if bank is None:
                    value = evaluate(child, density, tags=(tag,), grad=None)
                else:
                    with bank.activate(child, replay):
                        value = evaluate(child, density, tags=(tag,), grad=None)
                if progress:
                    progress("end", tag, phase, child.wavelengths_nm)
                return value
            if want:
                value = checkpoint(one, densities[d:d+1], use_reentrant=True)
            else:
                with torch.no_grad():
                    value = one(densities[d:d+1])
            spectra.append(value)
        outputs.append(torch.cat(spectra, dim=2))
    return torch.cat(outputs, dim=0)
