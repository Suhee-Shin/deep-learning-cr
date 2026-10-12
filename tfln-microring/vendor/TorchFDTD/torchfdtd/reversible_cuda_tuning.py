"""Instance-local launch selection for the recorded CPML CUDA path.

Timing runs on copies of writable arguments, never on simulation state. The
cache contains only launch metadata, not tensors or CUDA allocation owners.
"""
from __future__ import annotations

import hashlib
import re
import statistics
import threading
from collections import OrderedDict

import torch

from .cuda_kernels import _compile, _direct_cuda_view


BLOCKS = (128, 256, 512, 1024)
_CACHE = OrderedDict()
_LOCK = threading.RLock()


def select_launch(source, name, tensors, arrays, count, device, requested,
                  *, default=256, label, report):
    """Compile a fixed launch, or measure interleaved candidates on scratch.

    A failed optional specialization falls back to the original source and
    block size. Runtime launch errors propagate, as the CUDA context may no
    longer be usable. Allocation failures also propagate.
    """
    import cupy

    source = re.sub(r'__launch_bounds__\(\d+\)\s*', '', source)
    capability = cupy.cuda.Device(device).compute_capability
    digest = hashlib.sha256(source.encode()).hexdigest()
    key = (device, capability, digest, name, count)
    record = dict(source_sha256=digest, requested=requested, cached=False)

    def compiled(block):
        head = f'void {name}('
        if source.count(head) != 1:
            raise ValueError('Expected exactly one CUDA kernel declaration.')
        bounded = source.replace(head, f'void __launch_bounds__({block}) {name}(')
        return _compile(bounded, device, capability, name)

    stream = torch.cuda.current_stream(device)
    with _LOCK, cupy.cuda.Device(device), cupy.cuda.ExternalStream(stream.cuda_stream, device_id=device):
        if requested == 'auto':
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('CUDA launch tuning must precede graph capture.')
            if key in _CACHE:
                record.update(_CACHE[key], cached=True)
                _CACHE.move_to_end(key)
                block = record['block_size']
                try:
                    fn, module = compiled(block)
                except RuntimeError as exc:
                    if not str(exc).startswith('CUDA kernel compilation failed:'):
                        raise
                    # The compiled-module cache may evict an entry before the
                    # metadata cache. Its next compilation can then fail.
                    del _CACHE[key]
                    record.pop('median_ms', None)
                    record.update(cache_invalidated=True, fallback_reason=str(exc),
                                  compile_errors={str(block): str(exc)})
                    block = default
                    fn, module = _compile(source, device, capability, name)
                    record['block_size'] = block
            else:
                head = source.split(f'{name}(', 1)[1].split(')', 1)[0]
                parameters = head.split(',')
                if len(parameters) != len(arrays) or len(tensors) != len(arrays):
                    raise ValueError('CUDA tuning argument metadata does not match the signature.')
                scratch = {}
                arguments = []
                for parameter, tensor, array in zip(parameters, tensors, arrays):
                    if '*' in parameter and not parameter.strip().startswith('const '):
                        # Preserve aliases if a kernel declares the same writable
                        # storage more than once. Coefficients stay read-only.
                        identity = (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()))
                        if identity not in scratch:
                            scratch[identity] = tensor.clone()
                        arguments.append(_direct_cuda_view(cupy, scratch[identity]))
                    else:
                        arguments.append(array)
                timings, candidates, errors = {}, {}, {}
                for block in BLOCKS:
                    try:
                        candidates[block] = compiled(block)
                    except RuntimeError as exc:
                        if not str(exc).startswith('CUDA kernel compilation failed:'):
                            raise
                        errors[str(block)] = str(exc)
                for block, (fn, _) in candidates.items():
                    fn(((count + block - 1) // block,), (block,), tuple(arguments))
                    timings[block] = []
                for sweep in range(5):
                    order = list(candidates) if sweep % 2 == 0 else list(reversed(candidates))
                    for block in order:
                        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        start.record(stream)
                        candidates[block][0](((count + block - 1) // block,), (block,), tuple(arguments))
                        stop.record(stream)
                        stop.synchronize()
                        timings[block].append(start.elapsed_time(stop))
                del arguments, scratch
                if timings:
                    medians = {block: statistics.median(values) for block, values in timings.items()}
                    block = min(medians, key=medians.get)
                    fn, module = candidates[block]
                    record.update(block_size=block, median_ms=medians, compile_errors=errors)
                    _CACHE[key] = dict(record)
                    while len(_CACHE) > 128:
                        _CACHE.popitem(last=False)
                else:
                    block = default
                    fn, module = _compile(source, device, capability, name)
                    record.update(block_size=block, fallback_reason='No block-size specialization compiled.', compile_errors=errors)
        else:
            block = default if requested is None else requested
            try:
                fn, module = compiled(block)
            except RuntimeError as exc:
                if not str(exc).startswith('CUDA kernel compilation failed:'):
                    raise
                block = default
                fn, module = _compile(source, device, capability, name)
                record['fallback_reason'] = str(exc)
            record['block_size'] = block
    if report is not None:
        report.setdefault('cuda_launches', {})[label] = record
        if record.get('fallback_reason'):
            report.setdefault('fallback_reason', {})['block_size:'+label] = record['fallback_reason']
    return fn, module, block


def tune_yee(kernel, requested, report):
    """Install instance-local launches without replacing classes or functions."""
    kernel.tuned_blocks = {}
    for forward in (False, True):
        _, arrays, _ = kernel.launches[forward]
        source, tensors = kernel._source(forward)
        count = kernel.launch_count(kernel.grid, forward)
        fn, module, block = select_launch(source, 'yee_update', tensors, arrays,
            count, kernel.device, requested, label='yee_H' if forward else 'yee_E', report=report)
        kernel.launches[forward] = fn, arrays, module
        kernel.tuned_blocks[forward] = block


def tune_adjoint(kernel, requested, report):
    kernel.tuned_blocks = {}
    for (forward, phase), (_, arrays, _) in kernel.launches.items():
        source, tensors = kernel.source(forward, phase)
        fn, module, block = select_launch(source, 'adjoint_update', tensors, arrays,
            kernel.count, kernel.device, requested,
            label=f'adjoint_{"H" if forward else "E"}_{phase}', report=report)
        kernel.launches[forward, phase] = fn, arrays, module
        kernel.tuned_blocks[forward, phase] = block
