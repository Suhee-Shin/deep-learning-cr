"""Optional shared CUDA interpolation and online six-component plane DFT.

Torch owns the state. Three launches evaluate double-precision spectral phases,
sample all selected planes and accumulate their complex spectra.
No history of volume fields, time-by-frequency table, or atomic sum is needed.
"""
from __future__ import annotations

import numpy as np
import torch

from .cuda_kernels import _compile


class FusedFrequencyPlanes:
    def __init__(self, monitors, *, phase_kernel='cuda'):
        if phase_kernel not in ('torch','cuda'):
            raise ValueError('phase_kernel must be torch or cuda.')
        self.phase_kernel=phase_kernel
        self.monitors = list(monitors)
        if any(not m.torch or not m.grid.E.is_cuda or m.grid.E.is_complex()
               for m in self.monitors):
            raise ValueError('Fused frequency monitors require real CUDA fields. Use cuda_monitor_kernel="torch" for CPU or Bloch fields.')
        import cupy
        self.cp = cupy
        first = self.monitors[0]
        self.device = first.grid.E.device.index
        dtype, device = first.grid.E.dtype, first.grid.E.device
        if any(m.grid.E.dtype != dtype or m.grid.E.device != device or m.value.dtype!=first.value.dtype for m in self.monitors):
            raise ValueError('Fused frequency monitors must share a CUDA device and precision.')
        self.phases = []
        self.owners = []
        groups = {}
        pointers, sizes = [], []
        for m in self.monitors:
            key = (m.frequency.tobytes(), m.grid.time_step)
            if key not in groups:
                phase = torch.empty_like(m.omega)
                ephase = torch.empty_like(m.half_phase)
                hphase = torch.empty_like(ephase)
                group = (m.omega, m.half_phase, phase, ephase, hphase)
                groups[key] = group
                self.phases.append(group)
            _, _, _, ephase, hphase = groups[key]
            count = m.value.shape[1]
            nc=len(m.components)
            sampled = torch.empty((count, nc), device=device, dtype=dtype)
            padding=[torch.empty(0,device=device)]*(6-nc)
            arrays = [m.grid.E, m.grid.H, *[x[0] for x in m.maps],*padding,
                      *[x[1] for x in m.maps],*padding, m.value, *m.windows,
                      ephase, hphase, sampled]
            self.owners.extend(arrays)
            pointers.append([a.data_ptr() for a in arrays])
            sizes.append((count,len(m.frequency),m.maps[0][0].shape[0],nc,
                          sum(c.startswith('E') for c in m.components),m.monitor.time_downsample))
        self.max_samples = max(n*nc for n, _, _,nc,_,_ in sizes)
        self.max_values = max(n*f*nc for n, f, _,nc,_,_ in sizes)
        table = torch.tensor(pointers, device=device, dtype=torch.int64)
        metadata = torch.tensor(sizes, device=device, dtype=torch.int64)
        # One thread per frequency in each distinct frequency/dt group. Planes
        # share these buffers, with a single writer and ordered later readers.
        phase_table=torch.tensor([[omega.data_ptr(),half.data_ptr(),e.data_ptr(),h.data_ptr(),omega.numel()]
            for omega,half,_,e,h in self.phases],device=device,dtype=torch.int64)
        self.max_frequencies=max(group[0].numel() for group in self.phases)
        self.owners.extend((table, metadata,phase_table))
        source = r'''
typedef REAL real;
typedef SPECTRAL spectral;
extern "C" __global__ void spectral_phases(const long long* groups,
                                          const long long* counter) {
    const long long* row=groups+5*blockIdx.y;
    long long f=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=row[4]) return;
    const double* omega=(const double*)row[0];
    const spectral* half=(const spectral*)row[1];
    spectral* e=(spectral*)row[2];
    spectral* h=(spectral*)row[3];
    double angle=omega[2*f+1]*(double)(counter[0]+1);
    double si,co;
    sincos(angle,&si,&co);
    spectral er=(spectral)co, ei=(spectral)si;
    e[2*f]=er; e[2*f+1]=ei;
    h[2*f]=er*half[2*f]-ei*half[2*f+1];
    h[2*f+1]=er*half[2*f+1]+ei*half[2*f];
}
extern "C" __global__ void sample_planes(const long long* pointers,
                                       const long long* sizes, const long long* counter) {
    int m=blockIdx.y;
    const long long* row=pointers+20*m;
    long long n=sizes[6*m],k=sizes[6*m+2],nc=sizes[6*m+3],ne=sizes[6*m+4];
    if(counter[0]%sizes[6*m+5]) return;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc*n) return;
    int c=i%nc;
    long long p=i/nc;
    const real* field=(const real*)row[c<ne ? 0 : 1];
    const long long* indices=(const long long*)row[2+c];
    const real* weights=(const real*)row[8+c];
    real value=0;
    for(int j=0;j<k;j++) value+=field[indices[j*n+p]]*weights[j*n+p];
    const real* window=(const real*)row[c<ne ? 15 : 16];
    real* sampled=(real*)row[19];
    sampled[i]=value*window[counter[0]];
}
extern "C" __global__ void accumulate_planes(const long long* pointers,
                                           const long long* sizes,const long long* counter) {
    int m=blockIdx.y;
    const long long* row=pointers+20*m;
    long long n=sizes[6*m],nf=sizes[6*m+1],nc=sizes[6*m+3],ne=sizes[6*m+4];
    if(counter[0]%sizes[6*m+5]) return;
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nc*n*nf) return;
    long long f=i/(nc*n), sample=i%(nc*n);
    int c=i%nc;
    const spectral* phase=(const spectral*)row[c<ne ? 17 : 18];
    const real* sampled=(const real*)row[19];
    real value=sampled[sample];
    spectral* output=(spectral*)row[14];
    output[2*i]+=phase[2*f]*value;
    output[2*i+1]+=phase[2*f+1]*value;
}
'''.replace('REAL', 'double' if dtype == torch.float64 else 'float').replace('SPECTRAL','double' if first.value.dtype==torch.complex128 else 'float')
        with cupy.cuda.Device(self.device), self._stream():
            self.sample_kernel, self.module = _compile(source, self.device,
                cupy.cuda.Device(self.device).compute_capability, 'sample_planes')
            self.accumulate_kernel = self.module.get_function('accumulate_planes')
            self.phase_function = self.module.get_function('spectral_phases')
            self.table, self.metadata = (cupy.from_dlpack(a) for a in (table, metadata))
            self.phase_table=cupy.from_dlpack(phase_table)
        self.counter = None

    def _stream(self):
        return self.cp.cuda.ExternalStream(torch.cuda.current_stream(self.device).cuda_stream,
                                          device_id=self.device)

    def update(self, counter):
        # Retain the previous expression for independent ablation/validation.
        # The CUDA path computes phases from absolute time, never recurrence.
        if self.phase_kernel=='torch':
            for omega, half, phase, ephase, hphase in self.phases:
                torch.exp(omega*(counter+1), out=phase)
                ephase.copy_(phase)
                torch.mul(ephase, half, out=hphase)
        with self.cp.cuda.Device(self.device), self._stream():
            if self.counter is None:
                self.counter = counter
                self.counter_view = self.cp.from_dlpack(counter)
            elif counter is not self.counter:
                raise ValueError('A fused monitor group must retain its time counter.')
            if self.phase_kernel=='cuda':
                self.phase_function(((self.max_frequencies+127)//128,len(self.phases)),(128,),
                                    (self.phase_table,self.counter_view))
            self.sample_kernel(((self.max_samples+255)//256, len(self.monitors)), (256,),
                               (self.table, self.metadata, self.counter_view))
            self.accumulate_kernel(((self.max_values+255)//256, len(self.monitors)), (256,),
                                   (self.table, self.metadata,self.counter_view))
