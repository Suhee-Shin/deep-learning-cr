# Static voltage bias and LiNbO3 resonance shifts

The Python electro-optic API couples a DC field to the existing experimental
full-tensor FDTD solver. It does not drive the optical source with a DC voltage.
Each optical run sees the frozen material tensor at one applied bias:

1. Solve `div(epsilon_static grad(phi)) = 0` and obtain `E_dc = -grad(phi)`,
   or import a field from an external electrostatic solver.
2. In crystal coordinates form `eta = inv(epsilon_optical_0) + r:E_dc`.
3. Invert eta and rotate the full optical tensor into lab coordinates.
4. Run optical FDTD for each voltage and track the same resonance.

This workflow follows the electrical/optical separation described in
[Ansys's LN example](https://optics.ansys.com/hc/en-us/articles/19435937674387-Thin-Film-Lithium-Niobate-Electro-Optic-Phase-Modulator).
It is not an RF traveling-wave, drift-diffusion, thermal, piezoelectric,
photorefractive or dynamic modulation model. Geometry stays fixed. Optical
dispersion and metal loss are not supplied by this EO wrapper.

## Units and crystal convention

Geometry/coordinate arrays: **um**. Voltage: **V**. Electric field: **V/m**.
Pockels coefficients: **m/V**, so 30.8 pm/V is `30.8e-12`.
Static dielectric constants are independent of optical n squared.

`LithiumNiobate` defaults are illustrative near-1550-nm values, not a calibrated
material database. Override n_o, n_e, r13, r22, r33, r51 for the actual sample,
temperature, wavelength, and clamped/unclamped frequency regime. The independent
coefficients and their frequency dependence are discussed in
[Chelladurai et al., Nature Materials (2025)](https://www.nature.com/articles/s41563-025-02158-1).

Crystal z is the optic/c axis. The contracted order is xx,yy,zz,yz,xz,xy,
without an engineering shear factor of two. This module chooses
`r_xxy=-r22`, `r_yyy=+r22`, `r_xyx=+r22`, `r_xzx=r_yzy=r51`.
Coordinate/domain conventions must match the supplied coefficients. Rotation
columns are the crystal x,y,z unit vectors expressed in lab coordinates.
Identity means crystal z is lab z; do not rotate the optic axis tangentially
around a single-crystal ring.

## Build a field

```python
import numpy as np
from torchfdtd import solve_electrostatic, parallel_plate_field, resample_dc_field

# Uniform parallel-plate approximation, field directed from + plate to ground.
field_per_volt = parallel_plate_field(1., gap_um=2., direction=(0, 0, 1))

# Example DC node grid, axes x,y,z. Unknown potentials are NaN.
electrodes = np.full((21, 9, 9), np.nan)
electrodes[0], electrodes[-1] = 1., 0.
dc = solve_electrostatic(28., electrodes, spacing_um=(.1, .1, .1))
np.savez('dc_1volt.npz', potential_v=dc.potential_v, field_v_m=dc.field_v_m)
```

The DC solver uses harmonic face permittivities and insulating outer boundaries
except where a voltage is fixed. It accepts scalar or spatial isotropic epsilon,
or grid+(ndim,) **axis-aligned diagonal** static epsilon. A rotated off-diagonal
static tensor requires an external solver; no diagonal approximation is made.
Electrodes are ideal fixed-potential regions, not optical metal geometry.

For nonuniform rectilinear external coordinates use
`resample_dc_field((x,y,z), field, tuple(np.asarray(n[:-1]) for n in region.mesh_nodes))`.
Arrays must span the optical grid; extrapolation is rejected. Components stay
in lab xyz order. Divide a field computed at V0 by V0 to obtain field per volt
only for a linear dielectric problem with all driven voltages scaling together.

## Voltage sweep on a prepared optical project

```python
from torchfdtd import ElectroOpticSimulation, LithiumNiobate

# project: supported 3D Yee tensor scene with sources/point monitors.
# base_epsilon: real tensor of shape region.shape + (3,3), background and substrate.
# ln_mask: Boolean array of shape region.shape, including all LN regions.
# field_per_volt: (3,) uniform vector or region.shape+(3,) spatial field.
eo = ElectroOpticSimulation(project, base_epsilon, ln_mask, field_per_volt,
                            material=LithiumNiobate(), rotation=rotation,
                            cpml_material='tensor')
result = eo.sweep([0., 1., 2., 3.], wavelength_um, monitor_index=0, window=window)
result.save('voltage_scan', target_um=1.55, half_window_um=.002, kind='peak')
```

These inputs are prepared from the actual device, not implied defaults. The
tensor solver enforces its own CPML admissibility criterion: general rotated
anisotropic materials touching a PML face can be rejected. A fixed isotropic
CPML exterior is an alternative only when the device really fits inside it.
See [tensor solver scope](TENSOR_NATIVE.md).

`sweep` runs one voltage at a time without retaining autograd graphs and returns
**local monitor |DFT(field)| squared**, not calibrated bus transmission. Use a
post-excitation Hann window and `kind='peak'` for ringdown. Through-port dips and
drop-port peaks require appropriate reference normalization/port analysis; a
point-field trace alone does not measure modal transmitted power. For a voltage
gradient use `eo(voltage_tensor)` directly and form a differentiable objective.

Saved outputs:

- `voltage_spectra.npz`: raw spectra, wavelength grid and voltages.
- `voltage_sweep.json`: reference voltage, dt, duration and nominal frequency bin.
- `resonance_shift.csv`: voltage, resonance in nm, shift in pm relative to the first voltage.

Tracking requires an interior extremum, starts near the requested wavelength,
and follows the nearest extremum at the next voltage. Set the tracking window
below half an FSR and above the expected step shift. Avoid mode crossings or
inspect mode identity separately. A missing resonance raises an error while
preserving raw spectra. Quadratic interpolation is not proof of pm resolution:
validate grid, observation duration, spectral sampling, and bias-step convergence.

## Requested add-drop geometry

`add_drop_ring_structures(thickness_um=..., bus_length_um=...)` constructs:

- ring centreline radius **20 um**, width **1.4 um**;
- two straight buses with **0.1 um edge gaps** at CP1 and CP2;
- default bus width **1.4 um**, an assumption that can be overridden.

Thickness is mandatory. This factory assumes a full-etch ring in the xy plane.
No slab, crystal cut, substrate, electrode dimensions, voltage range, TE/TM
excitation or optical wavelength is inferred from the radius and width alone.
These remaining parameters are required before claiming an actual device shift.
The 100 nm coupling gap needs resolved mesh and convergence checks. A uniform
3D simulation of a 40 um diameter ring can be large; check GPU memory before
allocating the field arrays. Six float32 fields are only the minimum storage;
tensor coefficients, CPML, work buffers and precision add substantial memory.

The separate runnable `examples/linbo3_microring_voltage.py` is a small
z-invariant, locally excited ringdown demonstration with ideal uniform bias.
It is **not** the 20 um fabricated add-drop device and contains no metal/substrate.
Use `--help` for parameters. It writes numerical data, not a validated device result.

## Verification

### Source PDF verification: arXiv 2603.12934v3

The supplied 40-page PDF was read directly, including rendered pages 5,6,8,24,25.
The source hash and page/equation references are stored in the device's
`paper_interpretation.json`. In particular:

- Fig.4 is a **single-ring** baseline with parallel buses at 90/270 degrees.
  Fig.7 is a separate illustrative **two-ring cascade**, with a slanted shared
  bus and labelled pitch 50 um. The earlier factory remains usable as baseline
  geometry but is not an exact Fig.7 model.
- Fig.7 explicitly labels Ring 1 **V inverted**. With the same fixed crystal
  axes and inner-S/outer-G electrodes, use opposite drive signs on the two
  semicircles relative to a common control command. This relative wiring does
  not determine the absolute Bias/Control terminal-voltage mapping.
- Fig.7 labels the gap 3.5 um, whereas Table III and S4.2 say 5 um. Preserve the
  disagreement and retain the user's main-text value 5 um, not a guessed compromise.
- The FDTD band in Sec.IV.B/Table S6 is **1530–1570 nm**, not the earlier user
  range 1520–1580 nm. Both are recorded separately. Table S6 also gives 50 ps,
  auto shutoff 1e-6, conformal variant 1/accuracy 3, and grid 532x816x44 for
  Lumerical. These settings do not establish TorchFDTD mesh/time convergence.
- Table III/S4 use r33=30.9 pm/V, Gamma_EO=0.7, d_eff=2.5 um and f_EO=1/pi.
  For that coefficient use `LithiumNiobate(r33=30.9e-12)` explicitly; the generic
  defaults are illustrative. Other tensor coefficients require their own source.
  The 28.48 pm/V straight-section magnitude and approximately 9.1 pm/V ring
  magnitude are **first-order EO estimates with FDTD-calibrated Q/ng**. They
  are not evidence of a full electrode-plus-optical voltage sweep. When the
  spatial optical/DC field overlap and angular tensor dependence are resolved,
  do not apply the scalar overlap and fill factors again.
- Sec.II defines Delta omega = omega_L-omega_0 and b=-eta/Gamma; reversing
  a and b together leaves the transmission unchanged. Tables IV/S7 provide
  V_bias=abs(a)/b_V and V_ctrl=b*L/b_V as magnitudes, without an explicit signed
  terminal sum or voltage increment. The conditional examples remain conditional.
- N=30 fit coefficients differ slightly between Table IV and S7; both are
  preserved instead of silently replacing one set with the other.

For Ring 1 use the 135-degree arc with relative drive sign -1; Ring 2 uses
315 degrees with relative sign +1. Apply that sign by multiplying the effective
field map (or the common command) **once**, not both. Keep the local S-minus-G
voltage definition distinct from the common command. The signed-projection test
checks both arcs at their midpoint; no FDTD resonance shift is inferred from it.

### Specified X-cut rib device (2026-10-09)

The supplied finite-thickness device can be prepared without allocating an FDTD grid:

```powershell
python examples/prepare_linbo3_rib_device.py --out build/ln_rib_device
```

This writes a **design specification**, not a directly runnable native Project.
After the Fig. 7 clarification, `structures` is intentionally empty; the earlier
parallel-bus geometry is preserved under `provisional_structures_not_fig7`.
It must not be used as the paper's ring/bus layout.
`rib_add_drop_ring_structures(footprint_um=(46,46), bus_length_um=46)` supplies
the substrate, continuous slab and three raised rib structures:

| Region | Lab z range (um) |
|---|---|
| SiO2 lower layer | -2 to 0 |
| Continuous LN slab, including ring hole and coupling gaps | 0 to 0.1 |
| LN ring and bus ridges | 0.1 to 0.6 |

The 46 um footprint and bus length are preparation defaults, not measured device
dimensions or converged boundary locations. Bus width 1.4 um and vertical sidewalls
are explicitly recorded assumptions. The upper cladding is confirmed as air;
material below the silica remains unresolved. Requested optical band is 1520–1580 nm, with TE0 at 1550 nm.
The user-supplied modal indices (n_eff=1.903, n_g=2.24 and approximate FDTD n_g=2.30)
are references only and do not replace bulk birefringent LN material constants.

Use `rotation=x_cut_rotation(45.)` for both the slab and raised LN. Lab xy is
the substrate plane. Crystal x is lab +z, and crystal z is `(1,1,0)/sqrt(2)`;
the columns of the rotation matrix are crystal x,y,z. An azimuth of +45 degrees
is measured from lab +x toward +y. The tensor is fixed throughout the ring,
not co-rotating with propagation. A TE0 input requires the actual vector mode
profile; choosing an electric-field point-source component is not equivalent.

The rotated static tensor generally has off-diagonal xy terms; the internal
axis-aligned DC solver must not be used by discarding them. Use a valid external
DC field for that case. Likewise, a continuous rotated LN slab or rib in a PML
collar can fail the tensor solver's existing CPML admissibility check. This
geometry preparation does not override those checks or claim validated ports.
Electrodes, voltage range, optical mode launch, material calibration and boundary
acceptance remain necessary before a device resonance-shift run.

### Supplied arc electrodes

The device specification records lateral S–G semicircular electrodes (180 degrees)
on the 100 nm slab, so the metal bottom is lab z=0.1 um. The stated placement is
opposite the bus-coupling region, with **Signal inside and Ground outside**.
The user's Fig. 7 clarification gives separate local coordinates for each ring:

| Ring | Approximate coupling angles | Counterclockwise electrode arc |
|---|---|---|
| Left Ring 1 | 90, 30 degrees | 135 to 315 degrees |
| Right Ring 2 | 210, 270 degrees | 315 to 495 degrees (wraps through 0) |

Each ring's own coupling points lie outside its own electrode arc. The earlier
90/270-degree symmetric-bus template was not Fig. 7 and is now explicitly excluded
from the active geometry. Exact ring centres and bus paths await the source paper.
Do not reuse Ring 1's electrode arc on Ring 2. The global selected ring/arc is
unset in the specification so this choice cannot silently apply to every ring.

The physical S–G spacing is **5 um**, while **2.5 um** is the user's effective
distance for an approximate `E=V/d_eff` model. Keep them separate. A future field
solved from actual electrode geometry must not also be multiplied by 5/2.5.
No external DC field was supplied, and no field direction is inferred from the
scalar effective distance alone. With the confirmed inner Signal and outer Ground,
positive S-minus-G bias drives an ideal radially outward field. The full rotated Pockels tensor, not a uniform r33
substitution, applies after the lab field direction is defined.

Metal width/thickness/material, radial locations, the spacing's edge convention,
and bias/control electrical combination are still needed for physical
electrode simulation. The two bus widths are unconfirmed: 1.4 um remains only
the recorded preparation assumption. The specification does not create guessed
metal solids or launch a voltage sweep.

The supplied scenarios are `(ring count, bias V, control range V)`:
`(5,11.4,0..9.5)`, `(10,8.0,0..4.5)`, `(12,7.5,0..3.7)`,
`(20,6.7,0..2.2)`, `(30,6.3,0..1.5)`. These are **magnitudes** as reported by the
user, not signed terminal voltages. The paper does not supply their terminal
combination or voltage spacing according to the user's clarification.

For five rings the supplied normalized law is `d(I)=-2.0789+0.21658*I`, I from
0 to 8; the endpoint is -0.34626. I is not assumed to be voltage. Only under
the explicit assumptions d(0 V)=0, positive d/dV, and a common S-G drive can
one infer `V_SG=-11.4+(0..9.5)`, approximately -11.4 to -1.9 V. Reversing d/dV
gives +11.4 to +1.9 V. Neither is recorded as a confirmed paper sweep. In
particular **+11.4 to +20.9 V is not an adopted condition**. Nonzero initial
laser/resonance detuning changes this mapping.

The user's proposed 0.1 V step is a simulation setting, not a paper datum.
The quoted magnitude 9.1 pm/V, and hence approximately 0.91 pm per such step,
is a paper comparison value reported by the user, not an FDTD result or a
coefficient-fitting target. `paper_interpretation.json` records this provenance
and conditional mappings; `voltage_scenarios.csv` retains only the raw magnitudes.

`effective_arc_field_per_volt(xy_um, active_mask)` provides the explicitly
approximate spatial drive map for these angles, positive radial polarity and
2.5 um effective distance. `xy_um` has shape `(...,2)`; `active_mask` is a matching
Boolean LN interaction-region mask chosen by the caller. Output is `(...,3)`
in V/m per V. Multiply by the actual S-minus-G voltage once. The helper uses a
half-open angular interval and zero drive outside the mask/arc. It does not
solve Maxwell electrostatics at arc ends, metal interfaces or vertical boundaries.
Keep its approximate results separate from a future physical electrode solve.

### Tests

`python -m pytest tests/test_electro_optic.py tests/test_anisotropy.py -q`
checks analytic index slopes, tensor rotation, parallel-plate fields and gauge
invariance, dielectric-interface displacement continuity, actual FDTD voltage
gradients against finite differences, zero-bias parity, external-field interpolation,
sequential optical sweeps, output units, and the requested geometric gaps.
The synthetic resonance tests verify the tracker, not device accuracy.

This is a Python module extension in the local checkout. It does not add a
voltage control to the existing browser UI or deploy/restart the shared server.
# Research sweeps and principal-axis coordinates

Unspecified paper parameters can be design variables. Keep exploratory candidate
sets separate from source-verified specifications and never infer terminal bias
addition from voltage magnitudes alone. Start with a 0 V reference and signed
S–G perturbations; track the same mode before expanding the voltage range.

For X-cut LN, `x_cut_simulation_frame(45)` returns `(lab_from_sim,
crystal_to_sim)`. Transform **all** spatial geometry, fields, sources and monitors
consistently. Row-vector lab coordinates become `points_lab @ lab_from_sim`.
Supply `crystal_to_sim` to `ElectroOpticSimulation(rotation=...)`. Unbiased LN
then has an exactly diagonal dielectric tensor, admissible at principal-axis
CPML faces. Biased LN tensors still need independent collar admission checks;
this helper does not relax the solver's boundary validation.

This coordinate transform does not establish mesh convergence, a TE0 source,
electrode loss, or a physically solved DC field. An effective distance sweep
screens field strength; it is not equivalent to optimizing a fabricated gap.
