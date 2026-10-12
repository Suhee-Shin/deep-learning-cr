"""Static voltage bias -> Pockels tensors -> optical FDTD.

Geometry spacing is in micrometres; voltage in V, DC field in V/m and r in
m/V. Optical and static relative permittivities are separate inputs. The
optical response is frozen during each run (no RF/time-dependent modulation).
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from .tensor_linalg import inverse_3x3, symmetric_eigenvalue_bounds


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite.')


@dataclass(frozen=True)
class LithiumNiobate:
    """Illustrative near-1550-nm indices and user-overridable EO coefficients.

    Crystal axes: z is optic/c axis; contracted order xx, yy, zz, yz, xz,
    xy, with *no* engineering factor of two. Convention: r_xxy=-r22,
    r_yyy=+r22, r_xyx=+r22. Reverse ferroelectric domains via signed r.
    Rotation columns are crystal unit axes expressed in laboratory axes.
    These constants are not a dispersion/temperature/frequency calibration.
    """
    n_o: float = 2.211
    n_e: float = 2.138
    r13: float = 8.6e-12
    r22: float = 3.4e-12
    r33: float = 30.8e-12
    r51: float = 28e-12

    def permittivity(self, field_v_m, *, rotation=None):
        """Return (...,3,3) epsilon = inv(eta0 + r:E), retaining autograd."""
        field = torch.as_tensor(field_v_m)
        if field.is_complex():
            raise ValueError('DC field must be real.')
        if field.dtype not in (torch.float32, torch.float64):
            field = field.to(torch.float64)
        if field.ndim < 1 or field.shape[-1] != 3:
            raise ValueError('field_v_m must have shape (...,3).')
        _finite(field, 'field_v_m')
        if not all(math.isfinite(x) for x in vars(self).values()) or min(self.n_o, self.n_e) <= 0:
            raise ValueError('Indices must be positive and material constants finite.')
        if rotation is not None and torch.as_tensor(rotation).is_complex():
            raise ValueError('rotation must be real.')
        rot = torch.eye(3, dtype=field.dtype, device=field.device) if rotation is None else torch.as_tensor(
            rotation, dtype=field.dtype, device=field.device)
        _finite(rot, 'rotation')
        if rot.shape != (3, 3) or not torch.allclose(rot.T @ rot, torch.eye(3, dtype=field.dtype,
                device=field.device), atol=1e-6, rtol=1e-6) or not torch.isclose(
                torch.linalg.det(rot), field.new_tensor(1.), atol=1e-6, rtol=1e-6):
            raise ValueError('rotation must be a proper orthonormal 3x3 matrix.')
        ex, ey, ez = (field @ rot).unbind(-1)
        xx = self.n_o**-2 - self.r22*ey + self.r13*ez
        yy = self.n_o**-2 + self.r22*ey + self.r13*ez
        zz = self.n_e**-2 + self.r33*ez
        xy, xz, yz = self.r22*ex, self.r51*ex, self.r51*ey
        eta = torch.stack((xx, xy, xz, xy, yy, yz, xz, yz, zz), -1).reshape(field.shape[:-1]+(3, 3))
        if bool(symmetric_eigenvalue_bounds(eta)[0] <= 0):
            raise ValueError('Bias produces a non-positive impermeability tensor.')
        eps = rot @ inverse_3x3(eta) @ rot.T
        return (eps + eps.transpose(-1, -2))/2  # exact symmetry required by tensor solver


def parallel_plate_field(voltage_v, gap_um, direction=(0., 0., 1.)):
    """Uniform E=V/gap directed from the positive plate toward ground."""
    voltage = torch.as_tensor(voltage_v)
    if voltage.is_complex():
        raise ValueError('DC voltage must be real.')
    if voltage.dtype not in (torch.float32, torch.float64):
        voltage = voltage.to(torch.float64)
    direction = torch.as_tensor(direction, dtype=voltage.dtype, device=voltage.device)
    _finite(voltage, 'voltage_v')
    _finite(direction, 'direction')
    if not math.isfinite(gap_um) or gap_um <= 0 or direction.shape != (3,) or direction.norm() == 0:
        raise ValueError('gap_um must be positive and direction a nonzero 3-vector.')
    return voltage[..., None] * direction/direction.norm()/(gap_um*1e-6)


@dataclass
class ElectrostaticResult:
    potential_v: np.ndarray
    field_v_m: np.ndarray
    relative_residual: float


def solve_electrostatic(epsilon_static, electrode_potential_v, spacing_um, *, rtol=1e-10, maxiter=10000):
    """Solve div(eps_static grad(phi))=0 on a uniform node grid (CPU).

    electrode_potential_v is a 2D/3D array: finite entries fix electrode
    voltages, NaN entries are unknowns. Outer faces have zero normal flux
    unless fixed. epsilon_static is scalar, grid-shaped isotropic, or
    grid+(ndim,) axis-aligned diagonal. Off-diagonal static tensors are NOT
    approximated: import an external DC field for rotated anisotropic media.
    Harmonic face epsilon enforces normal displacement continuity. Returned
    fields are node-centred, lab xyz, including a zero z component in 2D.
    DC solves are not differentiable; a solved 1 V field can be scaled in Torch.
    """
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import cg
    if np.iscomplexobj(electrode_potential_v) or np.iscomplexobj(epsilon_static) or np.iscomplexobj(spacing_um):
        raise ValueError('Electrostatic inputs must be real.')
    fixed = np.asarray(electrode_potential_v, dtype=float)
    if fixed.ndim not in (2, 3) or min(fixed.shape) < 2 or np.isinf(fixed).any():
        raise ValueError('Electrode grid must be 2D/3D with >=2 nodes per axis and finite voltages/NaN.')
    spacing = np.asarray(spacing_um, dtype=float)
    if spacing.shape != (fixed.ndim,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('spacing_um must contain one positive spacing per grid axis.')
    if not math.isfinite(rtol) or rtol <= 0 or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError('rtol and maxiter must be positive.')
    pinned = np.isfinite(fixed).ravel()
    if not pinned.any():
        raise ValueError('At least one fixed-potential electrode is required.')
    eps = np.asarray(epsilon_static, dtype=float)
    if eps.ndim == 0 or eps.shape == fixed.shape:
        eps = np.broadcast_to(eps, fixed.shape)[..., None]
    if eps.shape not in (fixed.shape+(1,), fixed.shape+(fixed.ndim,)):
        raise ValueError('epsilon_static must be scalar, grid-shaped, or grid+(ndim,) diagonal.')
    eps = np.broadcast_to(eps, fixed.shape+(fixed.ndim,))
    if not np.isfinite(eps).all() or (eps <= 0).any():
        raise ValueError('Static permittivity must be finite and positive.')
    nodes = np.arange(fixed.size).reshape(fixed.shape)
    rows, cols, values = [], [], []
    for axis in range(fixed.ndim):
        lo, hi = [slice(None)]*fixed.ndim, [slice(None)]*fixed.ndim
        lo[axis], hi[axis] = slice(None, -1), slice(1, None)
        lo, hi = tuple(lo), tuple(hi)
        a, b = nodes[lo].ravel(), nodes[hi].ravel()
        ea, eb = eps[lo][..., axis], eps[hi][..., axis]
        # Common um^-2 scaling cancels from the homogeneous equation.
        g = (2/(1/ea+1/eb)/spacing[axis]**2).ravel()
        rows.extend((a, a, b, b)); cols.extend((a, b, a, b)); values.extend((g, -g, -g, g))
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(fixed.size, fixed.size)).tocsr()
    phi = np.nan_to_num(fixed.ravel(), nan=0.)
    free = ~pinned
    residual = 0.
    if free.any():
        operator = matrix[free][:, free]
        rhs = -(matrix[free][:, pinned] @ phi[pinned])
        # SciPy 1.10/1.11 use tol; 1.12+ use rtol.
        from inspect import signature
        tolerance = {'rtol' if 'rtol' in signature(cg).parameters else 'tol': rtol}
        solution, info = cg(operator, rhs, M=diags(1/operator.diagonal()), atol=0., maxiter=maxiter, **tolerance)
        residual = float(np.linalg.norm(operator @ solution-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny))
        if info != 0 or not np.isfinite(solution).all() or residual > 10*rtol:
            raise RuntimeError(f'Electrostatic solve failed: CG info={info}, residual={residual:g}.')
        phi[free] = solution
    phi = phi.reshape(fixed.shape)
    field = np.zeros(fixed.shape+(3,))
    for axis, gradient in enumerate(np.gradient(phi, *(spacing*1e-6), edge_order=1)):
        field[..., axis] = -gradient
    return ElectrostaticResult(phi, field, residual)


class ElectroOpticSimulation:
    """Apply a DC field per volt only inside an LN mask and run full-tensor FDTD.

    base_epsilon supplies background/other materials; masked entries are
    replaced by the LN tensor, including its zero-bias birefringence.
    Shapes are exactly the optical node grid; resample external fields first.
    """
    def __init__(self, project, base_epsilon, ln_mask, field_per_volt, *, material=None,
                 rotation=None, options=None, **tensor_options):
        from .anisotropy import TensorDielectricSimulation
        self.solver = TensorDielectricSimulation(project, options, **tensor_options)
        if not isinstance(base_epsilon, torch.Tensor):
            raise ValueError('base_epsilon must be a Torch tensor.')
        if base_epsilon.dtype not in (torch.float32, torch.float64):
            raise ValueError('base_epsilon must be real float32 or float64.')
        if torch.as_tensor(field_per_volt).is_complex():
            raise ValueError('field_per_volt must be real.')
        self.solver._validate_input_shape(base_epsilon)
        self.base_epsilon = base_epsilon
        self.mask = torch.as_tensor(ln_mask, device=base_epsilon.device)
        self.field = torch.as_tensor(field_per_volt, dtype=base_epsilon.dtype, device=base_epsilon.device)
        if self.mask.dtype != torch.bool or self.mask.shape != project.region.shape:
            raise ValueError('ln_mask must be a Boolean optical node grid.')
        if self.field.shape not in ((3,), project.region.shape+(3,)):
            raise ValueError('field_per_volt must have shape (3,) or optical grid+(3,).')
        _finite(self.field, 'field_per_volt')
        self.material = material or LithiumNiobate()
        self.rotation = rotation

    def permittivity(self, voltage_v):
        if torch.as_tensor(voltage_v).is_complex():
            raise ValueError('DC voltage must be real.')
        voltage = torch.as_tensor(voltage_v, dtype=self.base_epsilon.dtype, device=self.base_epsilon.device)
        if voltage.ndim != 0:
            raise ValueError('Each optical solve takes one scalar voltage.')
        # Evaluate Pockels tensors only in LN; air/substrate do not need matrix
        # inversion. Indexed assignment preserves gradients to field and voltage.
        field = self.field if self.field.shape == (3,) else self.field[self.mask]
        result = self.base_epsilon.clone()
        if bool(self.mask.any()):
            result[self.mask] = self.material.permittivity(field*voltage, rotation=self.rotation)
        return result

    def __call__(self, voltage_v):
        return self.solver(self.permittivity(voltage_v))

    def sweep(self, voltages_v, wavelength_um, *, monitor_index=0, window=None):
        """Sequential forward-only voltage sweep of one monitor's |DFT(E/H)|².

        This is a local field spectrum, NOT normalized port transmission.
        Supply a time-domain window to exclude source excitation for ringdown.
        Use __call__ instead when voltage derivatives are required.
        """
        voltages = np.asarray(voltages_v)
        wave = np.asarray(wavelength_um)
        if np.iscomplexobj(voltages) or voltages.ndim != 1 or not len(voltages) or not np.isfinite(voltages).all():
            raise ValueError('voltages_v must be a nonempty finite real vector.')
        if np.iscomplexobj(wave) or wave.ndim != 1 or len(wave) < 3 or not np.isfinite(wave).all() or (wave <= 0).any() or (np.diff(wave) <= 0).any():
            raise ValueError('wavelength_um must be positive, finite and strictly increasing.')
        if not isinstance(monitor_index, int) or monitor_index < 0:
            raise ValueError('monitor_index must be a nonnegative integer.')
        frequencies = torch.as_tensor(299792458./(wave*1e-6), device=self.base_epsilon.device,
                                      dtype=self.base_epsilon.dtype)
        rows = []
        with torch.no_grad():
            for voltage in voltages:
                result = self(float(voltage))
                if monitor_index >= result.signals.shape[1]:
                    raise ValueError('monitor_index exceeds the enabled monitor count.')
                power = result.spectrum(frequencies, window=window)[:, monitor_index].abs().square()
                if not bool(torch.isfinite(power).all()) or not bool((power > 0).any()):
                    raise RuntimeError(f'No finite nonzero monitor spectrum at {voltage:g} V.')
                rows.append(power.cpu().numpy())
        return VoltageSweepResult(voltages.astype(float), wave.astype(float), np.asarray(rows),
                                  result.time_step, result.signals.shape[0])


@dataclass
class VoltageSweepResult:
    """Raw monitor spectra; shifts are relative to the first supplied voltage."""
    voltages_v: np.ndarray
    wavelength_um: np.ndarray
    field_power_spectra: np.ndarray
    time_step_s: float
    steps: int

    def resonance_shifts(self, *, target_um, half_window_um, kind='peak'):
        resonance = track_resonances(self.wavelength_um, self.field_power_spectra,
                                    target_um=target_um, half_window_um=half_window_um, kind=kind)
        return resonance, (resonance-resonance[0])*1e6

    def save(self, directory, *, target_um=None, half_window_um=None, kind='peak'):
        """Save raw NPZ/metadata first, then optionally tracked nm/pm CSV.

        If no mode can be identified the raw results remain available, and
        tracking raises instead of fabricating a wavelength shift.
        """
        import csv
        import json
        from pathlib import Path
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        if (target_um is None) != (half_window_um is None):
            raise ValueError('Supply both target_um and half_window_um, or neither.')
        np.savez_compressed(path/'voltage_spectra.npz', voltages_v=self.voltages_v,
                            wavelength_um=self.wavelength_um, field_power_spectra=self.field_power_spectra)
        duration = self.time_step_s*self.steps
        (path/'voltage_sweep.json').write_text(json.dumps(dict(
            reference_voltage_v=float(self.voltages_v[0]), time_step_s=self.time_step_s,
            steps=self.steps, duration_s=duration, nominal_frequency_bin_hz=1/duration,
            spectrum_kind='local_monitor_field_DFT_magnitude_squared',
            limitations='Not calibrated port transmission. Validate mesh, observation duration and mode identity.'
        ), indent=2), encoding='utf-8')
        if target_um is not None:
            resonance, shifts = self.resonance_shifts(target_um=target_um, half_window_um=half_window_um, kind=kind)
            with (path/'resonance_shift.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle)
                writer.writerow(['voltage_v', 'resonance_nm', 'shift_pm'])
                writer.writerows(zip(self.voltages_v, resonance*1000, shifts))


def resample_dc_field(source_axes_um, field_v_m, target_axes_um):
    """Interpolate a rectilinear external 3D DC field onto optical mesh nodes.

    Each source/target axis is in lab xyz order and um; vector components
    remain lab xyz in V/m. Extrapolation is rejected. Rotation belongs in
    LithiumNiobate.permittivity, not in the array-index ordering.
    """
    from scipy.interpolate import RegularGridInterpolator
    if len(source_axes_um) != 3 or len(target_axes_um) != 3:
        raise ValueError('Provide three source and three target axes, ordered x,y,z.')
    source, target = [], []
    for axes, output, minimum in ((source_axes_um, source, 2), (target_axes_um, target, 1)):
        for values in axes:
            axis = np.asarray(values)
            if np.iscomplexobj(axis) or axis.ndim != 1 or len(axis) < minimum or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
                raise ValueError('Coordinate axes must be finite real increasing vectors.')
            output.append(axis.astype(float))
    field = np.asarray(field_v_m)
    if np.iscomplexobj(field) or field.shape != tuple(len(a) for a in source)+(3,) or not np.isfinite(field).all():
        raise ValueError('DC field must be finite real (Nx,Ny,Nz,3), in V/m.')
    points = np.stack(np.meshgrid(*target, indexing='ij'), axis=-1)
    return RegularGridInterpolator(source, field, bounds_error=True)(points)


def add_drop_ring_structures(*, thickness_um, bus_length_um, radius_um=20., width_um=1.4,
                             gap_um=.1, bus_width_um=None, material='LiNbO3', center_z_um=0.):
    """Full-etch circular ring and two straight buses, all in the xy plane.

    Radius is the waveguide centreline radius; gap is edge-to-edge at CP1/CP2.
    The bus width defaults to the ring width as an explicit geometric assumption.
    This does not supply a slab, substrate, electrodes, sources or port modes.
    """
    from .models import Structure
    bus_width = width_um if bus_width_um is None else bus_width_um
    values = (thickness_um, bus_length_um, radius_um, width_um, gap_um, bus_width, center_z_um)
    if not all(math.isfinite(v) for v in values) or min(values[:6]) <= 0:
        raise ValueError('Dimensions must be finite and positive, and center_z_um finite.')
    if radius_um <= width_um/2 or bus_length_um <= 2*(radius_um+width_um/2):
        raise ValueError('Require a positive ring hole and buses extending beyond the ring.')
    y_bus = radius_um+width_um/2+gap_um+bus_width/2
    return [Structure(name='LN-ring', kind='ring', material=material,
                      center=(0., 0., center_z_um), radius=radius_um+width_um/2,
                      inner_radius=radius_um-width_um/2, size=(2*radius_um+width_um, 2*radius_um+width_um, thickness_um)),
            Structure(name='LN-input-bus', material=material, center=(0., -y_bus, center_z_um),
                      size=(bus_length_um, bus_width, thickness_um)),
            Structure(name='LN-drop-bus', material=material, center=(0., y_bus, center_z_um),
                      size=(bus_length_um, bus_width, thickness_um))]


def effective_arc_field_per_volt(xy_um, active_mask, *, effective_distance_um=2.5,
                                 start_deg=135., span_deg=180., signal_inside=True,
                                 center_um=(0., 0.)):
    """Ideal radial EO drive map in V/m per V; NOT an electrostatic solution.

    Positive voltage means Signal minus Ground. The field is radial outward
    for an inner Signal, within the counterclockwise half-open angular arc.
    Caller supplies the LN interaction mask (e.g. ring rib/slab annulus) to
    avoid accidentally biasing all slab and buses. End fringing, vertical
    variation, metal screening and electrostatic boundary continuity are not
    modeled. Do not apply physical_gap/effective_distance to a solved field.
    """
    xy = torch.as_tensor(xy_um)
    if xy.is_complex() or xy.ndim < 1 or xy.shape[-1] != 2:
        raise ValueError('xy_um must be real with shape (...,2).')
    if xy.dtype not in (torch.float32, torch.float64):
        xy = xy.to(torch.float64)
    _finite(xy, 'xy_um')
    center = torch.as_tensor(center_um, dtype=xy.dtype, device=xy.device)
    if center.shape != (2,):
        raise ValueError('center_um must contain x and y.')
    _finite(center, 'center_um')
    mask = torch.as_tensor(active_mask, device=xy.device)
    if mask.dtype != torch.bool or mask.shape != xy.shape[:-1]:
        raise ValueError('active_mask must be Boolean with shape matching the coordinate points.')
    if not all(math.isfinite(v) for v in (effective_distance_um, start_deg, span_deg)) or effective_distance_um <= 0 or not 0 < span_deg <= 360:
        raise ValueError('Effective distance must be positive and arc span in (0,360].')
    if not isinstance(signal_inside, bool):
        raise ValueError('signal_inside must be Boolean.')
    relative = xy-center
    radius = torch.linalg.vector_norm(relative, dim=-1)
    angle = torch.rad2deg(torch.atan2(relative[..., 1], relative[..., 0]))
    enabled = mask & (radius > 0) & (torch.remainder(angle-start_deg, 360.) < span_deg)
    sign = 1. if signal_inside else -1.
    field_xy = sign*relative/radius.clamp_min(torch.finfo(xy.dtype).tiny)[..., None]/(effective_distance_um*1e-6)
    field = torch.cat((field_xy, torch.zeros_like(radius)[..., None]), dim=-1)
    return torch.where(enabled[..., None], field, torch.zeros_like(field))


def x_cut_simulation_frame(azimuth_deg=45.):
    """Return (lab_from_sim, crystal_to_sim) for a principal-axis grid.

    Column vectors transform as v_lab = lab_from_sim @ v_sim; row arrays
    transform as v_sim = v_lab @ lab_from_sim. Geometry, sources, monitors
    and DC fields must all use the same transformation. This is a coordinate
    change, not a change of crystal orientation. Unbiased LN is exactly
    diagonal in this frame, avoiding roundoff off-diagonals in CPML collars.
    Biased material still needs to be kept out of incompatible CPML collars.
    """
    if not math.isfinite(azimuth_deg):
        raise ValueError('azimuth_deg must be finite.')
    angle = math.radians(azimuth_deg)
    c, s = math.cos(angle), math.sin(angle)
    return (np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]),
            np.array([[0., 0., 1.], [0., -1., 0.], [1., 0., 0.]]))


def x_cut_rotation(azimuth_deg=45.):
    """Crystal x = lab +z; crystal z in lab xy at angle from +x toward +y.

    Columns are crystal axes in lab coordinates. The right-handed crystal y
    axis is z_crystal cross x_crystal. This fixed orientation applies to the
    entire single crystal, including the slab and curved ring.
    """
    if not math.isfinite(azimuth_deg):
        raise ValueError('azimuth_deg must be finite.')
    angle = math.radians(azimuth_deg)
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[0., s, c], [0., -c, s], [1., 0., 0.]])


def rib_add_drop_ring_structures(*, footprint_um, bus_length_um, total_thickness_um=.6,
                                etch_depth_um=.5, substrate_thickness_um=2.,
                                radius_um=20., width_um=1.4, gap_um=.1,
                                bus_width_um=None, material='LiNbO3', substrate_material='SiO2'):
    """Rib LN add-drop ring on a continuous LN slab, substrate top at lab z=0.

    The full lateral footprint retains the residual slab, including the
    ring hole and coupling gaps. Footprint is a computational truncation,
    not a physical chip edge. A buried layer below the SiO2 is not inferred.
    Returned structures are geometry only, not a source/boundary configuration.
    """
    from .models import Structure
    footprint = np.asarray(footprint_um, dtype=float)
    if footprint.shape != (2,) or not np.isfinite(footprint).all() or (footprint <= 0).any():
        raise ValueError('footprint_um must contain positive x and y spans.')
    if not all(math.isfinite(v) for v in (total_thickness_um, etch_depth_um, substrate_thickness_um)):
        raise ValueError('Vertical dimensions must be finite.')
    if not 0 < etch_depth_um < total_thickness_um or substrate_thickness_um <= 0:
        raise ValueError('Rib requires 0 < etch depth < total LN thickness and positive substrate thickness.')
    slab = total_thickness_um-etch_depth_um
    ridges = add_drop_ring_structures(thickness_um=etch_depth_um, bus_length_um=bus_length_um,
              radius_um=radius_um, width_um=width_um, gap_um=gap_um, bus_width_um=bus_width_um,
              material=material, center_z_um=slab+etch_depth_um/2)
    outer_bus_edge = abs(ridges[1].center[1])+ridges[1].size[1]/2
    if footprint[0] < bus_length_um or footprint[1] < 2*outer_bus_edge:
        raise ValueError('Footprint must contain both buses and the ring.')
    substrate = Structure(name='SiO2-substrate', material=substrate_material,
                          size=(*footprint, substrate_thickness_um), center=(0., 0., -substrate_thickness_um/2))
    slab_object = Structure(name='LN-residual-slab', material=material,
                            size=(*footprint, slab), center=(0., 0., slab/2))
    return [substrate, slab_object, *ridges]


def track_resonances(wavelength_um, spectra, *, target_um, half_window_um, kind='dip'):
    """Track a local extremum from row to row; return wavelengths in um.

    Start with the spectrum at the reference voltage. The window must be
    smaller than half an FSR yet wider than each expected voltage step shift.
    Three-point quadratic interpolation is used; missing/edge extrema raise
    instead of reporting a band edge as a resonance. This is not a Q fit.
    """
    from scipy.signal import find_peaks
    if np.iscomplexobj(wavelength_um) or np.iscomplexobj(spectra):
        raise ValueError('Wavelengths and power spectra must be real.')
    wave, power = np.asarray(wavelength_um, float), np.asarray(spectra, float)
    if wave.ndim != 1 or len(wave) < 3 or not np.isfinite(wave).all() or (wave <= 0).any() or (np.diff(wave) <= 0).any():
        raise ValueError('Wavelengths must be positive, finite and strictly increasing.')
    if power.ndim != 2 or not len(power) or power.shape[1] != len(wave) or not np.isfinite(power).all():
        raise ValueError('spectra must be finite with shape (voltages,wavelengths).')
    if kind not in ('dip', 'peak') or not math.isfinite(target_um) or not math.isfinite(half_window_um) or half_window_um <= 0:
        raise ValueError('Choose dip/peak and a finite target and positive tracking half-window.')
    tracked = []
    target = target_um
    for row in power:
        peaks, _ = find_peaks(-row if kind == 'dip' else row)
        peaks = peaks[np.abs(wave[peaks]-target) < half_window_um]
        if not len(peaks):
            raise ValueError(f'No interior {kind} near {target:g} um; expand spectral band/run duration or inspect mode tracking.')
        k = peaks[np.argmin(np.abs(wave[peaks]-target))]
        x = wave[k-1:k+2]-wave[k]
        a, b, _ = np.polyfit(x, row[k-1:k+2], 2)
        offset = -b/(2*a) if a != 0 else 0.
        target = float(wave[k]+np.clip(offset, x[0], x[-1]))
        tracked.append(target)
    return np.asarray(tracked)
