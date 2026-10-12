import numpy as np
import pytest
import torch


def test_principal_frame_covariance_and_unbiased_cpml():
    from torchfdtd import x_cut_simulation_frame, x_cut_rotation
    from torchfdtd.anisotropy import cpml_face_admissible
    q, rotation = x_cut_simulation_frame(45.)
    np.testing.assert_allclose(q @ rotation, x_cut_rotation(45.), atol=1e-15)
    np.testing.assert_allclose(q.T @ q, np.eye(3), atol=1e-15)
    material = LithiumNiobate()
    lab_field = torch.tensor([[1e6, -2e6, 3e5], [-1e5, 2e5, 0.]], dtype=torch.float64)
    q = torch.as_tensor(q, dtype=torch.float64)
    eps_lab = material.permittivity(lab_field, rotation=x_cut_rotation(45.))
    eps_sim = material.permittivity(lab_field @ q, rotation=rotation)
    torch.testing.assert_close(q @ eps_sim @ q.T, eps_lab, rtol=1e-13, atol=1e-13)
    unbiased = material.permittivity(torch.zeros(3, dtype=torch.float64), rotation=rotation)
    for axis in range(3):
        assert cpml_face_admissible(unbiased, axis)
    with pytest.raises(ValueError):
        x_cut_simulation_frame(float('nan'))

from torchfdtd import (LithiumNiobate, parallel_plate_field, solve_electrostatic,
                       ElectroOpticSimulation, track_resonances)


def test_plate_voltage_and_extraordinary_index_derivative():
    ln = LithiumNiobate()
    v = torch.tensor(0., dtype=torch.float64, requires_grad=True)
    eps = ln.permittivity(parallel_plate_field(v, 2.))
    torch.testing.assert_close(eps, torch.diag(v.new_tensor([ln.n_o**2, ln.n_o**2, ln.n_e**2])))
    derivative, = torch.autograd.grad(eps[2, 2].sqrt(), v)
    assert derivative.item() == pytest.approx(-.5*ln.n_e**3*ln.r33/(2e-6), rel=1e-12)
    assert ln.permittivity(parallel_plate_field(10., 2.))[2, 2] < eps[2, 2]


def test_tensor_rotation_and_independent_indicatrix():
    ln = LithiumNiobate()
    e = torch.tensor([1e6, -2e6, 3e6], dtype=torch.float64)
    eta = torch.linalg.inv(ln.permittivity(e))
    expected = e.new_tensor([[ln.n_o**-2-ln.r22*e[1]+ln.r13*e[2], ln.r22*e[0], ln.r51*e[0]],
                            [ln.r22*e[0], ln.n_o**-2+ln.r22*e[1]+ln.r13*e[2], ln.r51*e[1]],
                            [ln.r51*e[0], ln.r51*e[1], ln.n_e**-2+ln.r33*e[2]]])
    torch.testing.assert_close(eta, expected)
    rotation = e.new_tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    torch.testing.assert_close(ln.permittivity(rotation@e, rotation=rotation), rotation@ln.permittivity(e)@rotation.T)
    assert torch.equal(ln.permittivity(e), ln.permittivity(e).T)


@pytest.mark.parametrize('shape', [(17, 5), (17, 5, 3)])
def test_electrostatic_parallel_plates_and_gauge(shape):
    boundary = np.full(shape, np.nan)
    boundary[0], boundary[-1] = 3., 0.
    result = solve_electrostatic(28., boundary, [.2]*len(shape))
    analytic = np.linspace(3, 0, shape[0]).reshape((-1,)+(1,)*(len(shape)-1))
    np.testing.assert_allclose(result.potential_v, np.broadcast_to(analytic, shape), atol=1e-8)
    np.testing.assert_allclose(result.field_v_m[..., 0], 3/(16*.2e-6), rtol=1e-8)
    np.testing.assert_allclose(result.field_v_m[..., 1:], 0., atol=.01)
    shifted = solve_electrostatic(28., boundary+7., [.2]*len(shape))
    np.testing.assert_allclose(shifted.field_v_m, result.field_v_m, atol=.01)


def test_dielectric_interface_displacement_continuity():
    boundary = np.full((21, 3), np.nan)
    boundary[0], boundary[-1] = 1., 0.
    eps = np.ones_like(boundary); eps[10:] = 4.
    result = solve_electrostatic(eps, boundary, [1., 1.])
    face_eps = 2/(1/eps[1:]+1/eps[:-1])
    displacement = -face_eps*np.diff(result.potential_v, axis=0)
    np.testing.assert_allclose(displacement, displacement.mean(), atol=1e-9)


def test_voltage_fdtd_gradient_and_unbiased_parity():
    from test_anisotropy import scene
    from torchfdtd.anisotropy import TensorDielectricSimulation
    p = scene('float64', steps=12)
    base = torch.eye(3, dtype=torch.float64).expand(p.region.shape+(3, 3))
    mask = torch.ones(p.region.shape, dtype=torch.bool)
    model = ElectroOpticSimulation(p, base, mask, parallel_plate_field(1., 1.).double())
    with torch.no_grad():
        direct = TensorDielectricSimulation(p)(model.permittivity(0.)).signals
        torch.testing.assert_close(model(0.).signals, direct)
    voltage = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    loss = model(voltage).signals.square().sum()
    gradient, = torch.autograd.grad(loss, voltage)
    with torch.no_grad():
        finite_difference = (model(2.01).signals.square().sum()-model(1.99).signals.square().sum())/.02
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-5, atol=1e-12)
    assert abs(gradient) > 1e-10
    mask[0] = False
    torch.testing.assert_close(model.permittivity(2.)[0], base[0])


def test_tracking_known_shift_and_band_failure():
    wave = np.linspace(1.5, 1.6, 1001)
    centers = np.array([1.55003, 1.54921, 1.54839])
    dips = np.array([1-.8/(1+((wave-c)/.002)**2) for c in centers])
    np.testing.assert_allclose(track_resonances(wave, dips, target_um=1.55, half_window_um=.01), centers, atol=1e-7)
    with pytest.raises(ValueError, match='No interior'):
        track_resonances(wave, np.ones((1, len(wave))), target_um=1.55, half_window_um=.01)


def test_invalid_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        solve_electrostatic(1., np.full((3, 3), np.nan), [1., 1.])
    with pytest.raises(ValueError):
        solve_electrostatic(-1., np.zeros((3, 3)), [1., 1.])
    with pytest.raises(ValueError):
        LithiumNiobate().permittivity([0., 0., 0.], rotation=np.diag([1., 1., -1.]))
    with pytest.raises(ValueError):
        LithiumNiobate().permittivity([0., 0., -1e12])
    with pytest.raises(ValueError):
        parallel_plate_field(1., 0.)


def test_external_field_interpolation_and_bounds():
    from torchfdtd import resample_dc_field
    source = [np.linspace(-2, 2, 7), np.linspace(0, 3, 4), np.linspace(-1, 1, 3)]
    x, y, z = np.meshgrid(*source, indexing='ij')
    field = np.stack((2*x-y, 3*y+z, -z+x), axis=-1)*1e6
    target = [np.linspace(-1, 1, 6), np.array([.3, .6]), np.array([0.])]
    x, y, z = np.meshgrid(*target, indexing='ij')
    expected = np.stack((2*x-y, 3*y+z, -z+x), axis=-1)*1e6
    np.testing.assert_allclose(resample_dc_field(source, field, target), expected, atol=1e-8)
    with pytest.raises(ValueError):
        resample_dc_field(source, field, [np.array([3.]), target[1], target[2]])


def test_voltage_sweep_matches_independent_optical_solves(tmp_path):
    from test_anisotropy import scene
    p = scene('float64', steps=12)
    base = torch.eye(3, dtype=torch.float64).expand(p.region.shape+(3, 3))
    model = ElectroOpticSimulation(p, base, torch.ones(p.region.shape, dtype=torch.bool),
                                   parallel_plate_field(1., 1.))
    wave = np.linspace(1.1, 1.4, 5)
    sweep = model.sweep([0., 4.], wave, monitor_index=2)
    with torch.no_grad():
        direct = model(4.).spectrum(299792458./(wave*1e-6))[:, 2].abs().square().numpy()
    np.testing.assert_allclose(sweep.field_power_spectra[1], direct, rtol=1e-12)
    assert np.any(sweep.field_power_spectra[0] != sweep.field_power_spectra[1])
    sweep.save(tmp_path)
    np.testing.assert_array_equal(np.load(tmp_path/'voltage_spectra.npz')['voltages_v'], [0., 4.])
    with pytest.raises(ValueError, match='real'):
        model(1+2j)


def test_saved_shift_units_and_reference_voltage(tmp_path):
    from torchfdtd import VoltageSweepResult
    wave = np.linspace(1.5, 1.6, 1001)
    centers = np.array([1.55, 1.549, 1.548])
    power = np.asarray([1/(1+((wave-center)/.002)**2) for center in centers])
    result = VoltageSweepResult(np.array([0., 2., 4.]), wave, power, 1e-16, 10000)
    resonance, shifts = result.resonance_shifts(target_um=1.55, half_window_um=.01)
    np.testing.assert_allclose(resonance, centers, atol=1e-12)
    np.testing.assert_allclose(shifts, [0., -1000., -2000.], atol=1e-6)
    result.save(tmp_path, target_um=1.55, half_window_um=.01)
    saved = np.loadtxt(tmp_path/'resonance_shift.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(saved[:, 1], centers*1000)
    np.testing.assert_allclose(saved[:, 2], shifts)


def test_reject_complex_dc_and_empty_spectra():
    with pytest.raises(ValueError, match='real'):
        solve_electrostatic(2+1j, np.zeros((3, 3)), [1., 1.])
    with pytest.raises(ValueError):
        track_resonances([1., 2., 3.], np.empty((0, 3)), target_um=2., half_window_um=.5)


def test_requested_add_drop_ring_edge_gaps():
    from torchfdtd import add_drop_ring_structures
    ring, input_bus, drop_bus = add_drop_ring_structures(thickness_um=.6, bus_length_um=46.)
    assert (ring.radius+ring.inner_radius)/2 == pytest.approx(20.)
    assert ring.radius-ring.inner_radius == pytest.approx(1.4)
    for bus in (input_bus, drop_bus):
        assert abs(bus.center[1])-bus.size[1]/2-ring.radius == pytest.approx(.1)
        assert bus.size[2] == ring.size[2] == .6


def test_x_cut_45_optic_axis_and_zero_bias_tensor():
    from torchfdtd import x_cut_rotation
    rotation = x_cut_rotation(45.)
    np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-15)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    np.testing.assert_allclose(rotation[:, 0], [0., 0., 1.])
    optic_axis = np.array([1., 1., 0.])/np.sqrt(2.)
    np.testing.assert_allclose(rotation[:, 2], optic_axis, atol=1e-15)
    ln = LithiumNiobate()
    epsilon = ln.permittivity(np.zeros(3), rotation=rotation).numpy()
    analytic = ln.n_o**2*np.eye(3)+(ln.n_e**2-ln.n_o**2)*np.outer(optic_axis, optic_axis)
    np.testing.assert_allclose(epsilon, analytic, atol=1e-14)
    field = optic_axis*1e6
    eps_bias = ln.permittivity(field, rotation=rotation).numpy()
    expected_extraordinary = 1/(ln.n_e**-2+ln.r33*1e6)
    np.testing.assert_allclose(eps_bias@optic_axis, expected_extraordinary*optic_axis, atol=1e-14)


def test_rib_stack_leaves_slab_below_ring_hole_and_coupling_gaps():
    from torchfdtd import rib_add_drop_ring_structures
    substrate, slab, ring, bus, drop = rib_add_drop_ring_structures(footprint_um=(46., 46.), bus_length_um=46.)
    def bounds(obj):
        return np.asarray([obj.center[2]-obj.size[2]/2, obj.center[2]+obj.size[2]/2])
    np.testing.assert_allclose(bounds(substrate), [-2., 0.], atol=1e-15)
    np.testing.assert_allclose(bounds(slab), [0., .1], atol=1e-15)
    for ridge in (ring, bus, drop):
        np.testing.assert_allclose(bounds(ridge), [.1, .6], atol=1e-15)
    assert slab.kind == 'rectangle' and slab.size[:2] == (46., 46.)
    for guide in (bus, drop):
        assert abs(guide.center[1])-guide.size[1]/2-ring.radius == pytest.approx(.1)
    with pytest.raises(ValueError, match='Rib requires'):
        rib_add_drop_ring_structures(footprint_um=(46., 46.), bus_length_um=46., etch_depth_um=.6)


def test_effective_arc_sector_polarity_and_crystal_projection():
    from torchfdtd import effective_arc_field_per_volt, x_cut_rotation
    xy = torch.tensor([[20., 0.], [0., 20.], [-20., 0.], [0., -20.], [0., 0.]], dtype=torch.float64)
    field = effective_arc_field_per_volt(xy, torch.ones(5, dtype=torch.bool))
    # 0/90 degrees excluded, 180/270 included; centre has no radial direction.
    expected = torch.tensor([[0.,0.,0.], [0.,0.,0.], [-4e5,0.,0.], [0.,-4e5,0.], [0.,0.,0.]], dtype=torch.float64)
    torch.testing.assert_close(field, expected)
    reversed_field = effective_arc_field_per_volt(xy, torch.ones(5, dtype=torch.bool), signal_inside=False)
    torch.testing.assert_close(reversed_field, -field)
    crystal_field = field @ torch.tensor(x_cut_rotation(45.), dtype=torch.float64)
    torch.testing.assert_close(crystal_field[2:4, 2], torch.full((2,), -4e5/np.sqrt(2.), dtype=torch.float64))
    blocked = effective_arc_field_per_volt(xy, torch.zeros(5, dtype=torch.bool))
    assert torch.count_nonzero(blocked) == 0


def test_effective_arc_wrap_and_invalid_inputs():
    from torchfdtd import effective_arc_field_per_volt
    xy = np.array([[1.,0.], [0.,1.], [-1.,0.]])
    field = effective_arc_field_per_volt(xy, np.ones(3, dtype=bool), start_deg=300., span_deg=120.)
    np.testing.assert_allclose(field[0], [4e5,0.,0.])
    assert torch.count_nonzero(field[1:]) == 0
    with pytest.raises(ValueError):
        effective_arc_field_per_volt(xy, np.ones(3, dtype=bool), effective_distance_um=0.)


def test_fig7_each_ring_excludes_its_own_coupling_points():
    import runpy
    from pathlib import Path
    from torchfdtd import effective_arc_field_per_volt
    spec = runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/prepare_linbo3_rib_device.py'))
    for ring in spec['fig7_ring_layout']():
        angles = np.deg2rad(ring['coupling_angles_deg_approx'])
        points = 20*np.stack([np.cos(angles), np.sin(angles)], axis=-1)
        field = effective_arc_field_per_volt(points, np.ones(2, dtype=bool),
                        start_deg=ring['electrode_start_deg'], span_deg=ring['electrode_span_deg'])
        assert torch.count_nonzero(field) == 0
        midpoint = np.deg2rad(ring['electrode_start_deg']+90.)
        inside = effective_arc_field_per_volt(np.array([[20*np.cos(midpoint),20*np.sin(midpoint)]]),
                        np.ones(1,dtype=bool), start_deg=ring['electrode_start_deg'])
        assert inside.norm().item() == pytest.approx(4e5)
        # Fig.7's V-inverted wiring compensates the opposite crystal-axis
        # projection of the two semicircles, rather than changing the crystal.
        optic_axis = torch.tensor([1.,1.,0.], dtype=inside.dtype)/np.sqrt(2.)
        signed_projection = (inside[0]@optic_axis)*ring['drive_polarity_relative_to_common_control']
        assert signed_projection.item() == pytest.approx(4e5)
    interpretation = spec['paper_interpretation']()
    assert interpretation['normalized_detuning_5_rings']['d_at_I_max'] == pytest.approx(-.34626)
    assert all(row['voltage_combination'] is None for row in spec['voltage_scenarios']())
    assert spec['electrode_specification']()['selected_ring'] is None
