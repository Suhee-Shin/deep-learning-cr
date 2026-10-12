"""Prepare the user-specified X-cut rib geometry without launching FDTD.

No cladding, electrode, guided source profile or PML choices are fabricated.
The output is a design specification, not a runnable native Project JSON.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torchfdtd import rib_add_drop_ring_structures, x_cut_rotation


def electrode_specification():
    """User-provided dimensions, explicitly separate from unresolved geometry."""
    return dict(
        configuration='lateral_signal_ground', shape='semicircular_arc', arc_span_deg=180.,
        placement_description='On the 100 nm LN slab surface next to the ring, opposite the bus coupling region',
        electrode_bottom_z_um=.1, physical_signal_ground_gap_um=5.,
        effective_field_distance_um=2.5,
        effective_distance_role='User-supplied model approximation for E=V/d_eff; not a solved electrostatic gap',
        electric_field_file=None, arc_start_deg=None, arc_end_deg=None,
        selected_ring=None, ring_specific_layout=fig7_ring_layout(),
        angular_convention='0 degrees = lab +x (right), positive counterclockwise in lab xy',
        signal_side='inner', ground_side='outer', positive_voltage_field_direction='radially_outward_ideal_approximation',
        metal=None, metal_thickness_um=None, metal_width_um=None,
        radial_positions_um=None, gap_edge_definition_confirmed=False,
        field_model_status='effective_radial_arc_available; physical_electrostatics_not_solved',
        double_counting_rule='Use either a solved field from physical geometry or the effective-distance model; do not multiply the solved field by physical_gap/effective_distance')


def voltage_scenarios():
    return [dict(ring_count=n, bias_magnitude_v=bias, control_magnitude_min_v=0., control_magnitude_max_v=control,
                 control_step_v=None, voltage_combination=None,
                 provenance='Verified Table IV (PDF p.8), magnitudes only; terminal voltages unspecified')
            for n, bias, control in [(5, 11.4, 9.5), (10, 8., 4.5), (12, 7.5, 3.7),
                                     (20, 6.7, 2.2), (30, 6.3, 1.5)]]


def fig7_ring_layout():
    return [dict(ring_label='Ring 1', position='left', coupling_angles_deg_approx=[90., 30.],
                 electrode_start_deg=135., electrode_end_deg_unwrapped=315., electrode_span_deg=180.,
                 drive_polarity_relative_to_common_control=-1,
                 polarity_source='Fig.7 p.8 explicitly labels Ring 1 V inverted'),
            dict(ring_label='Ring 2', position='right', coupling_angles_deg_approx=[210., 270.],
                 electrode_start_deg=315., electrode_end_deg_unwrapped=495., electrode_span_deg=180.,
                 drive_polarity_relative_to_common_control=1,
                 polarity_source='Reference drive polarity relative to inverted Ring 1 in Fig.7')]


def paper_interpretation():
    return dict(
        provenance='Read source PDF text and visually checked Figs.3/4/7, Tables II/III/IV/S6/S7',
        paper_reference=dict(arxiv='2603.12934v3',
            title='Photonic Exponential Approximation via Cascaded TFLN Microring Resonators toward Softmax',
            pages=[2,5,6,8,9,24,25]),
        normalized_detuning_5_rings=dict(formula='d(I)=a+b*I', a=-2.0789, b=.21658,
                                         I_min=0., I_max=8., I_units='generic normalized control amplitude; not terminal volts (Sec.II)',
                                         d_at_I_min=-2.0789, d_at_I_max=-2.0789+.21658*8),
        conditional_voltage_examples_5_rings=dict(
            assumptions=['d(0 V)=0; zero-voltage resonance coincides with the chosen detuning origin',
                         'Control magnitude counteracts the bias detuning',
                         'Bias/control magnitudes refer to the same S-G drive for this illustrative mapping'],
            positive_d_dV=dict(formula='V_SG=-V_bias+V_ctrl', range_v=[-11.4, -1.9]),
            negative_d_dV=dict(formula='V_SG=+V_bias-V_ctrl', range_v=[11.4, 1.9]),
            status='conditional examples, not confirmed paper terminal-voltage sweeps',
            positive_additive_sweep_allowed=False),
        proposed_sampling=dict(control_step_v=.1, status='user-proposed simulation setting, not a paper value',
                               paper_sensitivity_magnitude_pm_per_v_approx=9.1,
                               sensitivity_provenance='Rounded 28.48/pi from Eqs.S4.3-S4.5; first-order EO estimate, not voltage-swept FDTD',
                               shift_per_step_pm_approx=.91,
                               use='comparison only; do not force the material coefficients to reproduce this slope'),
        fig7_geometry_source='Fig.7 visually checked; approximate angles, schematic pitch=50 um; exact bus dimensions not supplied',
        detuning_definition=dict(equations='8,10,11,12', delta_omega='omega_L-omega_0',
                                 a='(omega_L-omega_0_at_zero_control)/Gamma', b='-eta/Gamma',
                                 sign_note='Simultaneously reversing a,b leaves Lorentzian transmission unchanged; b_V magnitude does not fix terminal polarity'),
        paper_eo_reference=dict(r33_m_per_v=30.9e-12, n_e=2.138, overlap_factor=.7,
                               fill_factor='1/pi', effective_distance_um=2.5, physical_gap_um=5.,
                               straight_sensitivity_pm_per_v=28.48, normalized_sensitivity_per_v=.182,
                               calibration_Q_loaded_best=15500, calibration_group_index=2.30,
                               evidence='First-order EO estimate using FDTD-calibrated Q and group index; Sec.IV.E and S4.2',
                               no_double_counting='Do not multiply a resolved spatial tensor FDTD response by overlap=.7 and fill=1/pi again'),
        source_discrepancies=[
            dict(item='electrode gap', fig7_label_um=3.5, table_III_and_S4_um=5.,
                 model_choice_um=5., reason='retain user-requested main-text/Table III value; schematic conflict retained'),
            dict(item='optical band', earlier_user_range_nm=[1520,1580], paper_FDTD_range_nm=[1530,1570],
                 policy='preserve user range separately; do not relabel it as paper range'),
            dict(item='N=30 fit coefficients', table_IV=dict(a=-1.141,b=.03340),
                 table_S7=dict(a=-1.1392,b=.03341), policy='no silent selection of conflicting fit values')],
        baseline_geometry=dict(figure='Fig.4 p.6', coupling_angles_deg=[90.,270.],
                               note='Earlier symmetric-bus factory matches the single-ring topology, not the Fig.7 cascade'),
        paper_fdtd_setup=dict(solver='Lumerical 3D', wavelength_nm=[1530,1570], duration_ps=50.,
                              auto_shutoff=1e-6, grid=[532,816,44], source='TE0 mode source',
                              mesh='conformal variant 1, automatic accuracy 3',
                              note='Reference only; not directly transferable to uniform tensor TorchFDTD'),
        simulation_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--footprint-um', type=float, default=46.)
    parser.add_argument('--bus-length-um', type=float, default=46.)
    args = parser.parse_args()
    geometry = rib_add_drop_ring_structures(footprint_um=(args.footprint_um, args.footprint_um),
                                            bus_length_um=args.bus_length_um)
    data = dict(
        name='lab_shshin-LN-rib-add-drop-R20-Xcut45', status='paper_constraints_recorded_fig7_geometry_pending',
        coordinates='lab xy = substrate plane; lab z = surface normal; substrate top z=0; crystal x=lab +z',
        geometry_units='um', ring_radius_definition='waveguide centreline',
        ln_total_thickness_um=.6, etch_depth_um=.5, residual_slab_um=.1,
        substrate_material='SiO2', substrate_thickness_um=2.,
        upper_cladding='Air', electrodes=electrode_specification(),
        voltage_scenarios=voltage_scenarios(),
        paper_interpretation=paper_interpretation(),
        geometry_status='Provisional symmetric buses do not represent Fig.7; not valid for a paper-reproduction run',
        confirmed_bus_width_um=None,
        radius_um=20., width_um=1.4, coupling_gap_um=.1,
        crystal_cut='X', crystal_z_azimuth_deg=45.,
        crystal_to_lab_rotation=x_cut_rotation(45.).tolist(),
        wavelength_range_um=[1.52, 1.58], reference_wavelength_um=1.55, requested_mode='TE0',
        supplied_modal_reference=dict(n_eff=1.903, n_group_mode=2.24, n_group_fdtd_approx=2.30,
                                     use='user-provided comparison data; not the bulk LN refractive index'),
        assumptions=dict(bus_width_um=1.4, sidewall_angle_deg=90.,
                         computational_footprint_um=[args.footprint_um]*2,
                         bus_length_um=args.bus_length_um,
                         note='Bus width and vertical sidewalls pending confirmation; footprint is not a physical chip edge'),
        pending=dict(electrode_metal_dimensions_and_radial_positions=None,
                     bias_control_combination=None, voltage_step_v=None, array_layout_and_ring_addressing=None,
                     bus_width_um=None,
                     te0_mode_profile=None, bulk_material_calibration=None,
                     absorbing_boundary_validation=None, below_silica_stack=None),
        structures=[], provisional_structures_not_fig7=[obj.model_dump(mode='json') for obj in geometry],
        limitations=['The LN slab shares the rotated bulk tensor and voltage response.',
                     'Do not replace LN material indices with modal n_eff or group index.',
                     'A scalar point source does not certify TE0 launch.',
                     'A 45-degree in-plane anisotropic slab touching CPML can fail tensor PML admission; validate boundaries before a full run.'])
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out/'device_spec.json'
    if destination.exists():
        raise FileExistsError('Refusing to overwrite existing device specification')
    destination.write_text(json.dumps(data, indent=2), encoding='utf-8')
    print(destination.resolve())
    print('Prepared geometry only. No FDTD, GPU allocation, or resonance prediction performed.')


if __name__ == '__main__':
    main()
