import runpy
from pathlib import Path
import numpy as np
import pytest

analyze = runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/analyze_ln_ringdown.py'))['peak_from_trace']


def test_known_synthetic_damped_mode_voltage_shifts():
    # Synthetic diagnostic data only; not a device result.
    dt = 1e-15
    t = (np.arange(60000)+1)*dt
    found = []
    for shift_pm in (0., -5., 5.):
        wave = 1.55+shift_pm/1e6
        signal = np.exp(-t/20e-12)*np.cos(2*np.pi*299792458./(wave*1e-6)*t)
        result = analyze(np.stack([signal, .3*signal],axis=1), dt,
                         start_s=.2e-12, target_um=1.55, half_window_um=.0005)
        found.append(result['wavelength_um'])
    np.testing.assert_allclose((np.array(found)-found[0])*1e6, [0.,-5.,5.], atol=.03)


def test_short_trace_cannot_be_reported_as_resonance():
    with pytest.raises(ValueError, match='Insufficient'):
        analyze(np.ones((300,3)), 2.7e-16, start_s=.2e-12,
                target_um=1.55, half_window_um=.0005)


def test_resolved_competing_modes_are_rejected():
    dt=1e-15
    t=(np.arange(60000)+1)*dt
    signal=sum(np.exp(-t/20e-12)*np.cos(2*np.pi*299792458./(wave*1e-6)*t)
               for wave in (1.5497,1.5503))
    with pytest.raises(ValueError, match='found 2'):
        analyze(signal[:,None],dt,start_s=.2e-12,target_um=1.55,half_window_um=.0005)
