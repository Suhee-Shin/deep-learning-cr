import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import run_tio2_xypol_filter_voxel3d as r
from voxel_target_validation import target_reached
from portable_report import publish, save_history_plots, write_summary


class RoleTests(unittest.TestCase):
    def test_objective_swaps_roles_not_physical_results(self):
        tabs = torch.tensor([[[.2, .85], [.12, .79], [.3, .8]]], dtype=torch.float64, requires_grad=True)
        kwargs = dict(w_delta=6, w_tx=1.2, w_ty=2, target_tx=.8, target_ty=.1,
            w_tx_floor=40, w_ty_ceiling=20, worst_weight=5, objective_mode='softmin')
        y, _ = r.objective_from_tabs(tabs, pass_pol='y', **kwargs)
        x, _ = r.objective_from_tabs(tabs.flip(-1), pass_pol='x', **kwargs)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        grad, = torch.autograd.grad(y, tabs)
        self.assertTrue(bool((grad[..., 0] < 0).all()))
        self.assertTrue(bool((grad[..., 1] > 0).all()))
        v = torch.tensor([[[.1, -.2], [.3, .1], [-.2, .2]]], dtype=torch.float64)
        for delta in (1e-4, 1e-5, 1e-6):
            plus = r.objective_from_tabs(tabs.detach()+delta*v, pass_pol='y', **kwargs)[0]
            minus = r.objective_from_tabs(tabs.detach()-delta*v, pass_pol='y', **kwargs)[0]
            torch.testing.assert_close((plus-minus)/(2*delta), (grad*v).sum(), rtol=1e-5, atol=1e-8)
        rows = r.spectrum_rows([400, 500, 600], tabs, 'y')
        self.assertEqual(rows[0]['tx'], .2)
        self.assertEqual(rows[0]['ty'], .85)
        self.assertAlmostEqual(rows[0]['delta'], .65)
        self.assertAlmostEqual(rows[0]['er_db'], 10*np.log10(.85/.2))

    def test_acceptance_direction(self):
        self.assertTrue(target_reached([dict(tx=.05, ty=.9)], .8, .1, 'y'))
        self.assertFalse(target_reached([dict(tx=.9, ty=.05)], .8, .1, 'y'))
        self.assertFalse(target_reached([dict(tx=.1, ty=.8)], .8, .1, 'y'))

    def test_small_rcwa_rotation_symmetry(self):
        torch.set_num_threads(2)
        r.eigmod.EigConfig.backend = 'cpu'
        r.eigmod.EigConfig.cpu_workers = 1
        args = SimpleNamespace(period_um=.6, grid=12, layer_height_nm=100, layers=2,
            detector_gap_um=.35, orders=2, detector_samples=16, rayleigh_nudge=1e-4)
        stack, pupil, rays, _ = r.build_stack(args)
        forward = r.FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=[500, 650],
            rays=rays, device=torch.device('cpu'), chunk=1, dtype=torch.complex128, verbose_guard=False)
        density = torch.zeros((2, 12, 12), dtype=torch.float64)
        density[:, 2:6] = 1
        density[0, 8:10, 1:4] = 1
        with torch.no_grad():
            a = r.polarized_tabs(forward, density[None], tags=('x_original',), grad=[False])[0].sum(0)
            b = r.polarized_tabs(forward, torch.rot90(density, dims=(1, 2))[None],
                                  tags=('y_rotated',), grad=[False])[0].sum(0)
        torch.testing.assert_close(b, a.flip(-1), rtol=2e-7, atol=2e-8)
        self.assertGreater(float((a[:, 0]-a[:, 1]).abs().max()), .01)

    def test_recycle_exact_resume_and_moments(self):
        def mock(forward, density, **kwargs):
            mean = density.mean((1, 2, 3))
            return torch.stack((.5+.2*mean, .4-.1*mean), -1)[:, None, None].expand(-1, 1, 3, -1)
        def execute(out):
            argv = ['worker', '--out', str(out), '--device', 'cpu', '--grid', '2', '--layers', '1',
                '--sigma-nm', '0', '--layer-height-nm', '100', '--max-updates', '3', '--lr', '.002',
                '--lr-final', '.002', '--thresholds', '0.5', '--objective-mode', 'softmin',
                '--binary-forward-ste', '--binary-evaluation', 'filtered_threshold', '--beta-step', '1',
                '--pass-pol', 'y', '--gray-penalty-max', '.02', '--continuation-offset', '126']
            with contextlib.ExitStack() as ctx:
                ctx.enter_context(patch('sys.argv', argv))
                ctx.enter_context(patch.object(r, 'build_stack', return_value=({}, None, [], 100.)))
                ctx.enter_context(patch.object(r, 'FastPupilForward', return_value=SimpleNamespace()))
                ctx.enter_context(patch.object(r, 'polarized_tabs', side_effect=mock))
                for name in ('write_summary', 'save_history_plots', 'save_snapshot_plot', 'save_mask_images', 'save_evolution_overview'):
                    ctx.enter_context(patch.object(r, name))
                ctx.enter_context(contextlib.redirect_stdout(io.StringIO()))
                r.main()
        with tempfile.TemporaryDirectory() as folder:
            full, split = Path(folder)/'full', Path(folder)/'split'
            for out in (full, split):
                out.mkdir()
                torch.save(dict(theta=torch.full((1, 2, 2), .51, dtype=torch.float64),
                    m=torch.full((1, 2, 2), .001, dtype=torch.float64),
                    v=torch.full((1, 2, 2), .0001, dtype=torch.float64),
                    beta=4.5, step=-1, updates_completed=0, adam_updates=126), out/'state.pt')
            execute(full)
            (split/'STOP_REQUESTED').touch()
            for _ in range(3):
                execute(split)
            (split/'STOP_REQUESTED').unlink()
            execute(split)
            a, b = [torch.load(o/'state.pt', weights_only=False) for o in (full, split)]
            for key in ('theta', 'm', 'v'):
                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
            self.assertEqual(b['adam_updates'], 129)
            self.assertEqual(b['updates_completed'], 3)
            rows = json.loads((split/'history.json').read_text())
            self.assertEqual([x['step'] for x in rows], [0, 1, 2, 3])
            self.assertTrue(all(x['gray_penalty_weight'] == .02 for x in rows))
            self.assertFalse(rows[-1]['update_applied'])

    def test_reports_keep_actual_xy(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            rows = r.spectrum_rows([400, 500, 600, 700], torch.tensor([[[.1, .8]]*4]), 'y')
            metrics = r.metrics_from_tabs(torch.tensor([[[.1, .8]]*4]), 'y')
            record = dict(step=0, beta=4.5, spectrum=rows, bin_spectrum=rows, **metrics,
                **{'bin_'+k: v for k, v in metrics.items()})
            (out/'config.json').write_text(json.dumps(dict(pass_pol='y', max_updates=200)))
            (out/'history.json').write_text(json.dumps([record]))
            save_history_plots(out, [record], {})
            payload = publish(out)
            self.assertGreater(payload['latest']['bin_mean_ty'], payload['latest']['bin_mean_tx'])
            self.assertTrue((out/'performance.png').exists())
            self.assertIn('receiveStatus', (out/'status.js').read_text(encoding='utf-8'))
            write_summary(out, [400, 500, 600, 700], [record], {}, SimpleNamespace(max_updates=200))
            self.assertIn('TY-TX', (out/'REPORT_KO.md').read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
