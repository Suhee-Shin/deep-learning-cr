import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'src'))
import worker_entry
import run_x
import portable_report

r = worker_entry.configure()


class ResumeTests(unittest.TestCase):
    def test_unchanged_numerical_code(self):
        run_x.verify_code()

    def test_settings_continue_exactly_seventy(self):
        command = run_x.worker_command(Path('output'))
        self.assertEqual(command[command.index('--max-updates')+1], '270')
        self.assertNotIn('--init-state', command)
        self.assertNotIn('--init-binary', command)
        self.assertNotIn('--pass-pol', command)
        manifest = portable_report.read(ROOT/'seed/manifest.json')
        self.assertEqual(manifest['source_updates_completed'], 200)
        self.assertEqual(manifest['adam_updates'], 200)
        self.assertIsNone(manifest['rotation'])
        self.assertEqual(command[command.index('--lr')+1], '.002')
        self.assertEqual(command[command.index('--beta0')+1], '4.5')
        self.assertEqual(command[command.index('--wavelengths')+1], '400,500,600,700')

    def test_reporting_hook(self):
        self.assertIs(r.write_summary, portable_report.write_summary)
        self.assertIs(r.save_history_plots, portable_report.save_history_plots)

    def test_completed_run_extension_preserves_adam(self):
        def forward(fwd, density, **kwargs):
            mean = density.mean((1, 2, 3))
            return torch.stack((.5+.2*mean, .4-.1*mean), -1)[:, None, None].expand(-1, 1, 3, -1)
        def execute(out, budget):
            argv = ['worker', '--out', str(out), '--device', 'cpu', '--grid', '2', '--layers', '1',
                '--sigma-nm', '0', '--layer-height-nm', '100', '--max-updates', str(budget),
                '--lr', '.002', '--lr-final', '.002', '--thresholds', '0.5', '--objective-mode', 'softmin',
                '--binary-forward-ste', '--binary-evaluation', 'filtered_threshold', '--beta-step', '1']
            with contextlib.ExitStack() as ctx:
                ctx.enter_context(patch('sys.argv', argv))
                ctx.enter_context(patch.object(r, 'build_stack', return_value=({}, None, [], 100.)))
                ctx.enter_context(patch.object(r, 'FastPupilForward', return_value=SimpleNamespace()))
                ctx.enter_context(patch.object(r, 'polarized_tabs', side_effect=forward))
                for name in ('write_summary', 'save_history_plots', 'save_snapshot_plot', 'save_mask_images', 'save_evolution_overview'):
                    ctx.enter_context(patch.object(r, name))
                ctx.enter_context(contextlib.redirect_stdout(io.StringIO()))
                r.main()
        with tempfile.TemporaryDirectory() as folder:
            full, split = Path(folder)/'full', Path(folder)/'split'
            for out in (full, split):
                out.mkdir()
                torch.save(dict(theta=torch.full((1, 2, 2), .51, dtype=torch.float64),
                    m=torch.zeros((1, 2, 2), dtype=torch.float64), v=torch.zeros((1, 2, 2), dtype=torch.float64),
                    beta=4.5, step=-1, updates_completed=0, adam_updates=0), out/'state.pt')
            execute(full, 4)
            execute(split, 2)
            self.assertEqual(run_x.verified_updates(split, (1, 2, 2)), 2)
            (split/'STOP_REQUESTED').touch()
            execute(split, 4)
            execute(split, 4)
            (split/'STOP_REQUESTED').unlink()
            execute(split, 4)
            a, b = [torch.load(out/'state.pt', weights_only=False) for out in (full, split)]
            for key in ('theta', 'm', 'v'):
                torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
            self.assertEqual(b['adam_updates'], 4)
            self.assertEqual(run_x.verified_updates(split, (1, 2, 2)), 4)
            rows = portable_report.read(split/'history.json')
            self.assertEqual([row['step'] for row in rows], [0, 1, 2, 3, 4])
            self.assertFalse(rows[-1]['update_applied'])
            rows[-1]['adam_updates'] = 3
            portable_report.json_save(split/'history.json', rows)
            with self.assertRaises(ValueError):
                run_x.verified_updates(split, (1, 2, 2))

    def test_small_actual_rcwa_and_gradient(self):
        torch.set_num_threads(2)
        r.eigmod.EigConfig.backend = 'cpu'
        r.eigmod.EigConfig.cpu_workers = 1
        args = SimpleNamespace(period_um=.6, grid=12, layer_height_nm=100, layers=2,
            detector_gap_um=.35, orders=2, detector_samples=16, rayleigh_nudge=1e-4)
        stack, pupil, rays, _ = r.build_stack(args)
        fwd = r.FastPupilForward(stack=stack, pupil_spec=pupil, wavelengths_nm=[650],
            rays=rays, device=torch.device('cpu'), chunk=1, dtype=torch.complex128, verbose_guard=False)
        rho = torch.linspace(.1, .9, 288, dtype=torch.float64).reshape(2, 12, 12).requires_grad_()
        tabs = r.polarized_tabs(fwd, rho[None], tags=('smoke',), grad=[True])[0]
        objective = tabs[..., 0].sum()-tabs[..., 1].sum()
        grad, = torch.autograd.grad(objective, rho)
        self.assertTrue(bool(torch.isfinite(tabs).all() and torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().max()), 0)


if __name__ == '__main__':
    unittest.main()
