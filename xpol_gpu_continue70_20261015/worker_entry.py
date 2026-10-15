"""Replace only presentation hooks; leave the frozen numerical runner untouched."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
import run_tio2_xypol_filter_voxel3d as runner
import portable_report as report


def configure():
    for name in ('write_summary', 'save_history_plots', 'refresh_live_plots', 'write_live_status'):
        setattr(runner, name, getattr(report, name))
    return runner


if __name__ == '__main__':
    configure().main()
