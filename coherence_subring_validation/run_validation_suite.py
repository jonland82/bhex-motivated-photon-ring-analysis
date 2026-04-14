"""
run_validation_suite.py

Single-entry rerun script for the combined coherence + subring validation suite.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


SUITE_ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "01_generate_coherence_subring_dataset.py",
    "02_run_coherence_subring_benchmark.py",
    "03_build_coherence_subring_report.py",
]


for script_name in SCRIPTS:
    script_path = SUITE_ROOT / script_name
    print(f"\n==> Running {script_name}", flush=True)
    subprocess.run([sys.executable, str(script_path)], cwd=SUITE_ROOT.parent, check=True)

print("\nFinished the full coherence + subring validation suite.", flush=True)
