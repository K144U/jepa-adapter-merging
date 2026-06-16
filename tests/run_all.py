"""Run every test script without pytest. Exit nonzero on any
failure. Usage: .venv/bin/python tests/run_all.py [--fast]"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = str(HERE.parent / ".venv" / "bin" / "python")

tests = ["test_metrics.py", "test_sigreg.py", "test_lora_mergers.py"]
if "--fast" not in sys.argv:
    tests.append("test_e2e_toy.py")

failed = []
for t in tests:
    print(f"\n=== {t} ===")
    rc = subprocess.run([PY, str(HERE / t)]).returncode
    if rc != 0:
        failed.append(t)

print("\n" + ("ALL TEST SCRIPTS PASS" if not failed else f"FAILED: {failed}"))
sys.exit(1 if failed else 0)
