"""Shared script plumbing: repo-root sys.path, arg helpers, done-files."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def finish(done, payload=None):
    from isomerge.utils import write_done
    if done:
        write_done(ROOT / done if not str(done).startswith("/") else done,
                   payload)
