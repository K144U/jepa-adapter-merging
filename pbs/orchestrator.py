"""Multi-GPU work-queue orchestrator for the experiment grid.

Cells are shell commands with done-files for idempotency; one worker thread
per GPU; VRAM-gated launch with a no-charge requeue on rc=87 (a shared-GPU
admission gate); retry once then park; _QUEUE_COMPLETE / _QUEUE_FAILED
sentinels signal completion to an outer requeue loop.

  python pbs/orchestrator.py --manifest configs/manifests/<stage>.json [--dry-run]

GPUs come from env GPUS. State: logs/orchestrator_state.json (override via
ORCH_STATE); sentinel override via ORCH_SENTINEL (set both when running two
orchestrators concurrently).
"""

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGDIR = ROOT / "logs" / "orch"
STATE = ROOT / "logs" / os.environ.get("ORCH_STATE", "orchestrator_state.json")
BACKOFF_S = 600
VRAM_POLL_TRIES = 3


def free_gb(gpu: int) -> float:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)], capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip()) / 1024.0
    except Exception:
        return 0.0


def vram_snapshot(note: str) -> None:
    """All-GPU free-VRAM line to a login-visible file (the PBS stdout spool
    lives on the compute node, so this is the only way to see occupancy
    from the login node without ssh)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
        line = "; ".join(
            f"gpu{r.split(',')[0].strip()}:"
            f"{float(r.split(',')[1]) / 1024:.0f}/"
            f"{float(r.split(',')[2]) / 1024:.0f}GB free"
            for r in out.stdout.strip().splitlines())
        with open(ROOT / "logs" / "vram_snapshots.log", "a") as f:
            f.write(f"{time.strftime('%F %T')} [{note}] {line}\n")
    except Exception:
        pass


class Orchestrator:
    def __init__(self, cells, gpus, dry):
        self.q = queue.Queue()
        self.gpus = gpus
        self.dry = dry
        self.lock = threading.Lock()
        self.done, self.failed, self.running = [], [], {}
        self.attempts = {}
        self.skipped = []
        for c in cells:
            if Path(c["done"]).exists() or (ROOT / c["done"]).exists():
                self.skipped.append(c["name"])
            else:
                self.q.put(c)
        print(f"[orch] {self.q.qsize()} pending, {len(self.skipped)} already done",
              flush=True)

    def write_state(self):
        with self.lock:
            state = {
                "ts": time.strftime("%F %T"),
                "pending": self.q.qsize(),
                "running": dict(self.running),
                "done": self.done + self.skipped,
                "failed": self.failed,
            }
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(state, indent=1))

    def worker(self, gpu: int):
        while True:
            try:
                cell = self.q.get(timeout=30)
            except queue.Empty:
                return
            name = cell["name"]
            if Path(cell["done"]).exists() or (ROOT / cell["done"]).exists():
                with self.lock:
                    self.done.append(name)
                continue
            need = float(cell.get("min_free_gb", 30))
            ok = False
            for _ in range(VRAM_POLL_TRIES):
                if free_gb(gpu) >= need:
                    ok = True
                    break
                time.sleep(60)
            if not ok:
                print(f"[gpu{gpu}] {name}: <{need}GB free, requeue + backoff",
                      flush=True)
                vram_snapshot(f"gate-refused gpu{gpu} need={need}")
                self.q.put(cell)
                self.write_state()
                time.sleep(BACKOFF_S)
                continue
            with self.lock:
                self.running[name] = gpu
            self.write_state()
            print(f"[gpu{gpu}] LAUNCH {name}", flush=True)
            t0 = time.time()
            if self.dry:
                rc = 0
            else:
                env = dict(os.environ,
                           CUDA_VISIBLE_DEVICES=str(gpu),
                           PYTHONNOUSERSITE="1",
                           OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1",
                           TOKENIZERS_PARALLELISM="false")
                LOGDIR.mkdir(parents=True, exist_ok=True)
                with open(LOGDIR / f"{name}.log", "a") as lf:
                    rc = subprocess.run(
                        cell["cmd"], shell=True, cwd=ROOT, env=env,
                        stdout=lf, stderr=subprocess.STDOUT).returncode
            dt = (time.time() - t0) / 60
            gate_backoff = False
            with self.lock:
                self.running.pop(name, None)
                if rc == 0 and (Path(cell["done"]).exists()
                                or (ROOT / cell["done"]).exists() or self.dry):
                    self.done.append(name)
                    print(f"[gpu{gpu}] DONE {name} ({dt:.0f} min)", flush=True)
                elif rc == 87:
                    print(f"[gpu{gpu}] GATE {name} rc=87 ({dt:.0f} min) "
                          f"-- no-charge requeue + backoff", flush=True)
                    self.q.put(cell)
                    gate_backoff = True
                else:
                    n = self.attempts.get(name, 0) + 1
                    self.attempts[name] = n
                    if n < 2:
                        print(f"[gpu{gpu}] FAIL {name} rc={rc} ({dt:.0f} min) "
                              f"-- retry queued", flush=True)
                        self.q.put(cell)
                    else:
                        self.failed.append(name)
                        print(f"[gpu{gpu}] FAIL {name} rc={rc} -- parked",
                              flush=True)
            self.write_state()
            if gate_backoff:
                time.sleep(BACKOFF_S)

    def run(self):
        threads = [threading.Thread(target=self.worker, args=(g,), daemon=True)
                   for g in self.gpus]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.write_state()
        sentinel = os.environ.get("ORCH_SENTINEL", "_QUEUE_COMPLETE")
        if not self.failed:
            (ROOT / sentinel).write_text(time.strftime("%F %T\n"))
            print("[orch] queue complete", flush=True)
        else:
            (ROOT / "_QUEUE_FAILED").write_text("\n".join(self.failed) + "\n")
            print(f"[orch] finished with {len(self.failed)} failed cells",
                  flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cells = json.loads(Path(args.manifest).read_text())
    gpus = [int(g) for g in os.environ.get("GPUS", "0").split(",")]
    print(f"[orch] GPUs: {gpus}; manifest: {args.manifest} "
          f"({len(cells)} cells)", flush=True)
    vram_snapshot("job-start")
    Orchestrator(cells, gpus, args.dry_run).run()


if __name__ == "__main__":
    main()
