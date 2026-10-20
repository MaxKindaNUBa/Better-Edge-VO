"""One results folder per run of a test script.

    results/<YYYY-MM-DD_HH-MM-SS>_<test name>/
        run_info.json     what was run: command, options, sequences, start / end time,
                          and per sequence whether it finished, how long it took, any error
        log.txt           everything the script printed (console output, tracebacks)
        <sequence>/       that sequence's outputs (same files as before)

    with RunDir("run_dtvo", args, seqs) as run:
        for s in seqs:
            with run.sequence(s):
                ...  write into run.dir / s.name
"""
from __future__ import annotations

import json
import platform
import shlex
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


class _Tee:
    """Writes to the console and to the log file at the same time."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        self.stream.write(s)
        self.log.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):   # isatty, fileno, encoding, ... of the real console
        return getattr(self.stream, name)


def _command() -> str:
    """The command line, as it can be pasted back into a shell at the project root."""
    script = Path(sys.argv[0]).resolve()
    try:
        rel = script.relative_to(PROJECT_ROOT).with_suffix("")
        head = ["python", "-m", ".".join(rel.parts)]
    except ValueError:
        head = ["python", str(script)]
    return shlex.join(head + sys.argv[1:])


def _stamp(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="seconds")


class RunDir:
    def __init__(self, test_name: str, args, sequences: list[Path], root: Path | None = None):
        self.t0 = time.time()
        root = root or RESULTS_DIR
        base = f"{datetime.fromtimestamp(self.t0):%Y-%m-%d_%H-%M-%S}_{test_name}"
        self.dir = root / base
        k = 2
        while self.dir.exists():   # two runs started in the same second
            self.dir = root / f"{base}_{k}"
            k += 1
        self.dir.mkdir(parents=True)
        self.info = {
            "test": test_name,
            "command": _command(),
            "options": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                        if k != "sequence"},
            "started": _stamp(self.t0), "finished": None, "duration_s": None, "status": "running",
            "machine": platform.node(), "python": platform.python_version(),
            "sequences": {s.name: {"path": str(s), "status": "pending"} for s in sequences},
        }
        self._log = None
        self._streams = None

    def __enter__(self) -> "RunDir":
        self._log = open(self.dir / "log.txt", "w", buffering=1)
        self._streams = sys.stdout, sys.stderr
        sys.stdout = _Tee(sys.stdout, self._log)
        sys.stderr = _Tee(sys.stderr, self._log)
        print(f"run folder: {self.dir}", flush=True)
        self.save()
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            traceback.print_exception(exc_type, exc, tb)   # into log.txt as well
        t1 = time.time()
        self.info.update(finished=_stamp(t1), duration_s=round(t1 - self.t0, 1),
                         status="interrupted" if exc_type is KeyboardInterrupt
                         else "error" if exc_type is not None
                         else "finished with failures" if self.failed else "finished")
        self.save()
        print(f"run folder: {self.dir}  ({self.info['status']})", flush=True)
        sys.stdout, sys.stderr = self._streams
        self._log.close()
        return False

    @property
    def failed(self) -> list[str]:
        return [n for n, s in self.info["sequences"].items() if s["status"] == "failed"]

    def save(self):
        (self.dir / "run_info.json").write_text(json.dumps(self.info, indent=2, default=str))

    @contextmanager
    def sequence(self, seq_dir: Path):
        """Records how one sequence went. A failure is logged and the batch carries on."""
        rec = self.info["sequences"][seq_dir.name]
        t = time.time()
        rec.update(status="running", started=_stamp(t))
        self.save()
        try:
            yield self.dir / seq_dir.name
            rec["status"] = "done"
        except Exception as e:   # one broken sequence must not stop an overnight batch
            traceback.print_exc()
            print(f"!! {seq_dir.name} failed: {type(e).__name__}: {e}", flush=True)
            rec.update(status="failed", error=f"{type(e).__name__}: {e}")
        except BaseException:   # Ctrl+C
            rec["status"] = "interrupted"
            raise
        finally:
            rec["duration_s"] = round(time.time() - t, 1)
            self.save()
