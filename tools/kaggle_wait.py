"""Poll `kaggle kernels status <owner/slug>` until the run reaches a terminal state.

Usage:
    python tools/kaggle_wait.py <owner/slug> [--timeout-min 330] [--interval-s 180]

Exit codes: 0 complete, 1 error/cancelled, 2 timeout, 3 repeated CLI failures.
Handy to run in the background while a long Kaggle job is running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path


def kaggle_exe() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    for name in ("kaggle.exe", "kaggle"):
        candidate = Path(sys.executable).parent / name
        if candidate.exists():
            return str(candidate)
    return "kaggle"


def classify(text: str) -> str:
    t = text.lower()
    if "complete" in t:
        return "complete"
    if "cancel" in t:
        return "cancelled"
    if "error" in t or "fail" in t:
        return "error"
    if "running" in t:
        return "running"
    if "queued" in t or "pending" in t or "new_script" in t:
        return "queued"
    return "unknown"


def now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel", help="owner/slug")
    parser.add_argument("--timeout-min", type=float, default=330)
    parser.add_argument("--interval-s", type=float, default=180)
    args = parser.parse_args()

    exe = kaggle_exe()
    deadline = time.time() + args.timeout_min * 60
    last_state, cli_failures, polls = None, 0, 0
    while True:
        result = subprocess.run([exe, "kernels", "status", args.kernel], capture_output=True, text=True)
        text = ((result.stdout or "") + " " + (result.stderr or "")).strip()
        polls += 1
        if result.returncode != 0:
            cli_failures += 1
            print(f"[{now()}] CLI failure {cli_failures}: {text[:300]}", flush=True)
            if cli_failures >= 5:
                return 3
        else:
            cli_failures = 0
            state = classify(text)
            if state != last_state or polls % 10 == 0:
                print(f"[{now()}] {args.kernel}: {state} | {text[:200]}", flush=True)
                last_state = state
            if state == "complete":
                return 0
            if state in ("error", "cancelled"):
                return 1
        if time.time() >= deadline:
            print(f"[{now()}] timeout after {args.timeout_min} min (last state: {last_state})", flush=True)
            return 2
        time.sleep(args.interval_s)


if __name__ == "__main__":
    sys.exit(main())
