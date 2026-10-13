"""Kaggle stage runner: orchestrates GPU subprocesses for the setup / sft / dpo stages.

    python -m pa2.stage --stage setup|sft|dpo [--only task,task] [--skip-done]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOGS = OUT / "logs"
PY = sys.executable

ALPACA = "eval_sets/alpaca_eval_300_seed42.json"
HH = "eval_sets/hh_rlhf_eval_300_seed42.json"
PERSONA = "eval_sets/persona_style_eval.json"
GEN = ROOT / "outputs" / "generations"


def run(cmd, gpu: int, name: str, env_extra=None) -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env.update(env_extra or {})
    log = LOGS / f"{name}.log"
    print(f"[stage] ({name}) gpu{gpu}: {' '.join(cmd)}", flush=True)
    with log.open("w", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
        return p.wait()


def tail(path: Path, n=3) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def heartbeat(stop: threading.Event, interval=300):
    while not stop.wait(interval):
        for lg in sorted(LOGS.glob("*.log")):
            if time.time() - lg.stat().st_mtime < interval * 2:
                print(f"[hb] {lg.name}:\n{tail(lg)}", flush=True)


class Stage:
    def __init__(self, name, only, skip_done):
        self.name = name
        self.only = set(only.split(",")) if only else None
        self.skip_done = skip_done
        self.timings = {}
        self.lock = threading.Lock()

    def task(self, task_name, fn, done_check=None):
        """Run fn() (returns nothing); never raises."""
        if self.only and task_name not in self.only:
            return
        if self.skip_done and done_check and done_check():
            print(f"[stage] skip {task_name} (done)", flush=True)
            return
        t0 = time.time()
        try:
            fn()
            status = "ok"
        except Exception:
            status = "FAILED"
            tb = traceback.format_exc()
            (OUT).mkdir(parents=True, exist_ok=True)
            lg = LOGS / f"{task_name}.log"
            (OUT / f"STAGE_{self.name}_{task_name}_FAILED.txt").write_text(
                tb + "\n--- log tail ---\n" + tail(lg, 40), encoding="utf-8"
            )
            print(f"[stage] {task_name} FAILED\n{tb}", flush=True)
        with self.lock:
            self.timings[task_name] = {"status": status, "seconds": round(time.time() - t0, 1)}

    def finish(self):
        (OUT / f"STAGE_{self.name}_DONE.json").write_text(json.dumps(self.timings, indent=2), encoding="utf-8")
        zpath = OUT / f"{self.name}_small_outputs.zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for p in OUT.rglob("*"):
                if p.is_file() and p != zpath and p.suffix not in {".safetensors", ".bin", ".pt"} and "/data/" not in p.as_posix() and p.stat().st_size < 50e6:
                    z.write(p, p.relative_to(OUT).as_posix())
        print(f"[stage] {self.name} finished: {json.dumps(self.timings)}", flush=True)


def need(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what} exited with code {rc}")


# ----------------------------------------------------------------------------- #
def stage_setup(st: Stage):
    def tests():
        need(run([PY, "-m", "pytest", "-q", "tests"], 0, "pytest"), "pytest")

    def build_data():
        need(run([PY, "-m", "pa2.data", "--which", "eval,sft,hh,uf", "--manifest_dir", "outputs/manifests"], 0, "data_build"), "data build")

    def probe():
        need(run([PY, "-m", "pa2.probe", "--out", "outputs/probe.json"], "0,1", "probe"), "probe")

    def sanity():
        need(
            run([PY, "-m", "pa2.generate", "--model", "Qwen/Qwen2.5-0.5B", "--generator", "qwen2.5-0.5b-base",
                 "--eval_set", ALPACA, HH, "--out", "outputs/sanity_base_alpaca.json", "outputs/sanity_base_hh.json",
                 "--limit", "3", "--max_new_tokens", "128", "--batch_size", "3"], 1, "sanity_base"),
            "sanity generation",
        )

    st.task("pytest", tests)
    st.task("probe", probe)  # alone, so peak memory is not disturbed by other work
    th = []
    for fn, nm in ((build_data, "data_build"), (sanity, "sanity")):
        t = threading.Thread(target=st.task, args=(nm, fn))
        t.start()
        th.append(t)
    for t in th:
        t.join()

    # tiny end-to-end smoke of every training path on the real target hardware
    def smoke_sft():
        need(run([PY, "-m", "pa2.train_sft", "--config", "configs/smoke_sft.yaml", "precision=fp16",
                  "optimizer=adamw", "max_length=512", "max_steps=6"], 0, "smoke_sft"), "smoke sft")
        need(run([PY, "-m", "pa2.generate", "--model", "outputs/smoke/sft/best", "--generator", "smoke-sft",
                  "--eval_set", ALPACA, "--out", "outputs/smoke/gen_sft.json", "--limit", "3",
                  "--max_new_tokens", "48", "--batch_size", "3"], 0, "smoke_gen_sft"), "smoke gen")

    def smoke_dpo():
        # wait for the smoke SFT checkpoint (GPU0) to exist
        t0 = time.time()
        while not Path("outputs/smoke/sft/best/config.json").exists() and time.time() - t0 < 900:
            time.sleep(5)
        need(run([PY, "-m", "pa2.train_dpo", "--config", "configs/smoke_dpo.yaml", "precision=fp16",
                  "ref_dtype=fp16", "optimizer=rmsprop", "max_length=512"], 1, "smoke_dpo"), "smoke dpo")

    def smoke_lora():
        need(run([PY, "-m", "pa2.train_lora_dpo", "--config", "configs/lora_persona.yaml",
                  "out_dir=outputs/smoke/lora", "max_steps=3", "epochs=1"], 1, "smoke_lora"), "smoke lora")
        need(run([PY, "-m", "pa2.generate", "--model", "HuggingFaceTB/SmolLM2-360M-Instruct",
                  "--adapter", "outputs/smoke/lora/adapter", "--generator", "smoke-lora", "--eval_set", PERSONA,
                  "--out", "outputs/smoke/gen_lora.json", "--limit", "3", "--max_new_tokens", "48"], 1,
                 "smoke_gen_lora"), "smoke lora gen")

    th = [threading.Thread(target=st.task, args=(nm, fn)) for fn, nm in
          ((smoke_sft, "smoke_sft"), (smoke_dpo, "smoke_dpo"))]
    [t.start() for t in th]
    [t.join() for t in th]
    st.task("smoke_lora", smoke_lora)


def stage_sft(st: Stage):
    def sft():
        need(run([PY, "-m", "pa2.train_sft", "--config", "configs/sft.yaml"], 0, "sft"), "sft")

    def sft_gen():
        need(
            run([PY, "-m", "pa2.generate", "--model", "outputs/sft/best", "--generator", "qwen2.5-0.5b-sft",
                 "--eval_set", ALPACA, HH, "--out", "outputs/generations/sft_alpaca.json",
                 "outputs/generations/sft_hh.json"], 0, "gen_sft"),
            "sft generation",
        )

    def base_gen():
        need(
            run([PY, "-m", "pa2.generate", "--model", "Qwen/Qwen2.5-0.5B", "--generator", "qwen2.5-0.5b-base",
                 "--eval_set", ALPACA, HH, "--out", "outputs/generations/base_alpaca.json",
                 "outputs/generations/base_hh.json"], 1, "gen_base"),
            "base generation",
        )

    def lora():
        need(run([PY, "-m", "pa2.train_lora_dpo", "--config", "configs/lora_persona.yaml"], 1, "lora"), "lora")

    def persona_gen():
        need(
            run([PY, "-m", "pa2.generate", "--model", "HuggingFaceTB/SmolLM2-360M-Instruct",
                 "--generator", "smollm2-360m-instruct", "--eval_set", PERSONA,
                 "--out", "outputs/generations/persona_base.json", "--max_new_tokens", "256"], 1, "gen_persona_base"),
            "persona base generation",
        )
        need(
            run([PY, "-m", "pa2.generate", "--model", "HuggingFaceTB/SmolLM2-360M-Instruct",
                 "--adapter", "outputs/lora_persona/adapter", "--generator", "smollm2-360m-lora-dpo",
                 "--eval_set", PERSONA, "--out", "outputs/generations/persona_lora.json",
                 "--max_new_tokens", "256"], 1, "gen_persona_lora"),
            "persona lora generation",
        )

    def gpu0():
        st.task("sft", sft, lambda: Path("outputs/sft/final/config.json").exists())
        st.task("gen_sft", sft_gen, lambda: Path("outputs/generations/sft_hh.json").exists())

    def gpu1():
        st.task("gen_base", base_gen, lambda: Path("outputs/generations/base_hh.json").exists())
        st.task("lora", lora, lambda: Path("outputs/lora_persona/adapter").exists())
        st.task("gen_persona", persona_gen, lambda: Path("outputs/generations/persona_lora.json").exists())

    ts = [threading.Thread(target=gpu0), threading.Thread(target=gpu1)]
    [t.start() for t in ts]
    [t.join() for t in ts]


def find_sft_best() -> str:
    cands = glob.glob("/kaggle/input/**/outputs/sft/best/config.json", recursive=True)
    if not cands:
        cands = glob.glob(str(OUT / "sft" / "best" / "config.json"))
    if not cands:
        raise RuntimeError("SFT best checkpoint not found under /kaggle/input")
    return str(Path(cands[0]).parent)


def stage_dpo(st: Stage):
    holder = {}

    def sft_path():
        if "p" not in holder:
            holder["p"] = find_sft_best()
            print(f"[stage] SFT best = {holder['p']}", flush=True)
        return holder["p"]

    def track(tag, cfg, gpu, eval_set, gen_name, gen_out):
        def train():
            need(run([PY, "-m", "pa2.train_dpo", "--config", cfg, f"sft_model={sft_path()}"], gpu, f"dpo_{tag}"),
                 f"dpo_{tag}")

        def gen():
            need(
                run([PY, "-m", "pa2.generate", "--model", f"outputs/dpo_{tag}/best", "--generator", gen_name,
                     "--eval_set", eval_set, "--out", gen_out], gpu, f"gen_dpo_{tag}"),
                f"gen dpo_{tag}",
            )

        st.task(f"dpo_{tag}", train, lambda: Path(f"outputs/dpo_{tag}/final/config.json").exists())
        st.task(f"gen_dpo_{tag}", gen, lambda: Path(gen_out).exists())

    ts = [
        threading.Thread(target=track, args=("hh", "configs/dpo_hh.yaml", 0, HH, "qwen2.5-0.5b-dpo-hh",
                                             "outputs/generations/dpo_hh_hh.json")),
        threading.Thread(target=track, args=("uf", "configs/dpo_uf.yaml", 1, ALPACA, "qwen2.5-0.5b-dpo-uf",
                                             "outputs/generations/dpo_uf_alpaca.json")),
    ]
    [t.start() for t in ts]
    [t.join() for t in ts]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["setup", "sft", "dpo"])
    ap.add_argument("--only", default="")
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    st = Stage(a.stage, a.only, a.skip_done)
    stop = threading.Event()
    threading.Thread(target=heartbeat, args=(stop,), daemon=True).start()
    try:
        {"setup": stage_setup, "sft": stage_sft, "dpo": stage_dpo}[a.stage](st)
    except Exception:
        (OUT / f"STAGE_{a.stage}_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc()
    finally:
        stop.set()
        st.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
