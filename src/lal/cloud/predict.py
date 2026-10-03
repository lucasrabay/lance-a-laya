"""Laya inference on Modal: zero-shot or fine-tuned predictions, calibration fitting, latency benchmark.

    modal run -m lal.cloud.predict::main --model zeroshot --splits val          # -> data/preds/zeroshot/
    modal run -m lal.cloud.predict::main --model ft-v1 --calibration ft-v1 --splits val,test
    modal run -m lal.cloud.predict::bench --model ft-v1                          # T4 + L4 latency

`model` is "zeroshot" (laya-multilingual as shipped, pinned in the image) or a run name under the
Volume's ckpt/. All five questions are answered in one forward pass per window.
"""


import json
import time

import modal

VOLUME_NAME = "lal-data"
LAYA_VERSION = "0.3.24"
BASE_DIR = "/models/laya"
BASE_SUBFOLDER = "multilingual"


def _download_base() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download("convaiinnovations/laya", local_dir=BASE_DIR,
                      allow_patterns=[f"{BASE_SUBFOLDER}/*", f"{BASE_SUBFOLDER}/*/*"])


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(f"laya=={LAYA_VERSION}", "huggingface_hub>=0.25", "pyyaml")
    .run_function(_download_base)
    .add_local_python_source("lal")
)
app = modal.App("lal-predict", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


def model_path(model: str) -> str:
    return f"{BASE_DIR}/{BASE_SUBFOLDER}" if model == "zeroshot" else f"/data/ckpt/{model}"


@app.cls(gpu="T4", volumes={"/data": volume}, timeout=3600, scaledown_window=120)
class Predictor:
    model: str = modal.parameter(default="zeroshot")
    calibration: str = modal.parameter(default="")

    @modal.enter()
    def load(self):
        import laya

        volume.reload()
        cal = f"/data/calib/{self.calibration}.json" if self.calibration else None
        self.agent = laya.load(model_path(self.model), device="cuda", calibration=cal)
        self.gpu = self._gpu_name()

    @staticmethod
    def _gpu_name() -> str:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    @modal.method()
    def predict(self, states: list, questions: dict, batch_size: int = 64) -> list[dict]:
        results = self.agent.predict_batch(states, questions, batch_size=batch_size, sort_by_length=True)
        return [r["answers"] for r in results]

    @modal.method()
    def latency(self, states: list, questions: dict, n: int = 200) -> dict:
        """Per-window latency at batch size 1 (streaming) and batched throughput, after warm-up."""
        import statistics

        import torch

        for s in states[:10]:
            self.agent.predict(s, questions)
        times = []
        for s in states[:n]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self.agent.predict(s, questions)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        times.sort()
        t0 = time.perf_counter()
        self.agent.predict_batch(states, questions, batch_size=64, sort_by_length=True)
        torch.cuda.synchronize()
        batch_s = time.perf_counter() - t0
        return {
            "gpu": self.gpu, "model": self.model, "questions": len(questions), "n_single": len(times),
            "p50_ms": round(statistics.median(times), 1), "p90_ms": round(times[int(0.9 * len(times)) - 1], 1),
            "mean_ms": round(statistics.fmean(times), 1),
            "batched_windows": len(states), "batched_windows_per_s": round(len(states) / batch_s, 1),
        }

    @modal.method()
    def fit_calibration(self, pairs: list, name: str) -> dict:
        """pairs: [(state, questions, targets)] with targets[qid] a probability vector in option order."""
        from laya.calibrate import records_from_labeled

        records = records_from_labeled(self.agent, pairs)
        fit = self.agent.fit_temperatures(records, compute_ece=True)
        import os

        os.makedirs("/data/calib", exist_ok=True)
        self.agent.save_calibration(f"/data/calib/{name}.json")
        volume.commit()
        return json.loads(json.dumps(fit, default=str))


def _windows_for(splits: list[str], match: str = ""):
    from lal.config import load_matches
    from lal.windows import load_windows

    for m in load_matches():
        if m["split"] in splits and (not match or m["id"] == match):
            yield m["id"], load_windows(m["id"])


@app.local_entrypoint()
def main(model: str = "zeroshot", calibration: str = "", splits: str = "val", match: str = "",
         schema: str = "en", tag: str = ""):
    from lal.config import repo_path
    from lal.schema import questions_for, state_for

    questions = questions_for(schema)
    pred = Predictor(model=model, calibration=calibration)
    out_dir = repo_path(f"data/preds/{tag or model}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for mid, wins in _windows_for(splits.split(","), match):
        t0 = time.time()
        answers = pred.predict.remote([state_for(w["text"]) for w in wins], questions)
        with open(out_dir / f"{mid}.jsonl", "w") as f:
            for w, a in zip(wins, answers):
                f.write(json.dumps({"wid": w["wid"], "answers": a}) + "\n")
        print(f"{mid}: {len(wins)} windows in {time.time() - t0:.1f}s -> {out_dir}/{mid}.jsonl")


@app.local_entrypoint()
def bench(model: str = "zeroshot", calibration: str = "", match: str = "bra-sui-2022"):
    from lal.config import repo_path
    from lal.schema import QUESTIONS, state_for

    wins = dict(_windows_for(["val"], match))[match]
    states = [state_for(w["text"]) for w in wins]
    rows = []
    for gpu in ("T4", "L4"):
        res = Predictor.with_options(gpu=gpu)(model=model, calibration=calibration).latency.remote(states, QUESTIONS)
        print(json.dumps(res))
        rows.append(res)
    out = repo_path(f"outputs/latency_{model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))


@app.local_entrypoint()
def calib(model: str):
    """Fit Laya temperatures on the validation matches -> Volume calib/{model}.json."""
    from lal.calibrate import calibration_pairs
    from lal.config import repo_path

    pairs = calibration_pairs("val")
    fit = Predictor(model=model).fit_calibration.remote(pairs, model)
    print(json.dumps(fit, indent=1)[:2000])
    out = repo_path(f"outputs/laya_temperatures_{model}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(fit, indent=1))
