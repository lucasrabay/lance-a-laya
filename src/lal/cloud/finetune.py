"""Fine-tune laya-multilingual on Modal with the (vendored) official single-device RLCD script.

    modal run -m lal.cloud.finetune::main --run dry --limit 50 --epochs 1 --gpu T4   # format dry run
    modal run -m lal.cloud.finetune::main --run ft-v1 --epochs 3                      # A100-40GB

Uploads data/datasets/train.jsonl to the Volume, trains, and writes a laya.load()-able checkpoint to
ckpt/{run}/ on the Volume. Calibration is refit afterwards on the validation matches (lal.calibrate);
the script's own temperatures (fit on a slice of the training items) are not used for reporting.
"""

import json
import time

import modal

from lal.cloud.predict import BASE_DIR, BASE_SUBFOLDER, VOLUME_NAME, image

app = modal.App("lal-finetune", image=image)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(gpu="A100", volumes={"/data": volume}, timeout=4 * 3600, memory=16384)
def train(run: str, data: str = "datasets/train.jsonl", epochs: int = 3, max_len: int = 512,
          micro_batch: int = 8, limit: int = 0, seed: int = 0) -> dict:
    import subprocess

    volume.reload()
    path = f"/data/{data}"
    if limit:
        with open(path) as f:
            lines = [next(f) for _ in range(limit)]
        path = "/tmp/subset.jsonl"
        with open(path, "w") as f:
            f.writelines(lines)
    out = f"/data/ckpt/{run}"
    cmd = ["python", "-u", "-m", "lal.cloud.vendor.laya_finetune_single_device", "--data", path,
           "--model-dir", f"{BASE_DIR}/{BASE_SUBFOLDER}", "--output-dir", out, "--epochs", str(epochs),
           "--max-len", str(max_len), "--micro-batch", str(micro_batch), "--device", "cuda", "--seed", str(seed)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"training failed:\n{log[-4000:]}")
    meta = {"run": run, "data": data, "limit": limit, "epochs": epochs, "max_len": max_len,
            "micro_batch": micro_batch, "seed": seed, "train_s": round(time.time() - t0, 1),
            "log_tail": [l for l in log.splitlines() if l.startswith(("Train items", "Epoch", "Fitted", "Saved"))]}
    with open(f"{out}/train_meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    volume.commit()
    return meta


@app.local_entrypoint()
def main(run: str = "ft-v1", epochs: int = 3, max_len: int = 512, micro_batch: int = 8, limit: int = 0,
         gpu: str = "A100", seed: int = 0):
    from lal.config import repo_path

    local = repo_path("data/datasets/train.jsonl")
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local), "datasets/train.jsonl")
    print(f"uploaded {local} ({local.stat().st_size / 1e6:.1f} MB)")
    meta = train.with_options(gpu=gpu).remote(run, epochs=epochs, max_len=max_len, micro_batch=micro_batch,
                                              limit=limit, seed=seed)
    print(json.dumps(meta, indent=1))
    repo_path("outputs").mkdir(exist_ok=True)
    (repo_path("outputs") / f"train_{run}.json").write_text(json.dumps(meta, indent=1))
