"""Reproducible UCI HAR experiment. Run: python har.py --help.

All model selection uses validation participants only. Test evaluation is a
separate command and requires a sealed manifest of all eight fitted models.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import random
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
CONFIG_PATH = ROOT / "config.json"
CLASSES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
CN = ["步行", "上楼", "下楼", "坐姿", "站姿", "躺卧"]
EXPERIMENTS = {"A": ("logistic", 3), "B": ("logistic", 6), "C": ("cnn", 3), "D": ("cnn", 6)}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def config():
    return read_json(CONFIG_PATH)


def prepare():
    cfg = config()
    raw = DATA / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "uci_har_official.zip"
    if not archive.exists():
        print("Downloading official archive", flush=True)
        req = urllib.request.Request(cfg["dataset_url"], headers={"User-Agent": "HAR-course-research/1.0"})
        part = archive.with_suffix(".partial")
        with urllib.request.urlopen(req, timeout=120) as response, part.open("wb") as dest:
            while chunk := response.read(1024 * 1024):
                dest.write(chunk)
        part.replace(archive)
    digest = sha256(archive)
    manifest_path = raw / "source.json"
    if manifest_path.exists():
        assert read_json(manifest_path)["sha256"] == digest, "Raw archive checksum changed"
    else:
        write_json(manifest_path, {"url": cfg["dataset_url"], "sha256": digest,
            "retrieved_utc": utc_now(), "bytes": archive.stat().st_size,
            "checksum_note": "SHA-256 calculated locally; not an independently published UCI checksum.",
            "license": "CC BY 4.0", "doi": "10.24432/C54S4K"})
    with zipfile.ZipFile(archive) as outer:
        bad = outer.testzip()
        assert bad is None, f"Corrupt member: {bad}"
        inner = next((n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip")), None)
        z = zipfile.ZipFile(io.BytesIO(outer.read(inner))) if inner else outer
        def member(suffix):
            names = [n for n in z.namelist() if n.endswith(suffix) and not n.startswith("__MACOSX")]
            assert len(names) == 1, (suffix, names)
            return names[0]
        for name in ["README.txt", "features_info.txt", "activity_labels.txt"]:
            (raw / name).write_bytes(z.read(member("UCI HAR Dataset/" + name)))
        arrays = {}
        for split in ["train", "test"]:
            X = np.stack([np.loadtxt(io.BytesIO(z.read(member(f"{split}/Inertial Signals/{c}_{split}.txt"))), dtype=np.float32)
                          for c in cfg["channels"]], axis=1)
            y = np.loadtxt(io.BytesIO(z.read(member(f"{split}/y_{split}.txt"))), dtype=np.int64) - 1
            subjects = np.loadtxt(io.BytesIO(z.read(member(f"{split}/subject_{split}.txt"))), dtype=np.int64)
            assert X.shape == (len(y), 6, 128)
            assert np.isfinite(X).all() and set(np.unique(y)) == set(range(6))
            arrays[split] = (X, y, subjects)
        X, y, subjects = arrays["train"]
        people = np.unique(subjects)
        val_people = np.sort(np.random.default_rng(cfg["split_seed"]).choice(people, cfg["validation_subjects"], replace=False))
        val_mask = np.isin(subjects, val_people)
        parts = {"train": (X[~val_mask], y[~val_mask], subjects[~val_mask], np.flatnonzero(~val_mask) + 1),
                 "val": (X[val_mask], y[val_mask], subjects[val_mask], np.flatnonzero(val_mask) + 1),
                 "test": (*arrays["test"], np.arange(1, len(arrays["test"][1]) + 1))}
        assert sum(len(p[1]) for p in parts.values()) == 10299
        subject_sets = {k: set(v[2].tolist()) for k, v in parts.items()}
        assert not (subject_sets["train"] & subject_sets["val"] or subject_sets["train"] & subject_sets["test"] or subject_sets["val"] & subject_sets["test"])
        processed = DATA / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        manifest = {"split_seed": cfg["split_seed"], "selection_algorithm": "numpy.random.default_rng(seed).choice(sorted_subjects, 4, replace=False)",
                    "raw_sha256": digest, "channels": cfg["channels"], "classes": CLASSES, "splits": {}}
        for name, (xx, yy, ss, rows) in parts.items():
            np.savez_compressed(processed / f"{name}.npz", X=xx, y=yy, subjects=ss, source_rows=rows)
            manifest["splits"][name] = {"n": len(yy), "subjects": sorted(subject_sets[name]),
                "class_counts": np.bincount(yy, minlength=6).tolist(), "sha256": sha256(processed / f"{name}.npz")}
        write_json(DATA / "split_manifest.json", manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def load_split(name):
    with np.load(DATA / "processed" / f"{name}.npz") as z:
        return {k: z[k] for k in z.files}


def features(X):
    """Seven statistics per channel, flattened in statistic-major order."""
    x = X.astype(np.float64)
    return np.concatenate([x.mean(2), x.std(2, ddof=0), x.min(2), x.max(2),
        np.median(x, axis=2), np.percentile(x, 75, axis=2) - np.percentile(x, 25, axis=2),
        np.sqrt(np.mean(x * x, axis=2))], axis=1)


class TinyCNN(nn.Module):
    def __init__(self, channels, spec):
        super().__init__()
        layers = []
        for i, (width, kernel) in enumerate(zip(spec["widths"], spec["kernels"])):
            layers.extend([nn.Conv1d(channels, width, kernel, padding=kernel // 2), nn.ReLU()])
            if i < 2:
                layers.append(nn.MaxPool1d(2))
            channels = width
        layers.extend([nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(spec["dropout"]), nn.Linear(channels, 6)])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def metrics(y, pred):
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(6), zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)), "macro_f1": float(f.mean()),
            "per_class": [{"class": CLASSES[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])} for i in range(6)],
            "confusion_matrix": confusion_matrix(y, pred, labels=range(6)).tolist()}


def predict_tensor(model, X):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(chunk) for chunk in X.split(256)])


def train():
    cfg = config()
    if (OUT / "frozen_manifest.json").exists() or (OUT / "results.json").exists():
        raise RuntimeError("This experiment is already sealed. Use a new project copy for a new protocol.")
    OUT.mkdir(exist_ok=True)
    train_set, val_set = load_split("train"), load_split("val")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_json(OUT / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__, "sklearn": sklearn.__version__,
        "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "started_utc": utc_now()})
    trained = []
    for exp, (kind, channels) in EXPERIMENTS.items():
        seeds = [42] if kind == "logistic" else cfg["seeds"]
        for seed in seeds:
            run_id = f"{exp}_seed{seed}"
            folder = OUT / "runs" / run_id
            folder.mkdir(parents=True, exist_ok=True)
            if (folder / "selection.json").exists():
                previous = read_json(folder / "selection.json")
                assert previous["config_sha256"] == sha256(CONFIG_PATH)
                trained.append(previous)
                continue
            set_seed(seed)
            start = time.perf_counter()
            x_train, x_val = train_set["X"][:, :channels], val_set["X"][:, :channels]
            y_train, y_val = train_set["y"], val_set["y"]
            selection = {"run_id": run_id, "experiment": exp, "method": kind, "channels": channels, "seed": seed,
                         "config_sha256": sha256(CONFIG_PATH)}
            print(f"START {run_id} on {device}", flush=True)
            if kind == "logistic":
                ft, fv = features(x_train), features(x_val)
                candidates = []
                best_score, best = -1, None
                for C in cfg["lr_C_candidates"]:
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="lbfgs", max_iter=3000, random_state=seed))
                    model.fit(ft, y_train)
                    score = f1_score(y_val, model.predict(fv), average="macro")
                    candidates.append({"C": C, "val_macro_f1": float(score)})
                    if score > best_score:
                        best_score, best = score, model
                        selection["best_C"] = C
                model_path = folder / "model.joblib"
                joblib.dump(best, model_path)
                selection.update(val_macro_f1=float(best_score), candidates=candidates)
            else:
                spec = cfg["cnn"]
                mean = x_train.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
                std = x_train.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(np.float32)
                std = np.maximum(std, 1e-8)
                np.savez(folder / "normalization.npz", mean=mean, std=std)
                xt = torch.from_numpy((x_train - mean) / std).to(device)
                xv = torch.from_numpy((x_val - mean) / std).to(device)
                yt = torch.from_numpy(y_train).to(device)
                yv = torch.from_numpy(y_val).to(device)
                model = TinyCNN(channels, spec).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=spec["learning_rate"])
                criterion = nn.CrossEntropyLoss()
                best_score, stale = -1.0, 0
                history = []
                model_path = folder / "model.pt"
                generator = torch.Generator().manual_seed(seed)
                for epoch in range(1, spec["max_epochs"] + 1):
                    model.train()
                    permutation = torch.randperm(len(xt), generator=generator).to(device)
                    loss_sum, correct = 0.0, 0
                    for index in permutation.split(spec["batch_size"]):
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(xt[index])
                        loss = criterion(logits, yt[index])
                        loss.backward()
                        optimizer.step()
                        loss_sum += loss.item() * len(index)
                        correct += (logits.argmax(1) == yt[index]).sum().item()
                    logits = predict_tensor(model, xv)
                    val_loss = criterion(logits, yv).item()
                    score = f1_score(y_val, logits.argmax(1).cpu().numpy(), average="macro")
                    history.append({"epoch": epoch, "train_loss": loss_sum / len(xt), "train_accuracy_online": correct / len(xt), "val_loss": val_loss, "val_macro_f1": float(score)})
                    if score > best_score + 1e-12:
                        best_score, stale = float(score), 0
                        selection["best_epoch"] = epoch
                        torch.save({"state_dict": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            "channels": channels, "spec": spec, "seed": seed, "best_epoch": epoch}, model_path)
                    else:
                        stale += 1
                    if epoch == 1 or epoch % 5 == 0 or stale >= spec["patience"]:
                        print(f"{run_id} epoch={epoch} val_f1={score:.5f} best={best_score:.5f} stale={stale}", flush=True)
                    if stale >= spec["patience"]:
                        break
                csv_write(folder / "history.csv", history)
                selection.update(val_macro_f1=best_score, epochs=len(history), parameters=sum(p.numel() for p in model.parameters()))
            selection.update(training_seconds=time.perf_counter()-start,
                model_path=str(model_path.relative_to(ROOT)).replace("\\", "/"), model_sha256=sha256(model_path))
            write_json(folder / "selection.json", selection)
            trained.append(selection)
            print(f"DONE {run_id}: val_f1={selection['val_macro_f1']:.5f}", flush=True)
    files = [CONFIG_PATH, DATA / "split_manifest.json", Path(__file__)]
    files += list((DATA / "processed").glob("*.npz"))
    files += list((OUT / "runs").glob("*/model.*")) + list((OUT / "runs").glob("*/normalization.npz"))
    files += list((OUT / "runs").glob("*/selection.json"))
    write_json(OUT / "frozen_manifest.json", {"sealed_utc": utc_now(), "selection": "Validation Macro-F1 only; test not evaluated during training",
        "files": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p) for p in files}, "runs": trained})
    print("SEALED: all model choices frozen before test evaluation", flush=True)


def verify_seal():
    manifest = read_json(OUT / "frozen_manifest.json")
    for relative, digest in manifest["files"].items():
        assert sha256(ROOT / relative) == digest, f"Frozen file changed: {relative}"
    return manifest


def inference(run, dataset, device):
    channels = run["channels"]
    x = dataset["X"][:, :channels]
    path = ROOT / run["model_path"]
    if run["method"] == "logistic":
        # Only load the project's own locally created joblib files.
        model = joblib.load(path)
        prob = model.predict_proba(features(x))
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = TinyCNN(channels, ckpt["spec"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        with np.load(path.parent / "normalization.npz") as norm:
            xt = torch.from_numpy((x - norm["mean"]) / norm["std"]).to(device)
        prob = predict_tensor(model, xt).softmax(1).cpu().numpy()
    return prob.argmax(1), prob


def evaluate():
    manifest = verify_seal()
    set_seed(42)
    test = load_split("test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results, table = [], []
    for run in manifest["runs"]:
        pred, prob = inference(run, test, device)
        result = {**run, **metrics(test["y"], pred)}
        results.append(result)
        rows = [{"test_row": int(test["source_rows"][i]), "subject": int(test["subjects"][i]),
            "true_label": int(test["y"][i])+1, "true_activity": CLASSES[test["y"][i]],
            "pred_label": int(pred[i])+1, "pred_activity": CLASSES[pred[i]], "confidence": float(prob[i].max()),
            **{f"p_{c}": float(prob[i,j]) for j,c in enumerate(CLASSES)}} for i in range(len(pred))]
        csv_write(OUT / "predictions" / f"{run['run_id']}.csv", rows)
        for person in np.unique(test["subjects"]):
            mask = test["subjects"] == person
            table.append({"run_id": run["run_id"], "subject": int(person), "n": int(mask.sum()),
                "accuracy": float(accuracy_score(test["y"][mask], pred[mask])),
                "macro_f1": float(f1_score(test["y"][mask], pred[mask], labels=list(range(6)), average="macro", zero_division=0))})
        print(run["run_id"], f"accuracy={result['accuracy']:.6f}", f"macro_f1={result['macro_f1']:.6f}", flush=True)
    aggregate = []
    for exp in EXPERIMENTS:
        group = [r for r in results if r["experiment"] == exp]
        aggregate.append({"experiment": exp, "method": group[0]["method"], "channels": group[0]["channels"], "runs": len(group),
            **{f"{m}_{stat}": float(getattr(np, stat)([r[m] for r in group], **({"ddof": 1} if stat=="std" and len(group)>1 else {})))
               for m in ["accuracy", "macro_f1"] for stat in ["mean", "std"]}})
    write_json(OUT / "results.json", {"evaluated_utc": utc_now(), "frozen_manifest_sha256": sha256(OUT / "frozen_manifest.json"), "runs": results, "aggregate": aggregate})
    csv_write(OUT / "summary.csv", aggregate)
    csv_write(OUT / "per_subject.csv", table)
    csv_write(OUT / "per_class.csv", [{"run_id": r["run_id"], **c} for r in results for c in r["per_class"]])


def validate():
    manifest = verify_seal()
    cfg, split = config(), read_json(DATA / "split_manifest.json")
    parts = {s: load_split(s) for s in ["train", "val", "test"]}
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(name)
    check("official sample counts", len(parts["train"]["y"])+len(parts["val"]["y"])==7352 and len(parts["test"]["y"])==2947)
    sets = [set(parts[s]["subjects"]) for s in parts]
    check("participant isolation", all(not sets[i]&sets[j] for i in range(3) for j in range(i)))
    check("17/4/9 participants", [len(s) for s in sets]==[17,4,9])
    check("six classes in every partition", all(set(p["y"])==set(range(6)) for p in parts.values()))
    results = read_json(OUT / "results.json")
    expected_ids = {"A_seed42", "B_seed42", *(f"{e}_seed{s}" for e in "CD" for s in cfg["seeds"])}
    check("complete eight runs", {r["run_id"] for r in results["runs"]} == expected_ids)
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for run in results["runs"]:
        folder = ROOT / run["model_path"]
        c = run["channels"]
        x = parts["train"]["X"][:, :c]
        if run["method"] == "cnn":
            with np.load(folder.parent / "normalization.npz") as z:
                check(run["run_id"]+" train-only normalization", np.allclose(z["mean"],x.mean((0,2),keepdims=True,dtype=np.float64),rtol=1e-5,atol=1e-7) and np.allclose(z["std"],x.std((0,2),keepdims=True,dtype=np.float64),rtol=1e-5,atol=1e-7))
            with (folder.parent / "history.csv").open(encoding="utf-8-sig") as f:
                history = list(csv.DictReader(f))
            best = max(float(row["val_macro_f1"]) for row in history)
            check(run["run_id"]+" best validation checkpoint", abs(best-run["val_macro_f1"])<1e-12 and float(history[run["best_epoch"]-1]["val_macro_f1"])==best)
        else:
            model = joblib.load(folder)
            check(run["run_id"]+" train-only feature scaling", np.allclose(model[0].mean_,features(x).mean(0)))
            check(run["run_id"]+" validation C selection", run["val_macro_f1"]==max(t["val_macro_f1"] for t in run["candidates"]))
        val_pred, _ = inference(run, parts["val"], device)
        check(run["run_id"]+" restored validation score", abs(metrics(parts["val"]["y"],val_pred)["macro_f1"]-run["val_macro_f1"])<1e-12)
        pred, _ = inference(run, parts["test"], device)
        with (OUT / "predictions" / f"{run['run_id']}.csv").open(encoding="utf-8-sig") as f:
            saved = list(csv.DictReader(f))
        check(run["run_id"]+" restored predictions", np.array_equal(pred,np.array([int(r["pred_label"])-1 for r in saved])))
        score = metrics(parts["test"]["y"],pred)
        check(run["run_id"]+" restored metrics", abs(score["macro_f1"]-run["macro_f1"])<1e-12 and abs(score["accuracy"]-run["accuracy"])<1e-12)
    write_json(OUT / "validation.json", {"passed": True, "checked_utc": utc_now(), "checks": checks, "n_checks": len(checks)})
    print(f"PASS: {len(checks)} checks; all saved models reproduce validation and test predictions.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "validate"])
    args = parser.parse_args()
    globals()[args.command]()
